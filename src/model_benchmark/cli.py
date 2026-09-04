from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from model_benchmark.config import BenchmarkConfig
from model_benchmark.fixtures import load_suite
from model_benchmark.ollama import OllamaClient, OllamaError, skip_reason
from model_benchmark.reports import append_jsonl, write_reports
from model_benchmark.resources import ResourceCollector
from model_benchmark.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-benchmark",
        description="Qualify installed local Ollama models for structured knowledge-work tasks.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run the baseline benchmark")
    run.add_argument("--model", action="append", default=[], help="glob pattern; may be repeated")
    run.add_argument("--output-dir", default="benchmark-results")
    run.add_argument("--suite", default=None, help="path to a custom benchmark suite JSON")
    run.add_argument("--base-url", default="http://127.0.0.1:11434")
    run.add_argument("--model-timeout-seconds", type=float, default=600.0)
    run.add_argument("--test-timeout-seconds", type=float, default=300.0)
    run.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    run.add_argument("--min-available-ram-gb", type=float, default=8.0)
    run.add_argument("--context-length", type=int, default=4096)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-output-tokens", type=int, default=384)
    run.add_argument("--sample-interval-seconds", type=float, default=0.5)
    run.add_argument("--no-gpu-telemetry", action="store_true")
    run.add_argument(
        "--stop-started-ollama",
        action="store_true",
        help="stop Ollama after the run only when this harness started it",
    )

    subparsers.add_parser("list", help="list installed models and baseline eligibility")
    subparsers.add_parser("doctor", help="check Ollama and local telemetry prerequisites")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["run"]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "list":
            return _list()
        if args.command == "doctor":
            return _doctor()
    except (OllamaError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    parser.print_help()
    return 2


def _run(args: argparse.Namespace) -> int:
    if args.min_available_ram_gb <= 0:
        raise ValueError("--min-available-ram-gb must be positive")
    if args.test_timeout_seconds <= 0 or args.model_timeout_seconds <= 0:
        raise ValueError("timeouts must be positive")
    if args.context_length <= 0 or args.max_output_tokens <= 0:
        raise ValueError("context and output token limits must be positive")
    if args.sample_interval_seconds <= 0:
        raise ValueError("--sample-interval-seconds must be positive")
    config = BenchmarkConfig(
        base_url=args.base_url,
        output_dir=args.output_dir,
        model_timeout_seconds=args.model_timeout_seconds,
        test_timeout_seconds=args.test_timeout_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        min_available_ram_gb=args.min_available_ram_gb,
        context_length=args.context_length,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        sample_interval_seconds=args.sample_interval_seconds,
        stop_started_ollama=args.stop_started_ollama,
        gpu_telemetry=not args.no_gpu_telemetry,
        model_patterns=args.model,
        suite_path=args.suite,
    )
    suite = load_suite(args.suite)
    runner = BenchmarkRunner(config, suite)
    results: list[dict[str, object]] = []
    try:
        models = runner.prepare()
        metadata = runner.metadata(models)
        print(
            f"Run {runner.run_id}: {len(models)} eligible model(s), "
            f"{len(runner.skipped_models)} skipped. Results: {runner.run_dir}"
        )
        if not models:
            write_reports(runner.run_dir, metadata, [])
            print("No eligible installed local text/chat models were found.")
            return 1
        for index, model in enumerate(models, 1):
            print(f"\n=== Model {index}/{len(models)}: {model.name} ===")
            result = runner.run_model(model)
            results.append(result)
            append_jsonl(runner.run_dir / "results.jsonl", result)
        metadata["finished_at"] = datetime.now(UTC).isoformat()
        summary = write_reports(runner.run_dir, metadata, results)
        print("\n=== Final ranking ===")
        for item in summary["rankings"]:
            print(
                f"{item['rank']:>2}. {item['model']:<35} "
                f"composite={item['composite_score']:>5.1f} "
                f"quality={item['quality_score']:>5.1f} "
                f"operational={item['operational_score']:>5.1f} "
                f"status={item['status']}"
            )
        print(f"\nSummary: {runner.run_dir / 'summary.md'}")
        return 0 if any(item["status"] == "succeeded" for item in results) else 1
    finally:
        runner.cleanup()


def _list() -> int:
    client = OllamaClient()
    temp_log = Path("benchmark-results") / "ollama-list-serve.log"
    client.ensure_running(temp_log, 30)
    models = client.list_models()
    if not models:
        print("No installed Ollama models found.")
        return 1
    for model in models:
        reason = skip_reason(model)
        state = f"SKIP: {reason}" if reason else "ELIGIBLE"
        print(
            f"{model.name:<38} {model.parameter_size or '':>8} "
            f"{model.quantization_level or '':>10}  {state}"
        )
    return 0


def _doctor() -> int:
    client = OllamaClient()
    collector = ResourceCollector(gpu_telemetry=True)
    available_before = client.api_available()
    print(f"Ollama executable: {client.ollama_path or 'NOT FOUND'}")
    print(f"Ollama API reachable: {'yes' if available_before else 'no'}")
    print(f"nvidia-smi available: {'yes' if collector.nvidia_smi else 'no'}")
    snapshot = collector.hardware_snapshot()
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    if not client.ollama_path and not available_before:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
