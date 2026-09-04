# Model Benchmark

Model Benchmark is a small standalone CLI for qualifying installed local Ollama models for structured knowledge-work tasks. It is intentionally separate from Nerve Center so other tools can consume the same benchmark artifacts later.

The baseline answers one practical question: **what is the best model quality this workstation can deliver without making the workstation miserable to use?**

## What it measures

The v1 suite emphasizes quality over raw speed:

- meaningful keyword extraction with aggressive penalties for filler/chaff;
- semantic parsing into fixed structured schemas;
- relative quality/fit ranking with evidence-grounded reasoning;
- instruction and schema adherence;
- cold model load time and representative warm performance;
- token throughput and Ollama prompt/generation timings;
- system RAM, Ollama RSS, CPU, and optional NVIDIA VRAM/GPU utilization;
- timeouts and resource-safety aborts.

The headline composite is **70% quality / 30% operational fitness**, with hard caps when a model times out or threatens the configured memory safety floor. All component scores remain in the output so downstream consumers can use different weights.

## Install

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Ollama must already be installed, but it does not need to be running. The CLI starts `ollama serve` when the local API is unavailable.

## Run

Benchmark all eligible installed local text/chat models:

```powershell
model-benchmark run
```

Or without activating the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m model_benchmark run
```

Useful commands:

```powershell
model-benchmark doctor
model-benchmark list
model-benchmark run --model "qwen*"
model-benchmark run --min-available-ram-gb 12
model-benchmark run --model-timeout-seconds 900 --test-timeout-seconds 420
```

Defaults:

- 10 minutes maximum per model;
- 5 minutes maximum per individual test;
- 8 GB minimum available system RAM;
- temperature 0;
- fixed seed 42;
- 4096-token context;
- bounded output;
- one model loaded at a time.

The harness explicitly unloads a candidate, performs a tiny cold probe, runs the synthetic suite while warm, repeats one representative case as a warm probe, and unloads the model before proceeding. Every installed Ollama tag is treated as a distinct candidate. Embedding/reranking-only and cloud-only tags are skipped with a recorded reason.

## Results

Each invocation creates `benchmark-results/<run-id>/` containing:

- `results.jsonl` - canonical append-friendly record, one complete model result per line;
- `summary.json` - pretty ranked summary intended for downstream consumers such as Nerve Center;
- `summary.md` - human-readable report;
- `run-metadata.yaml` - benchmark configuration, machine snapshot, skipped models, and run metadata;
- `ollama-serve.log` - only when the harness had to start Ollama.

Benchmark outputs are ignored by Git by default.

## Resource protection

While each Ollama request runs, the harness samples host resources. If available system RAM falls below the configured floor, it calls `ollama stop <model>`, marks the candidate `resource_abort`, and does not pretend the run succeeded. A per-test timeout is handled similarly.

Full VRAM use by itself is not an abort condition because Ollama can legitimately spill into system RAM. VRAM pressure is recorded; system-memory headroom is the primary safety signal.

## Benchmark philosophy

This is not an academic LLM benchmark. It is a cheap, repeatable workstation qualification test for tasks that local agents actually need. Numerical scoring answers should be directionally reasonable rather than exactly deterministic. Keyword tests deliberately contain tempting filler so a model cannot score well by returning frequent-but-useless business language.

See `refs/planning/benchmark-contract.md` for the scoring contract.
