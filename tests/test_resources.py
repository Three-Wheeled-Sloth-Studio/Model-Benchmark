from pathlib import Path
from types import SimpleNamespace

from model_benchmark import resources


def test_nvidia_smi_falls_back_to_standard_windows_location(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resources.shutil, "which", lambda _name: None)
    monkeypatch.setattr(resources, "_where_nvidia_smi", lambda: None)
    executable = tmp_path / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("stub")

    found = resources.discover_nvidia_smi(
        platform_name="nt", environ={"ProgramFiles": str(tmp_path)}
    )

    assert found == str(executable)


def test_windows_candidates_include_canonical_system32_without_environment() -> None:
    candidates = resources._windows_nvidia_smi_candidates({})

    assert Path(r"C:\Windows\System32\nvidia-smi.exe") in candidates


def test_nvidia_smi_uses_where_exe_when_python_path_lookup_misses(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resources.shutil, "which", lambda _name: None)
    executable = tmp_path / "nvidia-smi.exe"
    executable.write_text("stub")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=f"{executable}\n")

    monkeypatch.setattr(resources.subprocess, "run", fake_run)

    found = resources.discover_nvidia_smi(platform_name="nt", environ={})

    assert found == str(executable)


def test_ollama_rss_only_reads_memory_for_matching_processes(monkeypatch) -> None:
    calls: list[str] = []

    class FakeProcess:
        def __init__(self, name: str, rss: int) -> None:
            self.info = {"name": name}
            self._rss = rss

        def memory_info(self):
            calls.append(self.info["name"])
            return SimpleNamespace(rss=self._rss)

    processes = [FakeProcess("chrome.exe", 999), FakeProcess("ollama.exe", 123)]

    def fake_process_iter(attrs):
        assert attrs == ["name"]
        return processes

    monkeypatch.setattr(resources.psutil, "process_iter", fake_process_iter)
    collector = resources.ResourceCollector(gpu_telemetry=False)

    assert collector._ollama_rss(10.0) == 123
    assert calls == ["ollama.exe"]

    # Cached process identities avoid another full process-table scan on the next sample.
    assert collector._ollama_rss(11.0) == 123
    assert calls == ["ollama.exe", "ollama.exe"]
