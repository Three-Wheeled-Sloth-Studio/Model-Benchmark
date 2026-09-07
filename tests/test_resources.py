from types import SimpleNamespace

from model_benchmark import resources


def test_nvidia_smi_falls_back_to_standard_windows_location(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resources.shutil, "which", lambda _name: None)
    executable = tmp_path / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("stub")

    found = resources.discover_nvidia_smi(
        platform_name="nt", environ={"ProgramFiles": str(tmp_path)}
    )

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
