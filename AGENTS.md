# Agent guidance

Model Benchmark is a small standalone qualification harness. Keep it deterministic, local-first, and safe to run on a workstation.

## Core boundaries
- Required tests and CI must never require Ollama, a GPU, or network access.
- Real model runs are manual/local integration tests.
- Benchmark fixtures are synthetic and safe for a public repository.
- Never add cloud-model execution as an implicit fallback. Local Ollama models are the target.
- Preserve result schema compatibility when possible. Version benchmark suites and material scoring changes.
- Resource safety beats benchmark completeness. A model that threatens host responsiveness must be stopped and recorded as such.

## Validation
See `refs/testing/validationCommands.yaml`.
