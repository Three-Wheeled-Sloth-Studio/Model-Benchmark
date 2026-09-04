# Baseline benchmark contract

## Purpose

Model Benchmark qualifies local Ollama tags for inexpensive, repeatable structured knowledge-work. It is intentionally not a general intelligence leaderboard.

## Headline score

The public headline is a 0-100 `composite_score`:

- 70% quality
- 30% operational fitness

The unpenalized value is retained as `composite_score_raw`. Safety/usability conditions may cap the headline value.

### Quality

Quality is initially composed of:

- 35% keyword quality
- 25% semantic parsing quality
- 20% relative ranking/reasoning quality
- 20% adherence

Keyword extraction is deliberately precision-sensitive. Synthetic cases contain high-frequency business filler and attractive-but-useless generic terms. F1 establishes the base accuracy and chaff rate applies an additional penalty.

Ranking cases score pairwise order rather than exact numerical agreement. Reasons receive credit for citing supplied evidence and lose credit for unsupported claims.

### Operational fitness

Operational fitness is initially:

- 40% median warm generation throughput
- 30% cold-load cost
- 30% minimum system-memory headroom

NVIDIA VRAM and GPU utilization are telemetry, not primary abort conditions. Full VRAM can be valid when system RAM remains healthy.

## Hard usability treatment

- Resource abort: composite capped at 20.
- Whole-model timeout: composite capped at 40.
- Individual test timeout: composite capped at 60.
- Severe memory pressure below 1.25x the configured RAM floor: composite capped at 70.
- More than 25% of suite cases incomplete: composite capped at 50.

These caps prevent a brilliant but workstation-hostile model from winning the recommendation.

## Reproducibility

Baseline generation uses temperature 0, fixed seed 42, a 4096-token context, bounded output, identical prompts, and one loaded candidate at a time. Every Ollama tag is a separate candidate.

Benchmark suite and result schemas are versioned. Material scoring changes require a suite or scoring-contract version change before historical comparisons are treated as equivalent.
