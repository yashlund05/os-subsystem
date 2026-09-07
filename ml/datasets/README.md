# Dataset Hygiene, Licensing & Reproduction Notice

## STRICT REPOSITORY SAFETY RULE: DO NOT COMMIT RAW TRACES OR DATASETS

This repository adheres to strict open-source licensing, privacy, and artifact hygiene standards.

### 1. Licensing & Distribution Restrictions
- **Google Borg Cluster Traces**: The Google cluster traces are distributed by Google under separate licensing agreements and terms of use. Raw trace files (`.csv.gz`, `.parquet`, `.tar.gz`) must **NEVER** be committed to this Git repository.
- **SPEC CPU2017 & MiBench**: Standard Performance Evaluation Corporation (SPEC) benchmarks are proprietary commercial software subject to strict non-redistribution licenses. Raw binaries, benchmark sources, and proprietary runtime profiles must **NEVER** be committed to this repository.

### 2. Dataset Ingestion Guidelines
- Store downloaded traces locally outside of Git tracking (e.g., in a local `data/` directory or under an environment path configured via `.env`):
  ```bash
  GOOGLE_BORG_TRACE_PATH=/path/to/local/google_traces/
  SPEC_CPU2017_TRACE_PATH=/path/to/local/spec_traces/
  ```
- Use the automated download and ingestion scripts located in `scripts/setup/` to fetch public trace slices or generate synthetic workloads.
- The repository `.gitignore` strictly excludes all `.parquet`, `.csv.gz`, `.trace`, and `.bin` data artifacts.
