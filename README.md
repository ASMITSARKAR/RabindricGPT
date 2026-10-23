# RabindricGPT

Do tokenizer rankings hold with scale? Small-scale language modeling of Bengali with adaptation to Tagore-era text.

## Research Question

For byte, Unicode-character, akshara (grapheme-cluster), BPE, and Unigram tokenizers (plus BPE-over-aksharas), how do bits-per-character (BPC), bits-per-byte (BPB), and generation quality change across model sizes (~10M, 25M, 50M, and ~100M parameters), and how quickly does each adapt to Rabindranath Tagore's literary works after pretraining on general Bengali? All tokenizers are evaluated under an equal-bytes protocol, testing whether subword representation advantages persist as compute and parameter scale increase, and how architectural compression interacts with domain adaptation to historical text.

## Setup and Quickstart

```bash
# Clone the repository
git clone https://github.com/ASMITSARKAR/RabindricGPT.git
cd RabindricGPT

# Create virtual environment and install dependencies
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix:
# source .venv/bin/activate

pip install -r requirements.txt

# Run full local preflight suite (P0–P4: env, static, all unit tests, mini e2e)
python scripts/00_preflight.py --full
```

## RUNBOOK: End-to-End Pipeline

Follow these exact steps to run each phase of the project:

### 1. Ingestion
```bash
# Ingest Rabindranath Tagore works from Bengali Wikisource
python scripts/01_ingest_wikisource.py

# Ingest general Bengali pretraining corpus from FineWeb-2
python scripts/02_ingest_fineweb2.py
```

### 2. Cleaning, Deduplication, and Splitting
```bash
# Normalize Unicode, clean markup, MinHash deduplicate, whole-work split
python scripts/03_clean_and_split.py
```

### 3. Gate G1: Data Audit
```bash
# Audit clean data, verify bytes/char (2.58), Bengali ratio (>=80%), and PUA absence
python scripts/04_audit.py
```

### 4. Tokenizer Training and Gate G2
```bash
# Train all 6 tokenizers (byte, char, akshara, BPE 16k, Unigram 16k, BPE 8k)
python scripts/05_train_tokenizers.py

# Verify roundtrip losslessness, PUA safety, and generate results/g2_report.json
python scripts/06_tokenizer_tests.py
```

### 5. Chunking and Sharding
```bash
# Build 1024-character windows into uint16 memmap binary shards under equal-bytes protocol
python scripts/07_build_shards.py
```

### 6. Model Architecture & Parameter Verification
```bash
# Verify analytic vs empirical parameters across all 4 sizes and 6 tokenizers
python scripts/08_model_param_report.py
```

### 7. Run Matrix & Planning
```bash
# Generate canonical run matrix (results/run_matrix.csv) and compute budgets
python scripts/10_make_run_matrix.py
```

### 8. Single-Run Training
```bash
# Run any planned experiment from the matrix (e.g. 10M BPE16 seed 0)
python scripts/11_run.py --run-id PT-bpe16-10m-s0

# Re-running finished runs is strictly idempotent (exits 0 immediately)
python scripts/11_run.py --run-id PT-bpe16-10m-s0
```

### 9. Final Evaluation & Hypotheses
```bash
# Compute BPC, BPB, bootstrap CIs, and generation quality on test split
python scripts/12_final_eval.py --run-id PT-bpe16-10m-s0

# Fit power laws, test hypotheses H1–H6, generate figures and summary tables
python scripts/13_fit_and_plot.py

# Aggregate run spend into results/ledger.md
python scripts/14_update_ledger.py
```

## Repository Layout

```text
RabindricGPT/
├── .github/workflows/  # CI workflows for preflight and unit tests
├── colab/              # Colab runner notebook and supervisor library
├── configs/            # YAML configs: models, tokenizers, decisions, GPU settings
├── data/               # Ingested raw corpora, cleaned text, and binary shards
├── docs/               # Architecture, theory, and experimental plans (read-only)
├── results/            # Deterministic, committed outputs, reports, and tables
├── scripts/            # 16 numbered CLI pipeline drivers (00 to 14)
├── src/rgpt/           # Core library
│   ├── common/         # Atomic IO, hashing, paths, JSON logging
│   ├── text/           # Bengali Unicode normalization, cleaning, regexes
│   ├── akshara/        # Linguistic grapheme-cluster segmenter
│   ├── tokenizers/     # 6 tokenizer implementations and intrinsic tests
│   ├── data/           # Ingest, cleaning, splits, dedup, chunks, shards
│   ├── model/          # Decoder-only transformer, RoPE, RMSNorm, SwiGLU, GPT
│   ├── train/          # DataLoader, scheduler, checkpointing, throughput, loop
│   ├── eval/           # BPC, bootstrap CIs, nucleus sampling, humaneval
│   ├── analysis/       # Power-law fitting, hypothesis testing, plotting, tables
│   ├── compute/        # FLOPs formulas, ledger aggregator, run matrix
│   └── preflight/      # Local preflight harness (P0 to P6)
└── tests/              # Pytest suite mirroring every package and gate
```

## Data and Licenses

No text corpora or datasets are included in this repository. All pretraining and adaptation data must be obtained directly from the original sources under their respective licensing terms:
- General Bengali pretraining corpus: FineWeb-2 (`ben_Beng`), available under the Open Data Commons Attribution License (`ODC-By`).
- Historical adaptation corpus: Bengali Wikisource (`bn.wikisource.org`), available under Creative Commons Attribution-ShareAlike 4.0 International (`CC BY-SA 4.0`).

## Tools

Parts of the code were written with AI coding assistance (Claude and Gemini in Google Antigravity); the research design, experiments and analysis are the author's own and were reviewed by the author.
