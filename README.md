# RabindricGPT

Do tokenizer rankings hold with scale? Small-scale language modeling of Bengali with adaptation to Tagore-era text.

## Research Question

For byte, Unicode-character, akshara (grapheme-cluster), BPE, and Unigram tokenizers (plus BPE-over-aksharas), how do bits-per-character (BPC), bits-per-byte (BPB), and generation quality change across model sizes (~10M, 25M, 50M, and ~100M parameters), and how quickly does each adapt to Rabindranath Tagore's literary works after pretraining on general Bengali? All tokenizers are evaluated under an equal-bytes protocol, testing whether subword representation advantages persist as compute and parameter scale increase, and how architectural compression interacts with domain adaptation to historical text.

## Setup and Run

Code implementation is currently in development.

```bash
# Setup instructions (placeholder)
git clone https://github.com/ASMITSARKAR/RabindricGPT.git
cd RabindricGPT
# Create virtual environment and install dependencies
python -m venv .venv
# On Windows: .venv\Scripts\activate; On Unix: source .venv/bin/activate
pip install -r requirements.txt  # Placeholder
```

```bash
# Pipeline execution (placeholder)
# 1. Train tokenizers
# 2. Run pretraining ladder (~10M, 25M, 50M)
# 3. Fine-tune / adapt on Tagore works and evaluate BPC/BPB
```

## Repository Layout

```text
RabindricGPT/
├── .gitignore          # Exclusion rules for local data, docs, and checkpoints
├── README.md           # Project overview and setup instructions
├── src/                # Model architecture, tokenizers, and training engine (planned)
├── scripts/            # Ingestion, cleaning, sharding, and evaluation scripts (planned)
└── configs/            # Experiment and model configuration files (planned)
```

## Data

No text corpora or datasets are included in this repository. All pretraining and adaptation data must be obtained directly from the original sources under their respective licensing terms:
- General Bengali pretraining corpus: FineWeb-2 (`ben_Beng`), available under the Open Data Commons Attribution License (`ODC-By`).
- Historical adaptation corpus: Bengali Wikisource (`bn.wikisource.org`), available under Creative Commons Attribution-ShareAlike 4.0 International (`CC BY-SA 4.0`).

## Tools

Parts of the code were written with AI coding assistance (Claude and Gemini in Google Antigravity); the research design, experiments and analysis are the author's own and were reviewed by the author.
