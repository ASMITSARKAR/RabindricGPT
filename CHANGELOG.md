# Changelog

All notable changes to the RabindricGPT project are documented in this file.

## [v0.2.1-audit-corrections] - 2026-10-24

### Retracted
- **Formal Retraction of Round 7 FLOPs Table**: The FLOPs table presented in Round 7 is hereby formally retracted. It erroneously mapped `10m` as $L=8, d=512$ (the 25M configuration), `25m` as $L=10, d=640$ (the 50M configuration), and `50m` as $L=12, d=768$ (not in ladder), and used outdated vocabulary sizes ($V=256$ for byte instead of 264; $V=2048$ for aks instead of 8192). All 24 cells have been restored to exact instantiated model ladder architectures (`10m`: $L=6, d=384, N_{\text{ne}}=10,621,824$; `25m`: $L=8, d=512, N_{\text{ne}}=25,698,816$; `50m`: $L=10, d=640, N_{\text{ne}}=49,575,040$; `100m`: $L=14, d=768, N_{\text{ne}}=99,112,704$) with exact hand-computed constants for all 24 cells in `tests/compute/test_flops.py` and `results/audits/audit_flops.txt`.

### Fixed
- **Equal-Bytes Compute Unit Accounting**: Replaced invented token scaling $D_{\text{tok}} = D_{\text{plan}} \cdot s_{\text{meas}}/s_{\text{plan}}$ with exact equal-bytes formula $D_{\text{tok}} = D_{\text{bytes}} / T_{\text{meas}}$, where $T_{\text{meas}} = \text{measured bytes per window} / s_{\text{meas}}$. Pretraining cells are evaluated on FineWeb-2 shards (2,701.33 B/win) and adaptation cells on Tagore shards (2,693.86 B/win). Byte cell units correctly reflect $D_{\text{tok}} = D_{\text{bytes}}$ independent of sequence length $s$.
- **Sequence Length Drift Gate (5%)**: Explicitly codified the 5% sequence length drift gate in `tests/compute/test_flops.py` and `scripts/audits/audit_flops.py`. On Tagore shards, char (+0.04%), aks (+4.99%), and bpe8 (+3.71%) pass $\le 5\%$, while byte (-12.31%), bpe16 (+15.51%), and uni16 (+14.66%) are detected and flagged.
- **Narrow Filter Compound Keyword**: Removed bare `"ডাকঘর"` (common noun) and retained compound `"নাটক ডাকঘর"`. Enforced NFC-normalization and stripped ZWNJ/ZWJ before keyword matching.
- **DR-4a Mathematical Rationale**: Replaced false self-token collapse claims with exact second-order Taylor expansion $\mathbb{E}[\mathcal{L}] \approx \ln V + \operatorname{Var}(z)/2 - \mathbb{E}[z_{\text{target}}]$. Under RMSNorm, $\|h_L\|^2 = d$, yielding $\operatorname{Var}(z) = d \cdot \sigma_E^2 = 0.35^2 = 0.1225$, so $\operatorname{Var}(z)/2 = 0.06125$. For next-token prediction at init, $\mathbb{E}[z_{\text{target}}] \approx 0$, explaining why initial loss sits $+0.059$ above $\ln V$.

## [v0.2.0-cleanup] - 2026-10-24

### Added
- **Maintained Audits Framework (`scripts/audits/`)**:
  - `audit_splits.py`: Computes whole-work splits, genre distribution, and exact clean character counts from `pages.jsonl`.
  - `audit_flops.py`: Verifies analytical FLOPs formula across 20 tokenizer $\times$ scale pairs against budget doc 08 anchors.
  - `audit_reconcile.py`: Validates shard `index.json` reconciliation, `encode()` consistency, token counts, and file integrity.
  - `audit_filter_provenance.py`: Documents broad vs narrow Tagore-exclusion filter provenance and FineWeb-2 sample distribution.
  - `audit_windows_gora.py`: Traces train windows 0–99 to source works/pages, computes consecutive 100-window compression profiles of Gora, and reconciles the 10-work compression table across all 6 tokenizers.
- **Pinned Audit Results (`results/audits/`)**:
  - `audit_splits.txt`, `audit_flops.txt`, `audit_reconcile.txt`, `audit_provenance.txt`, `audit_windows_gora.txt` (including git commit SHA and input file SHA-256 hashes).
- **Test Suite Extensions**:
  - `tests/data/test_splits.py`: Added `test_clean_chars_measured_matches_pages_jsonl` to recompute clean character counts from `pages.jsonl` and fail on mismatch.
  - `tests/compute/test_flops.py`: Automated assertion of FLOPs deviation $\le 0.9935\% < 1.000\%$ across all 20 configurations.
  - `tests/tokenizers/test_reconcile_audit.py`: Validates shard reconciliation and round-trip decodability.
  - `tests/data/test_filter_provenance.py`: Validates filter rule provenance and zero false positives.
  - `tests/tokenizers/test_windows_gora_audit.py`: Validates window mapping and Gora compression stability ($\ge 3.50$ chars/token).
  - `tests/model/test_init_dr4.py`: Comprehensive 20-test suite verifying embedding initialization scale $\sigma_E = 0.35 / \sqrt{d}$.
  - `tests/train/test_microbatch_equality.py`: Micro-batch gradient equality across accumulation steps.

### Changed
- **Level P3 Redefinition (`src/rgpt/preflight/e2e.py`)**:
  - Fixed P3 evaluation: compares chars/token on a held-out slice of `fw2_tokenizer_sample_100m.txt` (in-sample) against FineWeb-2 `egen.txt`, failing if $|ratio - 1| > 0.08$.
  - Added per-tokenizer minimum assertion on held-out Tagore validation set: fails if `bpe16 < 2.5` chars/token.
- **SentencePiece Wrapper (`src/rgpt/tokenizers/spm.py`)**:
  - Restored `hard_vocab_limit=True` default for production tokenizer training.
  - Restricted `hard_vocab_limit=False` exclusively to toy unit tests.
  - Added strict assertion `piece_count == requested_vocab_size`.
  - Added training run manifest generation (`*_manifest.json`) recording library version, model file SHA-256, and lines dropped by `max_sentence_length`.
- **Model Configuration & Initialization**:
  - Replaced string with `embedding_init_scale: 0.35` (float) in `configs/decisions.yaml`, `DecisionsConfig`, `ModelConfig`, and model implementation `src/rgpt/model/gpt.py`.
  - Updated $d$-set in `configs/decisions.yaml` DR-4a documentation to $\{384, 512, 576, 640, 768\}$.
  - Documented calibration rationale: calibrated against uniform categorical cross-entropy $\ln(V)$ under weight tying where logit variance is $d \cdot \sigma_E^2$, yielding $\sigma_{\text{logit}} \approx 0.35$.
- **Data Splits & Shard Index**:
  - Rebuilt all shards across 6 tokenizers and 3 splits (`train`: 1,573 chunks, `val`: 162 chunks, `test`: 464 chunks).
  - Updated `clean_chars_measured` in `configs/works_wikisource.yaml` to empirical character counts (sum: 2,246,333 clean chars).
  - Recorded Gitanjali `val` $\to$ `test` relocation in `configs/decisions.yaml` under DR-10 amendment (Date: 2026-10-23).
- **Preflight CLI (`scripts/00_preflight.py`)**:
  - Added explicit per-level print line for Level P4 invariant and fingerprint verifications.

### Removed
- **Voided Artifacts**: Purged invalidated experimental result files from `results/` (`eval_PT-*.json`, `fits.json`, `hypotheses.json`, `pilot.json`, `g1_audit.*`, `g2_report.json`, `ledger.md`, `main_summary_table.md`, `scaling_curves.png`).
