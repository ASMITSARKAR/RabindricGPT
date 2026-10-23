# Superseded and Voided Results Archive

This directory (`results/_void/`) contains experimental evaluations, preliminary pilot runs, and scaling fit artifacts generated prior to the Round 7-8 tokenizer retraining, shard rebuilding, and initialization corrections.

## Reason for Voiding
1. **Tokenizer Retraining:** Tokenizers in `tok/` were retrained with strict clean-corpus constraints and correct vocab limits, rendering previous token-level losses and evaluation outputs incomparable.
2. **Shard Reconstruction:** Shards in `shards/wikisource/` were rebuilt from corrected whole-work assignments and verified clean texts.
3. **DR-4a Embedding Init Scale:** Models run prior to commit `24b809c` used uncalibrated standard normal or flat initialization scales rather than $\sigma_E = 0.35/\sqrt{d}$, creating initialization-dependent loss shifts.

## Contents
- `eval_PT-*.json`: Superseded model evaluation metrics from early smoke/pilot runs.
- `pilot.json`, `fits.json`, `g1_audit.*`, `g2_report.json`: Pilot fit summaries and early audit logs.
- `figures/scaling_curves.png`: Scaling curve plots fitted against voided evaluation runs.
- `tables/main_summary_table.md`: Summary tables aggregating superseded evaluation runs.

*Note: Pre-registration files (`results/hypotheses.json` H1–H6 and `results/ledger.md`) remain active in `results/` as permanent project specifications.*
