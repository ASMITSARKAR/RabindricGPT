"""
src/rgpt/eval/humaneval.py : Exporter for blind human-evaluation sheets with anonymized keys.
Backlog/Phase : I12 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; 07_EVALUATION_PLAN.md
Contract      : samples_by_model: dict -> (eval_sheet.csv, key_file.json)
Invariants    : Sample order randomized; tokenizer identity stored in separate key file; blind evaluation
Failure modes : Leaking model identity in sample metadata
Status        : TESTED
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Union


def export_blind_sheet(
    samples_by_tokenizer: Dict[str, List[str]],
    output_dir: Union[str, Path],
    seed: int = 42,
) -> None:
    """Export randomized blind evaluation sheet and separate deciphering key."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    items = []
    for tok_name, texts in samples_by_tokenizer.items():
        for idx, text in enumerate(texts):
            items.append({
                "tokenizer": tok_name,
                "orig_idx": idx,
                "text": text,
            })

    rng = random.Random(seed)
    rng.shuffle(items)

    sheet_path = out_path / "eval_sheet.csv"
    key_path = out_path / "key_file.json"

    key_data: Dict[str, Dict[str, Union[str, int]]] = {}

    with open(sheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "text",
            "score_naturalness_1_to_5",
            "score_fluency_1_to_5",
            "score_overall_1_to_5",
            "evaluator_notes",
        ])

        for i, item in enumerate(items):
            sample_id = f"SAMP_{i+1:04d}"
            key_data[sample_id] = {
                "tokenizer": item["tokenizer"],
                "orig_idx": item["orig_idx"],
            }
            writer.writerow([
                sample_id,
                item["text"],
                "",  # empty for evaluator
                "",
                "",
                "",
            ])

    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key_data, f, indent=2)
