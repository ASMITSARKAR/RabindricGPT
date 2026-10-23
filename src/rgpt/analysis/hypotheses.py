"""
src/rgpt/analysis/hypotheses.py : Automated evaluation of research hypotheses H1–H6 based on formal decision criteria.
Backlog/Phase : I11 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; 01_CHARTER.md
Contract      : results: dict -> hypothesis_verdicts: dict[str, dict]
Invariants    : Evaluates all hypotheses H1 through H6 strictly per charter formulas; logs confirm/refute
Failure modes : Missing evaluation metric for required model size; ill-formed result dict
Status        : TESTED
"""

from typing import Any, Dict


def evaluate_hypotheses(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Evaluate hypotheses H1–H6 against results dictionary.

    Hypotheses (01_CHARTER.md §1):
        - H1: At fixed model size (10M), subword BPC lower than byte/char/akshara.
        - H2: BPB gap between tokenizers shrinks as model size grows.
        - H3: Akshara tokens beat Unicode-char tokens in BPC at equal compute.
        - H4: Shorter-sequence tokenizers beat long-sequence ones more under equal-compute than equal-bytes.
        - H5: Best pretrain BPC tokenizer also adapts fastest to Tagore text.
        - H6: Subwords produce fewer ill-formed Unicode outputs than byte models.

    Args:
        results: Structured dictionary containing metrics across tokenizers, sizes, and runs.

    Returns:
        Dictionary mapping hypothesis ID to verdict ('CONFIRMED' / 'REFUTED' / 'INSUFFICIENT_DATA') and delta/reason.
    """
    verdicts: Dict[str, Dict[str, Any]] = {}

    # H1: Subword vs byte/char/akshara at 10M
    h1_data = results.get("10m_bpc", {})
    if h1_data:
        subwords = [h1_data[k] for k in ["bpe16", "uni16", "bpe8"] if k in h1_data]
        others = [h1_data[k] for k in ["byte", "char", "akshara"] if k in h1_data]

        if subwords and others:
            best_subword = min(subwords)
            best_other = min(others)
            confirmed = best_subword < best_other
            verdicts["H1"] = {
                "verdict": "CONFIRMED" if confirmed else "REFUTED",
                "best_subword_bpc": best_subword,
                "best_other_bpc": best_other,
                "delta": best_other - best_subword,
                "reason": (
                    f"Subword BPC ({best_subword:.4f}) {'<' if confirmed else '>='} "
                    f"non-subword BPC ({best_other:.4f}) at 10M"
                ),
            }
        else:
            verdicts["H1"] = {"verdict": "INSUFFICIENT_DATA", "reason": "Missing tokenizer data at 10M"}
    else:
        verdicts["H1"] = {"verdict": "INSUFFICIENT_DATA", "reason": "No 10M BPC results found"}

    # H2: BPB gap shrinks with scale
    bpb_by_size = results.get("bpb_by_size", {})  # e.g. {"10m": {"bpe16": 1.2, ...}, "50m": {...}}
    if len(bpb_by_size) >= 2:
        sizes = sorted(bpb_by_size.keys(), key=lambda s: int(s.replace("m", "")))
        first_size = sizes[0]
        last_size = sizes[-1]
        first_vals = list(bpb_by_size[first_size].values())
        last_vals = list(bpb_by_size[last_size].values())

        gap_first = max(first_vals) - min(first_vals)
        gap_last = max(last_vals) - min(last_vals)
        confirmed = gap_last < gap_first
        verdicts["H2"] = {
            "verdict": "CONFIRMED" if confirmed else "REFUTED",
            "gap_10m": gap_first,
            f"gap_{last_size}": gap_last,
            "delta": gap_first - gap_last,
            "reason": (
                f"BPB gap shrank from {gap_first:.4f} at {first_size} to {gap_last:.4f} at {last_size}"
                if confirmed else
                f"BPB gap did not shrink ({gap_first:.4f} -> {gap_last:.4f})"
            ),
        }
    else:
        verdicts["H2"] = {"verdict": "INSUFFICIENT_DATA", "reason": "Requires at least 2 model sizes"}

    # H3: Akshara beats char across sizes
    akshara_vs_char = results.get("akshara_vs_char_bpc", {})  # size -> (akshara_bpc, char_bpc)
    if akshara_vs_char:
        losses = 0
        total = 0
        for size, pair in akshara_vs_char.items():
            akshara_bpc, char_bpc = pair
            total += 1
            if akshara_bpc >= char_bpc:
                losses += 1
        # Refuted if not lower at 2+ sizes
        refuted = losses >= 2
        verdicts["H3"] = {
            "verdict": "REFUTED" if refuted else "CONFIRMED",
            "evaluated_sizes": list(akshara_vs_char.keys()),
            "akshara_worse_count": losses,
            "reason": f"Akshara failed to beat char in {losses}/{total} evaluated sizes",
        }
    else:
        verdicts["H3"] = {"verdict": "INSUFFICIENT_DATA", "reason": "No akshara vs char data provided"}

    # H4: Equal-compute vs equal-bytes ranking difference
    equal_bytes_rank = results.get("equal_bytes_ranking")  # list of tokenizer names
    equal_compute_rank = results.get("equal_compute_ranking")  # list of tokenizer names
    if equal_bytes_rank and equal_compute_rank:
        differs = equal_bytes_rank != equal_compute_rank
        verdicts["H4"] = {
            "verdict": "CONFIRMED" if differs else "REFUTED",
            "equal_bytes_ranking": equal_bytes_rank,
            "equal_compute_ranking": equal_compute_rank,
            "reason": (
                "Ranking changed between equal-bytes and equal-compute"
                if differs else
                "Ranking remained identical under equal-bytes and equal-compute"
            ),
        }
    else:
        verdicts["H4"] = {"verdict": "INSUFFICIENT_DATA", "reason": "Missing equal-compute or equal-bytes rankings"}

    # H5: Best pretrain BPC adapts fastest
    best_pretrain = results.get("best_pretrain_tokenizer")
    fastest_adapt = results.get("fastest_adapt_tokenizer")
    if best_pretrain and fastest_adapt:
        confirmed = best_pretrain == fastest_adapt
        verdicts["H5"] = {
            "verdict": "CONFIRMED" if confirmed else "REFUTED",
            "best_pretrain_tokenizer": best_pretrain,
            "fastest_adapt_tokenizer": fastest_adapt,
            "reason": (
                f"Same tokenizer ({best_pretrain}) achieved best pretrain BPC and fastest adaptation"
                if confirmed else
                f"Best pretrain ({best_pretrain}) differed from fastest adapt ({fastest_adapt})"
            ),
        }
    else:
        verdicts["H5"] = {"verdict": "INSUFFICIENT_DATA", "reason": "Missing pretrain or adaptation winners"}

    # H6: Byte produces more ill-formed Unicode than subword
    byte_ill_rate = results.get("byte_ill_formed_rate")
    subword_ill_rate = results.get("subword_ill_formed_rate")
    if byte_ill_rate is not None and subword_ill_rate is not None:
        confirmed = byte_ill_rate > subword_ill_rate
        verdicts["H6"] = {
            "verdict": "CONFIRMED" if confirmed else "REFUTED",
            "byte_ill_formed_rate": byte_ill_rate,
            "subword_ill_formed_rate": subword_ill_rate,
            "reason": (
                f"Byte error rate ({byte_ill_rate:.4f}) > subword error rate ({subword_ill_rate:.4f})"
                if confirmed else
                f"Byte error rate ({byte_ill_rate:.4f}) <= subword error rate ({subword_ill_rate:.4f})"
            ),
        }
    else:
        verdicts["H6"] = {"verdict": "INSUFFICIENT_DATA", "reason": "Missing Unicode corruption rates"}

    return verdicts
