"""
src/rgpt/data/splits.py : Whole-work split assignment enforcement and variorum integrity checks.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §5.1, §5.3
Contract      : works: list[dict], option: str -> dict[str, list[dict]]
Invariants    : A canonical work belongs to exactly one split (train, val, or test); zero chapter leakage
Failure modes : Work assigned to multiple splits; unknown work ID; split option invalid
Status        : TESTED
"""

from typing import Any, Dict, List, Set


class SplitLeakageError(Exception):
    """Raised when the whole-work partition invariant is violated."""
    pass


def assign_work_splits(
    works: List[Dict[str, Any]],
    split_option: str = "Option_A",
) -> Dict[str, List[Dict[str, Any]]]:
    """Assign whole works to train, val, and test splits per configuration.

    Option A (Balanced Multi-Genre Holdout):
        Train: gora, galpaguchchha_1, chaturanga, sonar_tori, noukadubi, dakghar
        Val: shesher_kobita, gitanjali
        Test: jogajog, chitrangada

    Option B (Poetry & Lyric Generalization Stress Test):
        Train: gora, jogajog, galpaguchchha_1
        Val: chaturanga, noukadubi, dakghar
        Test: shesher_kobita, sonar_tori, gitanjali, chitrangada

    Option C (Novel-to-Story Parallel Benchmark):
        Train: gora, jogajog, chaturanga, sonar_tori, dakghar, noukadubi, chitrangada
        Val: shesher_kobita, gitanjali
        Test: galpaguchchha_1

    Args:
        works: List of work dictionaries.
        split_option: "Option_A", "Option_B", or "Option_C".

    Returns:
        Dictionary mapping split name ('train', 'val', 'test') to list of works.
    """
    option_map = {
        "Option_A": {
            "train": {"gora", "galpaguchchha_1", "chaturanga", "sonar_tori", "noukadubi", "dakghar"},
            "val": {"shesher_kobita"},
            "test": {"gitanjali", "jogajog", "chitrangada"},
        },
        "Option_B": {
            "train": {"gora", "jogajog", "galpaguchchha_1"},
            "val": {"chaturanga", "noukadubi", "dakghar"},
            "test": {"shesher_kobita", "sonar_tori", "gitanjali", "chitrangada"},
        },
        "Option_C": {
            "train": {"gora", "jogajog", "chaturanga", "sonar_tori", "dakghar", "noukadubi", "chitrangada"},
            "val": {"shesher_kobita", "gitanjali"},
            "test": {"galpaguchchha_1"},
        },
    }

    normalized_opt = split_option.replace("-", "_")
    if normalized_opt not in option_map:
        raise ValueError(
            f"Invalid split_option '{split_option}'. Must be one of: {list(option_map.keys())}"
        )

    mapping = option_map[normalized_opt]
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for work in works:
        wid = work.get("id")
        if not wid:
            raise ValueError(f"Work item missing mandatory 'id' field: {work}")

        # Check explicit split in work dict or look up in option map
        assigned = None
        if "split" in work and work["split"] in ("train", "val", "test"):
            assigned = work["split"]
        else:
            for sname, id_set in mapping.items():
                if wid in id_set:
                    assigned = sname
                    break

        if not assigned:
            raise ValueError(f"Work ID '{wid}' not found in split mapping for {split_option}")

        work_copy = dict(work)
        work_copy["split"] = assigned
        splits[assigned].append(work_copy)

    verify_whole_work_invariant(splits)
    return splits


def verify_whole_work_invariant(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Assert that no work appears in more than one split partition.

    Raises:
        SplitLeakageError: If any work ID appears in multiple split sets.
    """
    seen_ids: Dict[str, str] = {}
    split_sets: Dict[str, Set[str]] = {}

    for split_name in ("train", "val", "test"):
        items = splits.get(split_name, [])
        curr_set: Set[str] = set()
        for item in items:
            wid = item["id"]
            if wid in seen_ids:
                prev_split = seen_ids[wid]
                raise SplitLeakageError(
                    f"Variorum leakage: Work '{wid}' appears in both '{prev_split}' and '{split_name}'!"
                )
            seen_ids[wid] = split_name
            curr_set.add(wid)
        split_sets[split_name] = curr_set

    # Verify disjointness
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 != s2:
                overlap = split_sets[s1].intersection(split_sets[s2])
                if overlap:
                    raise SplitLeakageError(
                        f"Whole-work split disjointness failed between {s1} and {s2}: {overlap}"
                    )
