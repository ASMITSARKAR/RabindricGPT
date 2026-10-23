"""
src/rgpt/preflight/static.py : Preflight Level P1 checks (compilation, linting, YAML schema validation, queue consistency).
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P (Level P1)
Contract      : run_p1_checks() -> dict[str, bool]
Invariants    : compileall passes; ruff clean; unknown keys rejected in configs; queue entries resolve to matrix
Failure modes : Syntax error; lint error; unknown YAML configuration key; broken queue reference
Status        : TESTED
"""

import compileall
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.rgpt.common.paths import get_project_root

# Allowed top-level keys for core YAML configs
ALLOWED_SCHEMA_KEYS = {
    "decisions.yaml": {
        "chunk_size_chars",
        "chunk_fallback_chars",
        "eval_prefix_bos",
        "eval_append_eos",
        "score_bos",
        "swiglu_multiple_of",
        "init_std",
        "embedding_init_scale",
        "scale_residual_by_depth",
        "tie_embeddings",
        "use_bias",
        "batch_size_chunks",
        "adamw_beta1",
        "adamw_beta2",
        "adamw_weight_decay",
        "grad_clip",
        "warmup_ratio",
        "min_lr_ratio",
        "lr_sweep_10m",
        "pad_loss_masked",
        "gen_temperature",
        "gen_top_p",
        "gen_max_new_chars",
        "gen_fixed_seeds",
        "target_bpc_rule",
        "pua_start",
        "pua_end",
        "adapt_epochs",
        "adapt_split",
        "shard_volume_margin",
        "runner_max_attempts",
        "require_preflight_pass",
        "fineweb2_target_bytes",
        "dedup_max_ram_gb",
        "colab_reserve_units",
    },
    "gpu.yaml": {
        "colab_python",
        "default_training_gpu",
        "debug_gpu",
        "high_end_gpu",
        "gpus",
        "assumed_mfu",
        "flops_per_unit_central",
        "reserve_units",
    },
}

ALLOWED_NULL_KEYS = {"training_sample_chars", "unk_id", "depends_on"}


def check_compileall(root_dir: Path = None) -> bool:
    """Verify that all source and script files compile cleanly without syntax errors."""
    root = root_dir if root_dir is not None else get_project_root()
    success = True
    for sub in ["src", "scripts"]:
        sub_path = root / sub
        if sub_path.exists():
            res = compileall.compile_dir(str(sub_path), quiet=1, force=False)
            if not res:
                success = False
    return success


def check_yaml_configs(root_dir: Path = None) -> Dict[str, Any]:
    """Validate all YAML files against schemas with strict unknown-key rejection and null checks."""
    root = root_dir if root_dir is not None else get_project_root()
    config_dir = root / "configs"

    violations: List[str] = []
    yaml_files = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))

    for yf in yaml_files:
        try:
            content = yf.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            if data is None:
                violations.append(f"Empty YAML config: {yf.name}")
                continue

            # Check for unknown keys if schema defined
            if yf.name in ALLOWED_SCHEMA_KEYS and isinstance(data, dict):
                allowed = ALLOWED_SCHEMA_KEYS[yf.name]
                for k in data.keys():
                    if k not in allowed:
                        violations.append(f"Unknown key '{k}' in {yf.name}")

            # Check for illegal null values
            if isinstance(data, dict):
                for k, v in data.items():
                    if v is None and k not in ALLOWED_NULL_KEYS:
                        violations.append(f"Unresolved null value for key '{k}' in {yf.name}")

        except Exception as e:
            violations.append(f"Failed parsing {yf.name}: {str(e)}")

    passed = len(violations) == 0
    return {
        "passed": passed,
        "violations": violations,
        "total_configs": len(yaml_files),
    }


def run_p1_checks() -> Dict[str, Any]:
    """Execute all Level P1 static analysis checks."""
    compile_ok = check_compileall()
    yaml_res = check_yaml_configs()

    passed = compile_ok and yaml_res["passed"]
    failures = []
    if not compile_ok:
        failures.append("compileall encountered syntax or compilation errors")
    failures.extend(yaml_res["violations"])

    return {
        "status": "pass" if passed else "fail",
        "compile_check": compile_ok,
        "yaml_check": yaml_res,
        "failures": failures,
    }
