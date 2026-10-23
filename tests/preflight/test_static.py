"""
tests/preflight/test_static.py : Tests for preflight Level P1 static checks and YAML schema validation.
"""

import tempfile
from pathlib import Path

import yaml

from src.rgpt.preflight.static import check_compileall, check_yaml_configs


def test_p1_rejects_unknown_config_key() -> None:
    """Mutation test: inserting unknown key into a config must fail P1 check."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dir = Path(tmp_dir) / "configs"
        config_dir.mkdir()

        # Plant unknown key in decisions.yaml
        bad_config = {
            "text_window_chars": 1024,
            "ILLEGAL_UNKNOWN_KEY": "forbidden",
        }
        with open(config_dir / "decisions.yaml", "w", encoding="utf-8") as f:
            yaml.dump(bad_config, f)

        res = check_yaml_configs(root_dir=Path(tmp_dir))
        assert not res["passed"]
        assert any("Unknown key 'ILLEGAL_UNKNOWN_KEY'" in v for v in res["violations"])


def test_p1_rejects_unresolved_null_at_current_gate() -> None:
    """Mutation test: null value for a key required at current gate must fail P1 check."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dir = Path(tmp_dir) / "configs"
        config_dir.mkdir()

        # Plant null value
        null_config = {
            "text_window_chars": None,
        }
        with open(config_dir / "decisions.yaml", "w", encoding="utf-8") as f:
            yaml.dump(null_config, f)

        res = check_yaml_configs(root_dir=Path(tmp_dir))
        assert not res["passed"]
        assert any("Unresolved null value" in v for v in res["violations"])


def test_p1_compileall_passes() -> None:
    """compileall must compile current src/ without syntax errors."""
    assert check_compileall()
