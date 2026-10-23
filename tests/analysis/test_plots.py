"""
tests/analysis/test_plots.py : Tests for scaling curve plotting, BPB vs T, and trajectory rendering.
"""

import tempfile
from pathlib import Path

from src.rgpt.analysis.plots import (
    plot_adaptation_trajectories,
    plot_bpb_vs_compression,
    plot_scaling_curves,
)


def test_plot_generation_creates_image_files() -> None:
    """Plotting functions must successfully create non-empty image files from synthetic metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        p1 = tmp_path / "scaling.png"
        plot_scaling_curves(
            {
                "tokenizers": {
                    "bpe16": {"sizes": [10e6, 25e6], "bpc": [1.4, 1.2], "bpb": [1.0, 0.9]},
                    "byte": {"sizes": [10e6, 25e6], "bpc": [1.6, 1.4], "bpb": [1.2, 1.1]},
                }
            },
            p1,
        )
        assert p1.exists()
        assert p1.stat().st_size > 1000

        p2 = tmp_path / "compression.png"
        plot_bpb_vs_compression(
            {
                "sizes": {
                    "10m": {"T": [1.0, 2.0, 3.5], "bpb": [1.5, 1.3, 1.2]},
                }
            },
            p2,
        )
        assert p2.exists()
        assert p2.stat().st_size > 1000

        p3 = tmp_path / "adapt.png"
        plot_adaptation_trajectories(
            {
                "runs": {
                    "AD-bpe16-25m-lr1": {"steps": [0, 50, 100], "val_bpc": [1.4, 1.2, 1.0]},
                },
                "target_bpc": 1.1,
            },
            p3,
        )
        assert p3.exists()
        assert p3.stat().st_size > 1000
