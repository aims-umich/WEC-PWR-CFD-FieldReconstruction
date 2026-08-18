"""Plot all-layer metrics from the 3D split-sensitivity experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_FRACTIONS: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
RESULTS_PREFIX = "split_sensitivity_3d_train"
LAYER_TITLES: tuple[str, ...] = ("Plane 1", "Plane 2", "Plane 3", "Plane 4")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_results_root() -> Path:
    return _repo_root() / "experiments" / "field_reconstruction" / "results"


def _format_train_fraction(train_frac: float) -> str:
    return f"{train_frac:.2f}".replace(".", "p")


def _results_dir_name(train_frac: float) -> str:
    return f"{RESULTS_PREFIX}_{_format_train_fraction(train_frac)}"


def _load_split_metrics(results_root: Path) -> tuple[list[float], dict[str, list[list[float]]]]:
    """Load MAPE and R2 arrays for each configured training split."""
    train_fractions: list[float] = []
    metrics: dict[str, list[list[float]]] = {"mape": [], "r2": []}

    for requested_train_frac in TRAIN_FRACTIONS:
        run_dir = results_root / _results_dir_name(requested_train_frac)
        with (run_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        with (run_dir / "per_level_summary.json").open("r", encoding="utf-8") as f:
            per_level = json.load(f)

        train_fractions.append(float(config["split"]["train_frac"]))
        metrics["mape"].append([float(value) for value in per_level["mape"]])
        metrics["r2"].append([float(value) for value in per_level["r2"]])

    return train_fractions, metrics


def _plot_2x2_metric(
    train_fractions: list[float],
    values_by_split: list[list[float]],
    *,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(values_by_split, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    flat_axes = axes.ravel()

    for layer_idx, ax in enumerate(flat_axes):
        ax.plot(train_fractions, values[:, layer_idx], marker="o", linewidth=2)
        ax.set_title(LAYER_TITLES[layer_idx])
        ax.set_xlabel("Training fraction")
        ax.set_ylabel(ylabel)
        ax.set_xticks(train_fractions)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def _plot_combined_metric(
    train_fractions: list[float],
    values_by_split: list[list[float]],
    *,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(values_by_split, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))

    for layer_idx, layer_title in enumerate(LAYER_TITLES):
        ax.plot(
            train_fractions,
            values[:, layer_idx],
            marker="o",
            linewidth=2,
            label=layer_title,
        )

    ax.set_title(title)
    ax.set_xlabel("Training fraction")
    ax.set_ylabel(ylabel)
    ax.set_xticks(train_fractions)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_split_sensitivity(results_root: Path) -> None:
    train_fractions, metrics = _load_split_metrics(results_root)

    _plot_2x2_metric(
        train_fractions,
        metrics["mape"],
        ylabel="MAPE (%)",
        title="3D CNN split sensitivity: MAPE by axial plane",
        output_path=results_root / "split_sensitivity_3d_mape_all_layers.png",
    )
    _plot_2x2_metric(
        train_fractions,
        metrics["r2"],
        ylabel="R²",
        title="3D CNN split sensitivity: R² by axial plane",
        output_path=results_root / "split_sensitivity_3d_r2_all_layers.png",
    )
    _plot_combined_metric(
        train_fractions,
        metrics["mape"],
        ylabel="MAPE (%)",
        title="3D CNN split sensitivity: MAPE comparison",
        output_path=results_root / "split_sensitivity_3d_mape_combined.png",
    )
    _plot_combined_metric(
        train_fractions,
        metrics["r2"],
        ylabel="R²",
        title="3D CNN split sensitivity: R² comparison",
        output_path=results_root / "split_sensitivity_3d_r2_combined.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all-layer 3D split-sensitivity MAPE and R² results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_default_results_root(),
        help="Directory containing split_sensitivity_3d_train_* result folders.",
    )
    args = parser.parse_args()
    plot_split_sensitivity(args.results_root)


if __name__ == "__main__":
    main()
