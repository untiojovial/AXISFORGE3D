"""Build the diagnostic figure and paired table from the recorded trajectories."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(source, output):
    data = json.loads(source.read_text())
    cfg = data["config"]
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(cfg["seeds"]), figsize=(11, 3.9), sharex=True, sharey=True
    )
    palette = ["#666666", "#b45f06", "#007387", "#703da2"]
    names = ["Aleatorio", "Sobol", "GP sin historia", "Transferencia adaptativa"]
    with (output / "paired.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["seed", "method", "budget", "mean_best_so_far", "successes", "best_proxy"]
        )
        for ax, seed in zip(axes, cfg["seeds"], strict=True):
            for method, label, color in zip(
                cfg["methods"], names, palette, strict=True
            ):
                rows = data["runs"][f"{seed}/{method}"]
                if len(rows) != cfg["budget"]:
                    raise ValueError("incomplete trajectory")
                best = np.maximum.accumulate([row["quality"] for row in rows])
                ax.step(
                    np.arange(1, len(best) + 1),
                    best,
                    where="post",
                    color=color,
                    label=label,
                    linewidth=1.8,
                    alpha=0.9,
                )
                writer.writerow(
                    [
                        seed,
                        method,
                        len(rows),
                        f"{best.mean():.9f}",
                        sum(row["success"] for row in rows),
                        f"{best[-1]:.9f}",
                    ]
                )
            ax.set_title(f"Semilla {seed}", loc="left")
            ax.set_ylim(-0.035, 1.04)
            ax.set_xticks([1, 2, 4, 6, 8, 10])
            ax.set_xlabel("Evaluaciones del objeto")
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Mejor elevación normalizada")
    fig.suptitle(
        "Un cubo simulado · búsqueda del desplazamiento lateral", ha="left", x=0.075
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(output / "cube_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.output)
