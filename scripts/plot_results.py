"""Render figures from measured JSON/CSV artifacts; never uses example metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures"
ORDER = ["nearest_neighbor", "nearest_neighbor_2opt", "initial_policy", "gnn_reinforce", "gnn_reinforce_2opt"]
LABELS = ["Nearest", "Nearest + 2-opt", "Untrained policy", "GNN + RL", "GNN + RL + 2-opt"]
COLORS = ["#9AA7B0", "#596F80", "#C1B9AF", "#178477", "#07574D"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "svg.fonttype": "none"})
    metrics = json.loads((ROOT / "artifacts/routing/metrics.json").read_text(encoding="utf-8"))
    methods = metrics["evaluation"]["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, key, title in zip(axes, ["mean_distance", "mean_vehicles"],
                              ["Distance (dimensionless; lower is better)", "Vehicles used (unlimited fleet)"]):
        values = [methods[name][key] for name in ORDER]
        bars = ax.barh(LABELS, values, color=COLORS, height=0.65)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.3f", padding=5)
        ax.set_xlim(0, max(values) * 1.17)
        ax.set_title(title, loc="left", pad=15)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("128 held-out synthetic CVRP instances · 15 customers · capacity 30", fontsize=14)
    save(fig, "routing_comparison")

    with (ROOT / "artifacts/routing/training_trace.csv").open(encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    valid = [row for row in trace if row["validation_distance"]]
    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    ax.plot([int(r["step"]) for r in trace], [float(r["sampled_distance"]) for r in trace],
            alpha=0.45, color="#596F80", label="Sampled train batch (different instances)")
    ax.plot([int(r["step"]) for r in valid], [float(r["validation_distance"]) for r in valid],
            "o-", color="#178477", label="Fixed validation set / greedy decode")
    ax.axvline(metrics["training"]["selected_step"], linestyle="--", color="#DC9141", label="Selected checkpoint")
    ax.set(xlabel="Optimizer step", ylabel="Mean route distance", title="Actual CUDA training trace")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "training_trace")

    examples = json.loads((ROOT / "artifacts/routing/examples.json").read_text(encoding="utf-8"))
    example = examples[0]  # First test seed, never cherry-picked by improvement.
    coords = np.asarray(example["instance"]["coords"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    for ax, method, label in zip(axes, [ORDER[0], ORDER[1], ORDER[4]], [LABELS[0], LABELS[1], LABELS[4]]):
        solution = example["solutions"][method]
        for route in solution["routes"]:
            path = coords[route]
            ax.plot(path[:, 0], path[:, 1], "o-", markersize=4, linewidth=1.2)
        ax.scatter(*coords[0], s=130, marker="s", color="#17282E", zorder=10)
        for index, (x, y) in enumerate(coords[1:], 1):
            ax.annotate(str(index), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set(aspect="equal", xlim=(-0.05, 1.05), ylim=(-0.05, 1.05),
               title=f"{label}\nDistance {solution['distance']:.3f} · {len(solution['routes'])} vehicles")
    fig.suptitle(f"First held-out instance (seed {example['instance_seed']}) · synthetic coordinates", fontsize=13)
    save(fig, "route_example")
    print(f"Created measured routing figures in {OUT}")


if __name__ == "__main__":
    main()
