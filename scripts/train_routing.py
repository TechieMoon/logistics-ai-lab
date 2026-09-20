"""Train an actual GNN + REINFORCE policy, then evaluate untouched CVRP seeds.

Example: python scripts/train_routing.py --device cuda --steps 400
No claim of improvement is assumed: all comparison numbers are measured.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from logistics_ai.routing import (
    RoutingPolicy, batch_instances, generate_instance, nearest_neighbor,
    solve_with_policy, two_opt, validate_routes,
)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def validation_distance(model: RoutingPolicy, instances: list, device: str) -> float:
    model.eval()
    coords, demands = batch_instances(instances, device)
    costs, _, _ = model(coords, demands, decode_type="greedy")
    return float(costs.mean().item())


def train(args: argparse.Namespace) -> dict:
    if args.steps < 1 or args.batch_size < 2 or args.nodes < 2 or args.eval_instances < 1 or args.validation_instances < 1:
        raise ValueError("steps/eval/validation must be positive, batch and nodes at least two")
    if args.eval_every < 1 or args.hidden_dim < 8 or args.layers < 1 or args.seed < 0:
        raise ValueError("invalid evaluation interval, model size, or seed")
    if args.steps * args.batch_size >= 100_000_000:
        raise ValueError("training seed range would overlap validation")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a smaller smoke run")
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = RoutingPolicy(args.hidden_dim, args.layers).to(args.device)
    initial_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_base = args.seed * 1_000_000_000
    validation_base, test_base = train_base + 100_000_000, train_base + 200_000_000
    validation = [generate_instance(validation_base + index, args.nodes, args.capacity) for index in range(args.validation_instances)]
    initial_validation = validation_distance(model, validation, args.device)
    best_validation, best_step, best_state = float("inf"), 0, None
    trace = []
    synchronize(args.device)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        first_seed = train_base + (step - 1) * args.batch_size
        instances = [generate_instance(first_seed + index, args.nodes, args.capacity) for index in range(args.batch_size)]
        coords, demands = batch_instances(instances, args.device)
        model.train()
        with torch.no_grad():
            baseline_cost, _, _ = model(coords, demands, decode_type="greedy")
        costs, log_probability, _ = model(coords, demands, decode_type="sampling")
        # Minimize expected length: positive advantage means a worse sampled tour.
        advantage = (costs - baseline_cost).detach()
        loss = (advantage * log_probability).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        record = {"step": step, "sampled_distance": float(costs.mean().item()),
                  "greedy_baseline_distance": float(baseline_cost.mean().item()), "reinforce_loss": float(loss.item()),
                  "gradient_norm_before_clip": float(gradient_norm.item()), "validation_distance": ""}
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            value = validation_distance(model, validation, args.device)
            record["validation_distance"] = value
            if value < best_validation:
                best_validation, best_step = value, step
                best_state = copy.deepcopy({key: tensor.cpu() for key, tensor in model.state_dict().items()})
            print(f"step={step}/{args.steps} sampled={record['sampled_distance']:.4f} validation={value:.4f} best={best_validation:.4f}", flush=True)
        trace.append(record)
    synchronize(args.device)
    train_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    parameter_change_l2 = float(sum((best_state[key] - initial_state[key]).square().sum().item() for key in best_state) ** 0.5)
    checkpoint = {"model_config": {"hidden_dim": args.hidden_dim, "layers": args.layers}, "model_state_dict": best_state,
                  "training_step": best_step, "training_seed": args.seed, "n_customers": args.nodes,
                  "capacity": args.capacity, "validation_distance": best_validation,
                  "algorithm": "distance-aware message passing + REINFORCE with greedy self-critical baseline"}
    torch.save(checkpoint, output / "checkpoint.pt")
    torch.save({**checkpoint, "model_state_dict": initial_state, "training_step": 0,
                "validation_distance": initial_validation}, output / "initial_checkpoint.pt")
    with (output / "training_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    model.load_state_dict(best_state)
    initial_model = RoutingPolicy(args.hidden_dim, args.layers).to(args.device)
    initial_model.load_state_dict(initial_state)
    # Warm up inference before latency measurements. Test examples are generated
    # only after validation has selected the final checkpoint.
    for policy in (initial_model, model):
        solve_with_policy(validation[0], policy, args.device)
    synchronize(args.device)
    rows, examples = [], []
    evaluation_start = time.perf_counter()
    for index in range(args.eval_instances):
        instance_seed = test_base + index
        instance = generate_instance(instance_seed, args.nodes, args.capacity)
        solutions, latencies = {}, {}
        for method in ("nearest_neighbor", "nearest_neighbor_2opt", "initial_policy", "gnn_reinforce", "gnn_reinforce_2opt"):
            synchronize(args.device)
            tick = time.perf_counter()
            if method == "nearest_neighbor":
                solution = nearest_neighbor(instance)
            elif method == "nearest_neighbor_2opt":
                solution = two_opt(instance, solutions["nearest_neighbor"])
            elif method == "initial_policy":
                solution = solve_with_policy(instance, initial_model, args.device)
            elif method == "gnn_reinforce":
                solution = solve_with_policy(instance, model, args.device)
            else:
                solution = two_opt(instance, solutions["gnn_reinforce"])
            synchronize(args.device)
            elapsed_ms = (time.perf_counter() - tick) * 1000
            if method.endswith("_2opt"):
                elapsed_ms += latencies[method.removesuffix("_2opt")]
            checks = validate_routes(instance, solution.routes)
            if not checks["valid"] or abs(solution.distance - checks["distance"]) > 1e-6:
                raise RuntimeError(f"independent route verification failed: seed={instance_seed}, method={method}")
            solutions[method], latencies[method] = solution, elapsed_ms
            reference = solutions["nearest_neighbor"].distance
            rows.append({"instance_seed": instance_seed, "method": method, "distance": solution.distance,
                         "valid": checks["valid"], "vehicles_used": checks["vehicles_used"], "latency_ms": elapsed_ms,
                         "improvement_pct_vs_nearest_neighbor": 100 * (reference - solution.distance) / reference})
        if index < 3:
            examples.append({"instance_seed": instance_seed, "instance": instance.to_dict(),
                             "solutions": {key: value.to_dict() for key, value in solutions.items()}})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = {}
    for method in sorted({row["method"] for row in rows}):
        selection = [row for row in rows if row["method"] == method]
        methods[method] = {"mean_distance": float(np.mean([row["distance"] for row in selection])),
                           "std_distance": float(np.std([row["distance"] for row in selection])),
                           "valid_fraction": float(np.mean([row["valid"] for row in selection])),
                           "mean_vehicles": float(np.mean([row["vehicles_used"] for row in selection])),
                           "mean_latency_ms": float(np.mean([row["latency_ms"] for row in selection])),
                           "median_latency_ms": float(np.median([row["latency_ms"] for row in selection]))}
    neural = methods["gnn_reinforce"]["mean_distance"]
    initial = methods["initial_policy"]["mean_distance"]
    nn_distance = methods["nearest_neighbor"]["mean_distance"]
    metrics = {"problem": "synthetic Euclidean CVRP", "distance_unit": "dimensionless coordinate units",
               "objective": "minimize total distance including depot returns, subject to visit-once and capacity",
               "data": {"generator": "uniform [0,1]^2 customers; central depot; integer demand 1..9 capped by capacity",
                        "synthetic": True, "n_customers": args.nodes, "capacity": args.capacity,
                        "seed_ranges_inclusive": {"training": [train_base, train_base + args.steps * args.batch_size - 1],
                                                  "validation": [validation_base, validation_base + args.validation_instances - 1],
                                                  "test": [test_base, test_base + args.eval_instances - 1]},
                        "training_instances": args.steps * args.batch_size, "validation_instances": args.validation_instances,
                        "test_instances": args.eval_instances, "disjoint_seed_ranges": True},
               "training": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                            "seconds": train_seconds, "selected_step": best_step,
                            "checkpoint_selection": "lowest validation mean distance among evaluated trained checkpoints",
                            "initial_validation_distance": initial_validation, "selected_validation_distance": best_validation,
                            "parameter_change_l2": parameter_change_l2, "parameter_count": sum(p.numel() for p in model.parameters())},
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                               "device": args.device, "cuda_runtime": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                               "cpu_threads": args.cpu_threads, "deterministic_algorithms": True},
               "evaluation": {"seconds": time.perf_counter() - evaluation_start,
                              "timing_scope": "warm batch-size-1 solve with feasibility check; excludes checkpoint load; 2-opt includes its parent solve",
                              "methods": methods, "trained_improvement_pct_vs_initial": 100 * (initial - neural) / initial,
                              "trained_improvement_pct_vs_nearest_neighbor": 100 * (nn_distance - neural) / nn_distance,
                              "all_routes_independently_validated": True},
               "limitations": ["Synthetic instances only; no CJ or customer data.", "No road network, traffic, time windows, fleet limit, or production integration.",
                               "REINFORCE is stochastic and learned routing may underperform classical heuristics.",
                               "No optimal solver/reference; improvements are versus named baselines, never optimality gaps.",
                               "Distance prior contributes heuristic knowledge; initial-policy comparison isolates measured training effect.",
                               "Determinism is scoped to the same hardware/software environment."]}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_step": best_step, "parameter_change_l2": parameter_change_l2,
                      "trained_improvement_pct_vs_initial": metrics["evaluation"]["trained_improvement_pct_vs_initial"],
                      "methods": methods}, indent=2), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nodes", type=int, default=15)
    parser.add_argument("--capacity", type=float, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-instances", type=int, default=64)
    parser.add_argument("--eval-instances", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "routing"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
