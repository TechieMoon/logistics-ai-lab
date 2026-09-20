"""Synthetic CVRP, independent feasibility checks, and a message-passing policy.

Coordinates are dimensionless Euclidean points; distance is not road kilometres.
There is one depot (index 0), identical vehicles and an unlimited vehicle count.
The policy is a small educational implementation, not a reproduction of a paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CVRPInstance:
    coords: np.ndarray
    demands: np.ndarray
    capacity: float

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float64)
        self.demands = np.asarray(self.demands, dtype=np.float64)
        self.capacity = float(self.capacity)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2 or len(self.coords) < 2:
            raise ValueError("coords must contain depot plus customers, shape [N+1, 2]")
        if self.demands.shape != (len(self.coords),):
            raise ValueError("demands must have one value per coordinate")
        if not np.isfinite(self.coords).all() or not np.isfinite(self.demands).all():
            raise ValueError("coordinates and demands must be finite")
        if not np.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("capacity must be positive and finite")
        if self.demands[0] != 0 or np.any(self.demands[1:] <= 0):
            raise ValueError("depot demand must be zero; customer demands must be positive")
        if np.any(self.demands > self.capacity):
            raise ValueError("each customer demand must fit a vehicle")

    @property
    def n_customers(self) -> int:
        return len(self.coords) - 1

    def to_dict(self) -> dict[str, Any]:
        return {"coords": self.coords.tolist(), "demands": self.demands.tolist(), "capacity": self.capacity}


@dataclass
class RouteSolution:
    routes: list[list[int]]
    distance: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"routes": self.routes, "distance": self.distance, "valid": self.valid, "diagnostics": self.diagnostics}


def generate_instance(seed: int, n_customers: int = 15, capacity: float = 30) -> CVRPInstance:
    if not isinstance(n_customers, int) or n_customers < 1:
        raise ValueError("n_customers must be a positive integer")
    if not np.isfinite(capacity) or capacity < 1:
        raise ValueError("synthetic capacity must be at least 1")
    rng = np.random.default_rng(seed)
    coords = rng.random((n_customers + 1, 2))
    coords[0] = [0.5, 0.5]
    demands = rng.integers(1, min(9, int(capacity)) + 1, n_customers + 1).astype(float)
    demands[0] = 0
    return CVRPInstance(coords, demands, capacity)


def validate_routes(instance: CVRPInstance, routes: list[list[int]]) -> dict[str, Any]:
    """Recompute objective and constraints without trusting a solver's state."""
    errors: list[str] = []
    visited: list[int] = []
    route_loads: list[float] = []
    distance = 0.0
    for index, route in enumerate(routes):
        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            errors.append(f"route {index}: must start/end at depot and serve a customer")
        if any(not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_)) or node < 0 or node > instance.n_customers for node in route):
            errors.append(f"route {index}: invalid node index")
            continue
        if 0 in route[1:-1]:
            errors.append(f"route {index}: depot occurs inside route")
        customers = [int(node) for node in route if node != 0]
        visited.extend(customers)
        load = float(instance.demands[customers].sum())
        route_loads.append(load)
        if load > instance.capacity + 1e-8:
            errors.append(f"route {index}: load {load} exceeds capacity {instance.capacity}")
        # Independent NumPy calculation includes the return to the depot.
        for a, b in zip(route[:-1], route[1:]):
            distance += float(np.linalg.norm(instance.coords[a] - instance.coords[b]))
    counts = np.bincount(visited, minlength=instance.n_customers + 1)
    missing = [i for i in range(1, instance.n_customers + 1) if counts[i] == 0]
    duplicate = [i for i in range(1, instance.n_customers + 1) if counts[i] > 1]
    if missing:
        errors.append(f"missing customers: {missing}")
    if duplicate:
        errors.append(f"duplicate customers: {duplicate}")
    return {"valid": not errors, "errors": errors, "distance": distance, "route_loads": route_loads,
            "vehicles_used": len(routes), "customers_served": len(visited)}


def _solution(instance: CVRPInstance, routes: list[list[int]], method: str) -> RouteSolution:
    checks = validate_routes(instance, routes)
    return RouteSolution(routes, checks["distance"], checks["valid"], {"method": method, **checks})


def nearest_neighbor(instance: CVRPInstance) -> RouteSolution:
    remaining = set(range(1, instance.n_customers + 1))
    routes: list[list[int]] = []
    while remaining:
        route, load = [0], 0.0
        while True:
            feasible = [node for node in sorted(remaining) if load + instance.demands[node] <= instance.capacity + 1e-8]
            if not feasible:
                break
            next_node = min(feasible, key=lambda node: (np.linalg.norm(instance.coords[route[-1]] - instance.coords[node]), node))
            route.append(next_node)
            load += float(instance.demands[next_node])
            remaining.remove(next_node)
        route.append(0)
        routes.append(route)
    return _solution(instance, routes, "nearest_neighbor")


def two_opt(instance: CVRPInstance, solution: RouteSolution) -> RouteSolution:
    """Strictly improving within-route 2-opt; assignment/capacity stay unchanged."""
    if not validate_routes(instance, solution.routes)["valid"]:
        raise ValueError("2-opt requires a feasible starting solution")
    distances = np.linalg.norm(instance.coords[:, None, :] - instance.coords[None, :, :], axis=-1)
    improved_routes = []
    for source in solution.routes:
        route = source.copy()
        changed = True
        while changed:
            changed = False
            for left in range(1, len(route) - 2):
                for right in range(left + 1, len(route) - 1):
                    a, b, c, d = route[left - 1], route[left], route[right], route[right + 1]
                    gain = distances[a, c] + distances[b, d] - distances[a, b] - distances[c, d]
                    if gain < -1e-10:
                        route[left:right + 1] = reversed(route[left:right + 1])
                        changed = True
            # Strict improvement and finitely many permutations imply termination.
        improved_routes.append(route)
    return _solution(instance, improved_routes, solution.diagnostics.get("method", "route") + "+2opt")


def batch_instances(instances: list[CVRPInstance], device: str | torch.device = "cpu") -> tuple[Tensor, Tensor]:
    if not instances or len({len(item.coords) for item in instances}) != 1:
        raise ValueError("a nonempty batch must contain equal customer counts")
    coords = torch.as_tensor(np.stack([item.coords for item in instances]), dtype=torch.float32, device=device)
    demands = torch.as_tensor(np.stack([item.demands / item.capacity for item in instances]), dtype=torch.float32, device=device)
    return coords, demands


class MessagePassingLayer(nn.Module):
    """Distance-aware, dense graph attention followed by a residual node update."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.distance_weight = nn.Parameter(torch.tensor(1.0))
        self.scale = hidden_dim ** -0.5

    def forward(self, nodes: Tensor, distances: Tensor) -> Tensor:
        scores = torch.bmm(self.query(nodes), self.key(nodes).transpose(1, 2)) * self.scale
        scores = scores - F.softplus(self.distance_weight) * distances
        messages = torch.bmm(scores.softmax(-1), self.value(nodes))
        return self.norm(nodes + self.update(torch.cat((nodes, messages), dim=-1)))


class RoutingPolicy(nn.Module):
    """Graph encoder and masked sequential CVRP decoder.

    The trainable distance prior makes short runs usable, while policy-gradient
    learning updates both message-passing encoder and decoder parameters.
    """

    def __init__(self, hidden_dim: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim, self.layers = hidden_dim, layers
        self.embedding = nn.Linear(4, hidden_dim)
        self.graph_layers = nn.ModuleList([MessagePassingLayer(hidden_dim) for _ in range(layers)])
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context = nn.Linear(2 * hidden_dim + 1, hidden_dim)
        self.local_score = nn.Sequential(nn.Linear(7, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.distance_weight = nn.Parameter(torch.tensor(4.0))
        self.depot_penalty = nn.Parameter(torch.tensor(1.5))
        nn.init.normal_(self.local_score[-1].weight, std=0.01)
        nn.init.zeros_(self.local_score[-1].bias)

    def forward(self, coords: Tensor, demands: Tensor, decode_type: str = "sampling") -> tuple[Tensor, Tensor, Tensor]:
        """Return normalized-coordinate costs, log probabilities, and node sequences.

        ``demands`` must be divided by vehicle capacity and have zero at depot.
        Completed rows produce only zero-cost depot padding. At most 2N+1
        actions are needed because repeated depot visits are prohibited.
        """
        if decode_type not in {"sampling", "greedy"}:
            raise ValueError("decode_type must be sampling or greedy")
        batch, nodes, _ = coords.shape
        device = coords.device
        depot_flag = torch.zeros_like(demands)
        depot_flag[:, 0] = 1
        embedding = self.embedding(torch.cat((coords, demands.unsqueeze(-1), depot_flag.unsqueeze(-1)), -1))
        distances = torch.cdist(coords, coords)
        for layer in self.graph_layers:
            embedding = layer(embedding, distances)
        keys, global_embedding = self.key(embedding), embedding.mean(1)
        row = torch.arange(batch, device=device)
        visited = torch.zeros((batch, nodes), dtype=torch.bool, device=device)
        current = torch.zeros(batch, dtype=torch.long, device=device)
        remaining_capacity = torch.ones(batch, device=device)
        costs = torch.zeros(batch, device=device)
        log_probability = torch.zeros(batch, device=device)
        actions = []
        for _ in range(2 * (nodes - 1) + 1):
            all_served = visited[:, 1:].all(1)
            done = all_served & current.eq(0)
            feasible = ~visited & (demands <= remaining_capacity[:, None] + 1e-7)
            feasible[:, 0] = current.ne(0) | done
            feasible[done, 1:] = False
            context = torch.cat((embedding[row, current], global_embedding, remaining_capacity[:, None]), -1)
            query = self.context(context)
            scores = (keys * query[:, None, :]).sum(-1) / self.hidden_dim ** 0.5
            deltas = coords - coords[row, current][:, None, :]
            distance = distances[row, current]
            features = torch.cat((deltas, distance.unsqueeze(-1), distances[:, 0].unsqueeze(-1),
                                  demands.unsqueeze(-1), remaining_capacity[:, None, None].expand(-1, nodes, -1),
                                  depot_flag.unsqueeze(-1)), -1)
            scores = scores + self.local_score(features).squeeze(-1) - F.softplus(self.distance_weight) * distance
            # A trainable depot bias discourages unnecessary one-customer tours.
            scores = scores - depot_flag * F.softplus(self.depot_penalty) * (~all_served)[:, None]
            scores = scores.masked_fill(~feasible, float("-inf"))
            log_probs = F.log_softmax(scores, dim=-1)
            if decode_type == "sampling":
                next_node = torch.multinomial(log_probs.exp(), 1).squeeze(1)
            else:
                next_node = scores.argmax(-1)
            log_probability = log_probability + log_probs[row, next_node]
            costs = costs + distances[row, current, next_node]
            # New tensors avoid modifying masks saved by autograd.
            visited = visited.clone()
            visited[row, next_node] = True
            remaining_capacity = torch.where(next_node.eq(0), torch.ones_like(remaining_capacity),
                                             remaining_capacity - demands[row, next_node])
            current = next_node
            actions.append(next_node)
        return costs, log_probability, torch.stack(actions, 1)


def sequence_to_routes(sequence: list[int]) -> list[list[int]]:
    routes, route = [], [0]
    for raw_node in sequence:
        node = int(raw_node)
        if node == 0:
            if len(route) > 1:
                routes.append(route + [0])
                route = [0]
        else:
            route.append(node)
    if len(route) > 1:
        route.append(0)
        routes.append(route)
    return routes


@torch.no_grad()
def solve_with_policy(instance: CVRPInstance, policy: RoutingPolicy, device: str | torch.device = "cpu") -> RouteSolution:
    policy.eval()
    coords, demands = batch_instances([instance], device)
    _, _, sequence = policy(coords, demands, decode_type="greedy")
    solution = _solution(instance, sequence_to_routes(sequence[0].cpu().tolist()), "gnn_reinforce_greedy")
    if not solution.valid:
        raise RuntimeError(f"Policy produced invalid route: {solution.diagnostics['errors']}")
    return solution


def load_policy(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> RoutingPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    model = RoutingPolicy(hidden_dim=int(config["hidden_dim"]), layers=int(config["layers"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def solve_learned(instance: CVRPInstance, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> RouteSolution:
    return solve_with_policy(instance, load_policy(checkpoint_path, device), device)
