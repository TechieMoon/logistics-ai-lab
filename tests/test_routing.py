"""Constraint, objective, and real gradient/checkpoint tests for the CVRP module."""

import numpy as np
import pytest
import torch

from logistics_ai.routing import (
    CVRPInstance, RoutingPolicy, batch_instances, generate_instance, nearest_neighbor,
    sequence_to_routes, solve_learned, two_opt, validate_routes,
)


def test_validator_rejects_capacity_duplicate_missing_and_depot_errors():
    instance = CVRPInstance([[0, 0], [1, 0], [0, 1], [1, 1]], [0, 2, 2, 1], 3)
    assert validate_routes(instance, [[0, 1, 0], [0, 2, 3, 0]])["valid"]
    for bad in ([[0, 1, 2, 3, 0]], [[0, 1, 1, 0], [0, 2, 3, 0]],
                [[0, 1, 0]], [[1, 0], [0, 2, 3, 0]], [[0, 1, 0, 2, 0], [0, 3, 0]],
                [[0, 99, 0]], [[0, -1, 0]], [[0, True, 0]]):
        assert not validate_routes(instance, bad)["valid"]


def test_return_to_depot_is_included_in_distance():
    instance = CVRPInstance([[0, 0], [3, 4]], [0, 1], 1)
    check = validate_routes(instance, [[0, 1, 0]])
    assert check["valid"]
    assert check["distance"] == pytest.approx(10.0)


@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_baselines_are_valid_and_two_opt_never_worsens(seed):
    instance = generate_instance(seed, 15)
    original = nearest_neighbor(instance)
    original_routes = [route.copy() for route in original.routes]
    improved = two_opt(instance, original)
    assert original.valid and improved.valid
    assert improved.distance <= original.distance + 1e-8
    assert original.routes == original_routes
    assert sorted(node for route in improved.routes for node in route if node) == list(range(1, 16))
    assert all(instance.demands[route].sum() <= instance.capacity for route in improved.routes)


def test_two_opt_reduces_a_crossing_route():
    instance = CVRPInstance([[0, 0], [1, 1], [1, 0], [0, 1]], [0, 1, 1, 1], 3)
    from logistics_ai.routing import RouteSolution
    routes = [[0, 1, 2, 3, 0]]
    original = RouteSolution(routes, validate_routes(instance, routes)["distance"], True)
    assert two_opt(instance, original).distance < original.distance


def test_generator_and_input_validation():
    first, second = generate_instance(17, 10), generate_instance(17, 10)
    np.testing.assert_array_equal(first.coords, second.coords)
    np.testing.assert_array_equal(first.demands, second.demands)
    with pytest.raises(ValueError):
        CVRPInstance([[0, 0], [1, 1]], [0, 5], 4)
    with pytest.raises(ValueError):
        CVRPInstance([[0, 0], [float("nan"), 1]], [0, 1], 4)
    with pytest.raises(ValueError):
        CVRPInstance([[0, 0], [1, 1]], [1, 1], 4)


def test_learned_policy_gradient_constraints_and_checkpoint(tmp_path):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # Tiny capacity forces depot returns; each feasible customer fits individually.
    instances = [generate_instance(100 + seed, 6, capacity=4) for seed in range(8)]
    model = RoutingPolicy(hidden_dim=16, layers=1)
    coords, demands = batch_instances(instances)
    initial_parameter = model.embedding.weight.detach().clone()
    with torch.no_grad():
        baseline, _, _ = model(coords, demands, "greedy")
    costs, log_probs, sequences = model(coords, demands, "sampling")
    for index, instance in enumerate(instances):
        routes = sequence_to_routes(sequences[index].tolist())
        check = validate_routes(instance, routes)
        assert check["valid"], check["errors"]
        assert costs[index].item() == pytest.approx(check["distance"], abs=2e-6)
        assert len(routes) > 1
    loss = ((costs - baseline).detach() * log_probs).mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()
    assert model.embedding.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(initial_parameter, model.embedding.weight.detach())
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_config": {"hidden_dim": 16, "layers": 1}, "model_state_dict": model.state_dict()}, checkpoint)
    solution = solve_learned(instances[0], checkpoint)
    assert solution.valid
    assert solution.distance == pytest.approx(validate_routes(instances[0], solution.routes)["distance"])
    again = solve_learned(instances[0], checkpoint)
    assert solution.routes == again.routes
