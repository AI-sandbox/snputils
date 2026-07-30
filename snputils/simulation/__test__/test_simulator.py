import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "Tensor"):
    pytest.skip("PyTorch is required for simulator tests.", allow_module_level=True)

from snputils.simulation.simulator import OnlineSimulator


def test_map_breakpoints_use_poisson_crossover_counts(monkeypatch):
    simulator = OnlineSimulator.__new__(OnlineSimulator)
    simulator.rate_per_snp = np.array([0.001, 0.002, 0.003])
    observed = {}

    def poisson(lam):
        observed["lam"] = lam
        return np.array([1, 2, 3])

    monkeypatch.setattr(np.random, "poisson", poisson)

    split_points = simulator._draw_split_points(
        n_snps=4,
        num_generation_max=10,
        num_generations=10,
    )

    np.testing.assert_allclose(observed["lam"], 10 * simulator.rate_per_snp)
    np.testing.assert_array_equal(split_points, [1, 3])
