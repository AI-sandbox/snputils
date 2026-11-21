import numpy as np
import pytest

from snputils.processing._utils.iterative_svd import IterativeSVD


@pytest.mark.parametrize("choose_best", [False, True])
def test_iterative_svd_fills_missing_values(tmp_path, monkeypatch, choose_best):
    monkeypatch.chdir(tmp_path)  # choose_best=True saves IterativeSVD_CVplot.png
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2)) @ rng.normal(size=(2, 20))
    X_missing = X.copy()
    X_missing[rng.random(X.shape) < 0.1] = np.nan

    X_filled = IterativeSVD(
        start_rank=1, end_rank=3, rank=2, choose_best=choose_best, verbose=False
    ).fit_transform(X_missing)

    observed = ~np.isnan(X_missing)
    assert np.isfinite(X_filled).all()
    np.testing.assert_array_equal(X_filled[observed], X[observed])
