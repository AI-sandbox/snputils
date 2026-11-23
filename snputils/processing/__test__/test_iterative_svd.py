import numpy as np
import pytest

from snputils.processing._utils.iterative_svd import IterativeSVD
from snputils.processing.mdpca import mdPCA


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


@pytest.mark.filterwarnings("error:Mean of empty slice:RuntimeWarning")
def test_mdpca_imputes_small_covariance_matrix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cov = np.array([
        [4, 2, 1, np.nan],
        [2, 5, 2, 1],
        [1, 2, 6, 3],
        [np.nan, 1, 3, 7],
    ])
    completed = mdPCA._iterative_svd_matrix_completion(cov)

    assert np.isfinite(completed).all()
    observed = ~np.isnan(cov)
    np.testing.assert_array_equal(completed[observed], cov[observed])


def test_iterative_svd_cannot_cross_validate_one_observation():
    matrix = np.array([[1., np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError, match="at least two observed entries"):
        IterativeSVD(init_fill_method="zero").fit_transform(matrix)
