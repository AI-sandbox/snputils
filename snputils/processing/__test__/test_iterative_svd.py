import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD

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


def test_covariance_completion_averages_estimates_and_preserves_observed_mirrors(monkeypatch):
    cov = np.array([[4, np.nan, 1], [np.nan, 5, 2], [np.nan, 2, 6]])
    estimates = np.array([[4., 0.5, 1.], [1.5, 5., 2.], [0.4, 2., 6.]])
    monkeypatch.setattr(IterativeSVD, "fit_transform", lambda self, matrix: estimates.copy())

    completed = mdPCA._iterative_svd_matrix_completion(cov)
    np.testing.assert_array_equal(completed, [[4, 1, 1], [1, 5, 2], [1, 2, 6]])


def test_covariance_validation_holds_out_pairs_and_retains_training_links():
    # Each sample has only one cross-covariance with another sample.
    observed = np.eye(4, dtype=bool)
    observed[0, 1] = observed[1, 0] = observed[2, 3] = observed[3, 2] = True
    solver = IterativeSVD(symmetric=True)
    holdout = solver.create_validation_mask(observed.astype(float), ~observed)
    assert np.count_nonzero(holdout) == 1
    assert np.all(holdout == np.diag(holdout.diagonal()))
    assert not np.any(holdout & ~observed)

    # With all diagonals missing, every eligible pair must hide two cells.
    missing = np.eye(30, dtype=bool)
    holdout = solver.create_validation_mask(np.ones((30, 30)), missing)
    np.testing.assert_array_equal(holdout, holdout.T)
    assert not np.any(holdout & missing)
    assert np.count_nonzero(holdout) == 2 * (435 // 100)


def test_covariance_rank_scoring_counts_each_pair_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cov = np.array([[8., 4., 3.], [4., 6., 1.], [3., 1., np.nan]])
    holdout = np.array([[True, True, False], [True, False, False], [False, False, False]])
    monkeypatch.setattr(IterativeSVD, "create_validation_mask", lambda self, X, missing: holdout)

    fits = []

    def record_fit(self, X):
        fits.append(X.copy())
        return np.zeros((len(X), self.n_components))

    predictions = np.array([[5., 2., 3.], [4., 6., 1.], [3., 1., 9.]])
    monkeypatch.setattr(TruncatedSVD, "fit_transform", record_fit)
    monkeypatch.setattr(TruncatedSVD, "inverse_transform", lambda self, X: predictions)
    errors = []
    monkeypatch.setattr("matplotlib.pyplot.plot", lambda ranks, values: errors.extend(values))
    mdPCA._iterative_svd_matrix_completion(cov)

    # Both directions and the held-out diagonal start at their training column means.
    np.testing.assert_array_equal(fits[0][holdout], [3., 3.5, 3.])
    # Diagonal error (8 - 5)^2 and pair error (4 - mean(2, 4))^2.
    np.testing.assert_allclose(errors, [5., 5.])
