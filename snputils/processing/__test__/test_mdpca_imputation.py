import numpy as np
import pytest

from snputils.processing.mdpca import mdPCA


def test_ils_fits_completed_covariance_and_projects_observed_covariances(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    genotypes = np.random.default_rng(182).binomial(2, 0.3, size=(12, 60)).astype(float)
    genotypes[8:, :10] = np.nan
    model = mdPCA(
        method="cov_matrix_imputation_ils", n_components=2, percent_vals_masked=20,
        covariance_matrix_file=tmp_path / "cov.npy",
    )
    scores = model._run_cov_matrix(genotypes, np.ones(12))

    _, singular, vh = np.linalg.svd(np.load(tmp_path / "cov.npy"))
    expected = vh[:2].T * np.sqrt(singular[:2])
    overlap = np.isfinite(genotypes).astype(float)
    counts = overlap @ overlap.T
    calls = np.nan_to_num(genotypes)
    observed_cov = (calls @ calls.T) / (counts - 1)
    observed_cov -= observed_cov.mean(axis=0) + observed_cov.mean(axis=1)[:, None] - observed_cov.mean()
    for i in range(8, 12):
        weights = counts[i, :i]
        expected[i] = np.linalg.lstsq(
            expected[:i] * weights[:, None], observed_cov[i, :i] * weights, rcond=None
        )[0]

    # Component signs may differ between SVD implementations.
    np.testing.assert_allclose(scores @ scores.T, expected @ expected.T, rtol=1e-4, atol=1e-6)


def test_ils_requires_a_complete_sample():
    genotypes = np.array([[0, 1, np.nan], [np.nan, 1, 2], [1, np.nan, 0]])
    model = mdPCA(method="cov_matrix_imputation_ils", n_components=1)
    with pytest.raises(ValueError, match="at least one fully observed"):
        model._run_cov_matrix(genotypes, np.ones(3))
