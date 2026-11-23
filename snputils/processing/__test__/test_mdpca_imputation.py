import numpy as np
import pytest

from snputils.processing.mdpca import mdPCA


def test_imputed_covariance_stays_symmetric_with_tied_overlap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    genotypes = np.random.default_rng(182).binomial(1, 0.3, size=(12, 60)).astype(float)
    genotypes[8:, :10] = np.nan
    model = mdPCA()
    observed, completed, strength = model._complete_covariance(genotypes, percent_vals_masked=20)
    masked = np.isnan(model._create_cov_mask(observed, strength, 20))

    np.testing.assert_array_equal(masked, masked.T)
    assert masked.sum() == 28
    assert strength[masked].max() <= strength[~masked].min()
    np.testing.assert_array_equal(completed[~masked], observed[~masked])
    np.testing.assert_array_equal(completed, completed.T)
    assert np.isfinite(completed).all()

    path = tmp_path / "cov.npy"
    model._fit_completed_covariance(completed, path, n_components=2)
    centered = np.load(path)
    np.testing.assert_array_equal(centered, centered.T)
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-6)


def test_covariance_mask_counts_diagonals_once_and_keeps_pairs_whole():
    strength = np.array([[0.25, 0.25, 0.25], [0.25, 1, 1], [0.25, 1, 1]])
    cov = np.eye(3)
    # The weakest diagonal costs one cell and the next pair costs two.
    for percent, expected_count in [(0, 0), (23, 1), (34, 3), (100, 9)]:
        masked = np.isnan(mdPCA._create_cov_mask(cov, strength, percent))
        np.testing.assert_array_equal(masked, masked.T)
        assert masked.sum() == expected_count


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


@pytest.mark.parametrize(("n_complete", "message"), [
    (1, "at least 2 fully observed samples"),
    (4, "rank 1 for 2 requested components"),
])
def test_ils_rejects_insufficient_projection_rank(n_complete, message):
    genotypes = np.random.default_rng(182).binomial(1, 0.3, size=(12, 60)).astype(float)
    genotypes[:n_complete] = genotypes[0].copy()
    incomplete = np.arange(n_complete, len(genotypes))
    genotypes[incomplete, incomplete] = np.nan

    model = mdPCA(method="cov_matrix_imputation_ils", n_components=2)
    with pytest.raises(ValueError, match=message):
        model._run_cov_matrix(genotypes, np.ones(len(genotypes)))


@pytest.mark.parametrize("n_components", [1, 2])
def test_ils_accepts_one_independent_complete_sample_per_component(n_components):
    genotypes = np.random.default_rng(182).binomial(1, 0.3, size=(12, 60)).astype(float)
    incomplete = np.arange(n_components, len(genotypes))
    genotypes[incomplete, incomplete] = np.nan

    model = mdPCA(method="cov_matrix_imputation_ils", n_components=n_components)
    scores = model._run_cov_matrix(genotypes, np.ones(len(genotypes)))
    assert np.linalg.matrix_rank(scores) == n_components


@pytest.mark.parametrize("method", ["cov_matrix_imputation", "cov_matrix_imputation_ils"])
def test_imputation_receives_only_undefined_covariances_as_missing(method, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    genotypes = np.array([
        [0, 1, 1, 2, 0, 2], [1, 0, 1, 0, 2, 0], [1, 1, 0, 2, 0, 1],
        [0, 0, 2, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, 1, 1, 0],
        [np.nan, np.nan, 1, 1, 1, np.nan],
    ])
    model = mdPCA(method=method, n_components=2, covariance_matrix_file=tmp_path / "cov.npy")
    impute = model._iterative_svd_matrix_completion

    def check_and_impute(cov):
        called = np.isfinite(genotypes).astype(int)
        np.testing.assert_array_equal(np.isnan(cov), (called @ called.T) < 2)
        assert cov[1, 4] == 1  # three shared SNPs
        assert cov[2, 3] == 0  # a defined zero stays observed
        return impute(cov)

    monkeypatch.setattr(model, "_iterative_svd_matrix_completion", check_and_impute)
    scores = model._run_cov_matrix(genotypes, np.ones(6))
    assert scores.shape == (6, 2)
    assert np.isfinite(scores).all()
    if method == "cov_matrix_imputation_ils":
        # Sample 5 shares just one SNP with sample 3. Its other predecessors have defined covariances.
        neighbors = [0, 1, 2, 4]
        counts = np.array([3, 3, 3, 2])
        centered = np.load(tmp_path / "cov.npy")
        expected = np.linalg.lstsq(
            scores[neighbors] * counts[:, None], centered[5, neighbors] * counts, rcond=None
        )[0]
        np.testing.assert_allclose(scores[5], expected, rtol=1e-5, atol=1e-6)
