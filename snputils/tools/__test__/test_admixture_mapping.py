from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

from snputils.ancestry.io.local.read.__test__.fixtures import (
    make_synthetic_dataset,
    make_synthetic_dataset_with_covariates,
    make_synthetic_quantitative_dataset,
    write_msp,
)
from snputils.tools.admixture_mapping import run_admixture_mapping


def _write_phe(path: Path, sample_ids: Sequence[str], y_binary: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID PHENO\n")
        for sid, yi in zip(sample_ids, y_binary):
            status = 2 if int(yi) == 1 else 1
            handle.write(f"{sid} {sid} {status}\n")


def _write_covar(
    path: Path,
    sample_ids: Sequence[str],
    covar_names: Sequence[str],
    covar_matrix: np.ndarray,
    include_fid: bool = True,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        if include_fid:
            header = ["#FID", "IID"] + list(covar_names)
        else:
            header = ["#IID"] + list(covar_names)
        handle.write(" ".join(header) + "\n")
        for sid, row in zip(sample_ids, covar_matrix):
            values = [f"{float(v):.12g}" for v in row.tolist()]
            if include_fid:
                handle.write(f"{sid} {sid} {' '.join(values)}\n")
            else:
                handle.write(f"{sid} {' '.join(values)}\n")


def _write_sample_list(path: Path, sample_ids: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for sid in sample_ids:
            handle.write(f"{sid} {sid}\n")


def _reference_linear_with_covariates(
    g: np.ndarray,
    y: np.ndarray,
    covar: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    if g.size < (covar.shape[1] + 3):
        return None
    if np.all(g == g[0]):
        return None

    x = np.column_stack([np.ones(g.size, dtype=np.float64), g.astype(np.float64, copy=False), covar])
    y_f = y.astype(np.float64, copy=False)
    try:
        beta_hat, _, rank, _ = np.linalg.lstsq(x, y_f, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < x.shape[1]:
        return None

    dof = g.size - x.shape[1]
    if dof <= 0:
        return None
    residuals = y_f - x @ beta_hat
    sse = float(np.sum(residuals * residuals))
    mse = sse / float(dof)
    try:
        xtx_inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return None
    se2 = float(mse * xtx_inv[1, 1])
    if se2 <= 0.0 or not np.isfinite(se2):
        return None
    se = float(np.sqrt(se2))
    t_stat = float(beta_hat[1] / se)
    p = float(2.0 * stats.t.sf(np.abs(t_stat), df=dof))
    return float(beta_hat[1]), t_stat, p


def _reference_logistic_with_covariates(
    g: np.ndarray,
    y: np.ndarray,
    covar: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    if g.size == 0:
        return None
    if int(np.sum(y)) == 0 or int(np.sum(y)) == y.size:
        return None
    if np.all(g == g[0]):
        return None

    x = np.column_stack([np.ones(g.size, dtype=np.float64), g.astype(np.float64, copy=False), covar])
    y_f = y.astype(np.float64, copy=False)
    prevalence = np.clip(float(np.mean(y_f)), 1e-6, 1.0 - 1e-6)
    init = np.zeros(x.shape[1], dtype=np.float64)
    init[0] = np.log(prevalence / (1.0 - prevalence))

    def neg_loglik(beta: np.ndarray) -> float:
        eta = x @ beta
        mu = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
        ll = np.sum(y_f * np.log(mu) + (1.0 - y_f) * np.log(1.0 - mu))
        return float(-ll)

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        mu = expit(eta)
        return x.T @ (mu - y_f)

    res = minimize(
        neg_loglik,
        init,
        jac=grad,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 800},
    )
    if (not res.success) or (not np.all(np.isfinite(res.x))):
        return None

    beta = res.x.astype(np.float64, copy=False)
    mu = np.clip(expit(x @ beta), 1e-12, 1.0 - 1e-12)
    w = mu * (1.0 - mu)
    info = x.T @ (x * w[:, None])
    try:
        info_inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None
    se2 = float(info_inv[1, 1])
    if se2 <= 0.0 or not np.isfinite(se2):
        return None
    se = float(np.sqrt(se2))
    z = float(beta[1] / se)
    p = float(2.0 * stats.norm.sf(np.abs(z)))
    return float(beta[1]), z, p


def _reference_logistic_from_dosage(g: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if g.size == 0:
        return None
    if int(np.sum(y)) == 0 or int(np.sum(y)) == y.size:
        return None
    if np.all(g == g[0]):
        return None

    x_vals = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    n = np.array([(g == 0).sum(), (g == 1).sum(), (g == 2).sum()], dtype=np.float64)
    c = np.array(
        [y[g == 0].sum(), y[g == 1].sum(), y[g == 2].sum()],
        dtype=np.float64,
    )

    prevalence = np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6)
    init = np.array([np.log(prevalence / (1.0 - prevalence)), 0.0], dtype=np.float64)

    def neg_loglik(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * x_vals
        mu = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
        ll = np.sum(c * np.log(mu) + (n - c) * np.log(1.0 - mu))
        return float(-ll)

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = beta[0] + beta[1] * x_vals
        mu = expit(eta)
        diff = c - n * mu
        return np.array([-np.sum(diff), -np.sum(diff * x_vals)], dtype=np.float64)

    res = minimize(
        neg_loglik,
        init,
        jac=grad,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 400},
    )
    if (not res.success) or (not np.all(np.isfinite(res.x))):
        return None

    beta0 = float(res.x[0])
    beta = float(res.x[1])
    mu = np.clip(expit(beta0 + beta * x_vals), 1e-12, 1.0 - 1e-12)
    w = n * mu * (1.0 - mu)
    i00 = float(np.sum(w))
    i01 = float(np.sum(w * x_vals))
    i11 = float(np.sum(w * x_vals * x_vals))
    det = i00 * i11 - i01 * i01
    if det <= 1e-12:
        return None
    se2 = float((i00 / det))
    if (not np.isfinite(se2)) or se2 <= 0.0:
        return None
    se = float(np.sqrt(se2))
    z = float(beta / se)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return beta, z, p


def _build_reference_df(
    sample_ids: Sequence[str],
    y: np.ndarray,
    lai: np.ndarray,
    chromosomes: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    ancestry_map: Dict[int, str],
) -> pd.DataFrame:
    rows = []
    for code, label in sorted(ancestry_map.items()):
        dosage = (lai[:, 0::2] == code).astype(np.uint8) + (lai[:, 1::2] == code).astype(np.uint8)
        for w in range(lai.shape[0]):
            ref = _reference_logistic_from_dosage(dosage[w], y)
            if ref is None:
                continue
            beta, z, p = ref
            rows.append(
                {
                    "#CHROM": int(chromosomes[w]),
                    "POS": int(starts[w]),
                    "END": int(ends[w]),
                    "ANCESTRY": label,
                    "BETA_ref": beta,
                    "Z_ref": z,
                    "P_ref": p,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "n_samples,n_windows,seed",
    [
        (48, 24, 11),
        (160, 80, 22),
        (400, 160, 33),
    ],
)
def test_internal_matches_reference_logistic_across_scales(
    tmp_path: Path, n_samples: int, n_windows: int, seed: int
):
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_dataset(
        n_samples=n_samples, n_windows=n_windows, seed=seed
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    out_dir = tmp_path / "out"
    _write_phe(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    internal = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=out_dir,
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
    )
    internal = internal[internal["TEST"] == "ADD"].copy()
    internal = internal[np.isfinite(pd.to_numeric(internal["P"], errors="coerce"))].copy()
    internal["#CHROM"] = pd.to_numeric(internal["#CHROM"], errors="coerce").astype(int)
    internal["POS"] = pd.to_numeric(internal["POS"], errors="coerce").astype(int)
    internal["END"] = pd.to_numeric(internal["END"], errors="coerce").astype(int)
    internal["BETA"] = pd.to_numeric(internal["BETA"], errors="coerce")
    internal["Z_STAT"] = pd.to_numeric(internal["Z_STAT"], errors="coerce")
    internal["P"] = pd.to_numeric(internal["P"], errors="coerce")

    reference = _build_reference_df(sample_ids, y, lai, chromosomes, starts, ends, ancestry_map)
    merged = internal.merge(reference, on=["#CHROM", "POS", "END", "ANCESTRY"], how="inner")
    merged = merged.dropna(subset=["BETA", "Z_STAT", "P", "BETA_ref", "Z_ref", "P_ref"])

    assert not merged.empty
    np.testing.assert_allclose(merged["BETA"].to_numpy(), merged["BETA_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["Z_STAT"].to_numpy(), merged["Z_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["P"].to_numpy(), merged["P_ref"].to_numpy(), rtol=3e-2, atol=1e-8)


def test_internal_streaming_memory_matches_default(tmp_path: Path):
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_dataset(
        n_samples=140, n_windows=75, seed=909
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    out_default = tmp_path / "out_default"
    out_stream = tmp_path / "out_stream"

    _write_phe(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    baseline = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=out_default,
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=None,
    ).copy()
    streamed = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=out_stream,
        phe_id="toy",
        batch_size=7,
        keep_hla=True,
        memory=2048,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ANCESTRY", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P"]

    for col in ("#CHROM", "POS", "END"):
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce").astype(int)
        streamed[col] = pd.to_numeric(streamed[col], errors="coerce").astype(int)
    for col in num_cols:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
        streamed[col] = pd.to_numeric(streamed[col], errors="coerce")

    baseline = baseline.sort_values(key_cols).reset_index(drop=True)
    streamed = streamed.sort_values(key_cols).reset_index(drop=True)

    assert len(baseline) == len(streamed)
    pd.testing.assert_series_equal(baseline["ID"], streamed["ID"], check_names=False)
    pd.testing.assert_series_equal(baseline["ANCESTRY"], streamed["ANCESTRY"], check_names=False)
    pd.testing.assert_series_equal(baseline["TEST"], streamed["TEST"], check_names=False)
    pd.testing.assert_series_equal(baseline["ERRCODE"], streamed["ERRCODE"], check_names=False)
    np.testing.assert_allclose(
        baseline[num_cols].to_numpy(),
        streamed[num_cols].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_internal_total_memory_cap_enforced(tmp_path: Path):
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_dataset(
        n_samples=32, n_windows=16, seed=777
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    out_dir = tmp_path / "out"
    _write_phe(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    with pytest.raises(MemoryError):
        run_admixture_mapping(
            phe_path=phe_path,
            msp_path=msp_path,
            results_path=out_dir,
            phe_id="toy",
            batch_size=8,
            keep_hla=True,
            memory=1,
        )


# ---------------------------------------------------------------------------
# Quantitative trait helpers and tests
# ---------------------------------------------------------------------------


def _write_phe_quantitative(path: Path, sample_ids: Sequence[str], y: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID PHENO\n")
        for sid, yi in zip(sample_ids, y):
            handle.write(f"{sid} {sid} {yi}\n")


def _reference_linear_from_dosage(g: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Individual-level OLS regression of y on dosage g.

    Returns (beta, t_stat, p) or None if the regression is degenerate.
    """
    if g.size < 3:
        return None
    if np.all(g == g[0]):
        return None

    n = g.size
    g_f = g.astype(np.float64, copy=False)
    y_f = y.astype(np.float64, copy=False)

    g_bar = np.mean(g_f)
    y_bar = np.mean(y_f)
    sxx = np.sum((g_f - g_bar) ** 2)
    sxy = np.sum((g_f - g_bar) * (y_f - y_bar))
    if sxx <= 0.0:
        return None

    beta1 = sxy / sxx
    y_hat = y_bar + beta1 * (g_f - g_bar)
    residuals = y_f - y_hat
    ssr = np.sum(residuals ** 2)
    df = n - 2
    if df <= 0:
        return None
    mse = ssr / df
    se = np.sqrt(mse / sxx)
    if se <= 0.0 or not np.isfinite(se):
        return None

    t_stat = beta1 / se
    p = float(2.0 * stats.t.sf(np.abs(t_stat), df=df))
    return float(beta1), float(t_stat), float(p)


def _build_reference_df_quantitative(
    sample_ids: Sequence[str],
    y: np.ndarray,
    lai: np.ndarray,
    chromosomes: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    ancestry_map: Dict[int, str],
) -> pd.DataFrame:
    rows = []
    for code, label in sorted(ancestry_map.items()):
        dosage = (lai[:, 0::2] == code).astype(np.uint8) + (lai[:, 1::2] == code).astype(np.uint8)
        for w in range(lai.shape[0]):
            ref = _reference_linear_from_dosage(dosage[w], y)
            if ref is None:
                continue
            beta, t_stat, p = ref
            rows.append(
                {
                    "#CHROM": int(chromosomes[w]),
                    "POS": int(starts[w]),
                    "END": int(ends[w]),
                    "ANCESTRY": label,
                    "BETA_ref": beta,
                    "T_ref": t_stat,
                    "P_ref": p,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "n_samples,n_windows,seed",
    [
        (48, 24, 11),
        (160, 80, 22),
        (400, 160, 33),
    ],
)
def test_quantitative_internal_matches_reference_ols_across_scales(
    tmp_path: Path, n_samples: int, n_windows: int, seed: int
):
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_quantitative_dataset(
        n_samples=n_samples, n_windows=n_windows, seed=seed
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    out_dir = tmp_path / "out"
    _write_phe_quantitative(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    internal = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=out_dir,
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
    )
    internal = internal[internal["ERRCODE"] == "."].copy()
    internal = internal[np.isfinite(pd.to_numeric(internal["P"], errors="coerce"))].copy()
    internal["#CHROM"] = pd.to_numeric(internal["#CHROM"], errors="coerce").astype(int)
    internal["POS"] = pd.to_numeric(internal["POS"], errors="coerce").astype(int)
    internal["END"] = pd.to_numeric(internal["END"], errors="coerce").astype(int)
    internal["BETA"] = pd.to_numeric(internal["BETA"], errors="coerce")
    internal["T_STAT"] = pd.to_numeric(internal["T_STAT"], errors="coerce")
    internal["P"] = pd.to_numeric(internal["P"], errors="coerce")

    reference = _build_reference_df_quantitative(sample_ids, y, lai, chromosomes, starts, ends, ancestry_map)
    merged = internal.merge(reference, on=["#CHROM", "POS", "END", "ANCESTRY"], how="inner")
    merged = merged.dropna(subset=["BETA", "T_STAT", "P", "BETA_ref", "T_ref", "P_ref"])

    assert not merged.empty
    np.testing.assert_allclose(merged["BETA"].to_numpy(), merged["BETA_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["T_STAT"].to_numpy(), merged["T_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["P"].to_numpy(), merged["P_ref"].to_numpy(), rtol=3e-2, atol=1e-8)


def test_quantitative_streaming_memory_matches_default(tmp_path: Path):
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_quantitative_dataset(
        n_samples=140, n_windows=75, seed=909
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    out_default = tmp_path / "out_default"
    out_stream = tmp_path / "out_stream"

    _write_phe_quantitative(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    baseline = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=out_default,
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=None,
    ).copy()
    streamed = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=out_stream,
        phe_id="toy",
        batch_size=7,
        keep_hla=True,
        memory=2048,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ANCESTRY", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "SE", "T_STAT", "P"]

    for col in ("#CHROM", "POS", "END"):
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce").astype(int)
        streamed[col] = pd.to_numeric(streamed[col], errors="coerce").astype(int)
    for col in num_cols:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
        streamed[col] = pd.to_numeric(streamed[col], errors="coerce")

    baseline = baseline.sort_values(key_cols).reset_index(drop=True)
    streamed = streamed.sort_values(key_cols).reset_index(drop=True)

    assert len(baseline) == len(streamed)
    pd.testing.assert_series_equal(baseline["ID"], streamed["ID"], check_names=False)
    pd.testing.assert_series_equal(baseline["ANCESTRY"], streamed["ANCESTRY"], check_names=False)
    pd.testing.assert_series_equal(baseline["TEST"], streamed["TEST"], check_names=False)
    pd.testing.assert_series_equal(baseline["ERRCODE"], streamed["ERRCODE"], check_names=False)
    np.testing.assert_allclose(
        baseline[num_cols].to_numpy(),
        streamed[num_cols].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_quantitative_covariates_match_reference_with_col_selection_standardization(tmp_path: Path):
    (
        sample_ids,
        _y_binary,
        y_quant,
        lai,
        chromosomes,
        starts,
        ends,
        ancestry_map,
        covar_names,
        covar_matrix,
        _keep_ids,
        _remove_ids,
    ) = make_synthetic_dataset_with_covariates(
        n_samples=120,
        n_windows=48,
        n_covariates=4,
        seed=707,
    )
    phe_path = tmp_path / "quant.phe"
    msp_path = tmp_path / "quant.msp"
    covar_path = tmp_path / "quant.covar"
    _write_phe_quantitative(phe_path, sample_ids, y_quant)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)
    _write_covar(covar_path, sample_ids, covar_names, covar_matrix, include_fid=True)

    selected_idx = [0, 2]
    cov_selected = covar_matrix[:, selected_idx].astype(np.float64, copy=True)
    cov_selected = (cov_selected - np.mean(cov_selected, axis=0)) / np.std(cov_selected, axis=0, ddof=0)

    adjusted = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_adjusted",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
        covar_path=covar_path,
        covar_col_nums="1,3",
        covar_variance_standardize=True,
    ).copy()
    unadjusted = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_unadjusted",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
    ).copy()

    adjusted = adjusted[adjusted["ERRCODE"] == "."].copy()
    adjusted["#CHROM"] = pd.to_numeric(adjusted["#CHROM"], errors="coerce").astype(int)
    adjusted["POS"] = pd.to_numeric(adjusted["POS"], errors="coerce").astype(int)
    adjusted["END"] = pd.to_numeric(adjusted["END"], errors="coerce").astype(int)
    adjusted["BETA"] = pd.to_numeric(adjusted["BETA"], errors="coerce")
    adjusted["T_STAT"] = pd.to_numeric(adjusted["T_STAT"], errors="coerce")
    adjusted["P"] = pd.to_numeric(adjusted["P"], errors="coerce")

    ref_rows = []
    for code, label in sorted(ancestry_map.items()):
        dosage = (lai[:, 0::2] == code).astype(np.uint8) + (lai[:, 1::2] == code).astype(np.uint8)
        for w in range(lai.shape[0]):
            ref = _reference_linear_with_covariates(dosage[w], y_quant, cov_selected)
            if ref is None:
                continue
            beta, t_stat, p = ref
            ref_rows.append(
                {
                    "#CHROM": int(chromosomes[w]),
                    "POS": int(starts[w]),
                    "END": int(ends[w]),
                    "ANCESTRY": label,
                    "BETA_ref": beta,
                    "T_ref": t_stat,
                    "P_ref": p,
                }
            )
    reference = pd.DataFrame(ref_rows)
    merged = adjusted.merge(reference, on=["#CHROM", "POS", "END", "ANCESTRY"], how="inner")
    merged = merged.dropna(subset=["BETA", "T_STAT", "P", "BETA_ref", "T_ref", "P_ref"])
    assert not merged.empty

    np.testing.assert_allclose(merged["BETA"].to_numpy(), merged["BETA_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["T_STAT"].to_numpy(), merged["T_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["P"].to_numpy(), merged["P_ref"].to_numpy(), rtol=4e-2, atol=1e-8)

    for col in ("#CHROM", "POS", "END"):
        unadjusted[col] = pd.to_numeric(unadjusted[col], errors="coerce").astype(int)
    unadjusted["BETA"] = pd.to_numeric(unadjusted["BETA"], errors="coerce")
    diff = adjusted.merge(
        unadjusted[["#CHROM", "POS", "END", "ANCESTRY", "BETA"]],
        on=["#CHROM", "POS", "END", "ANCESTRY"],
        how="inner",
        suffixes=("_adj", "_base"),
    )
    diff = diff.dropna(subset=["BETA_adj", "BETA_base"])
    assert not diff.empty
    assert np.mean(np.abs(diff["BETA_adj"] - diff["BETA_base"]) > 1e-6) > 0.25


def test_logistic_covariates_match_reference_and_differ_from_unadjusted(tmp_path: Path):
    (
        sample_ids,
        y_binary,
        _y_quant,
        lai,
        chromosomes,
        starts,
        ends,
        ancestry_map,
        covar_names,
        covar_matrix,
        _keep_ids,
        _remove_ids,
    ) = make_synthetic_dataset_with_covariates(
        n_samples=130,
        n_windows=52,
        n_covariates=3,
        seed=808,
    )
    phe_path = tmp_path / "binary.phe"
    msp_path = tmp_path / "binary.msp"
    covar_path = tmp_path / "binary.covar"
    _write_phe(phe_path, sample_ids, y_binary)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)
    _write_covar(covar_path, sample_ids, covar_names, covar_matrix, include_fid=False)

    adjusted = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_adjusted",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
        covar_path=covar_path,
        covar_col_nums="1-3",
    ).copy()
    unadjusted = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_unadjusted",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
    ).copy()

    adjusted = adjusted[(adjusted["ERRCODE"] == ".") & (adjusted["TEST"] == "ADD")].copy()
    adjusted["#CHROM"] = pd.to_numeric(adjusted["#CHROM"], errors="coerce").astype(int)
    adjusted["POS"] = pd.to_numeric(adjusted["POS"], errors="coerce").astype(int)
    adjusted["END"] = pd.to_numeric(adjusted["END"], errors="coerce").astype(int)
    adjusted["BETA"] = pd.to_numeric(adjusted["BETA"], errors="coerce")
    adjusted["Z_STAT"] = pd.to_numeric(adjusted["Z_STAT"], errors="coerce")
    adjusted["P"] = pd.to_numeric(adjusted["P"], errors="coerce")

    ref_rows = []
    for code, label in sorted(ancestry_map.items()):
        dosage = (lai[:, 0::2] == code).astype(np.uint8) + (lai[:, 1::2] == code).astype(np.uint8)
        for w in range(lai.shape[0]):
            ref = _reference_logistic_with_covariates(dosage[w], y_binary, covar_matrix)
            if ref is None:
                continue
            beta, z, p = ref
            ref_rows.append(
                {
                    "#CHROM": int(chromosomes[w]),
                    "POS": int(starts[w]),
                    "END": int(ends[w]),
                    "ANCESTRY": label,
                    "BETA_ref": beta,
                    "Z_ref": z,
                    "P_ref": p,
                }
            )
    reference = pd.DataFrame(ref_rows)
    merged = adjusted.merge(reference, on=["#CHROM", "POS", "END", "ANCESTRY"], how="inner")
    merged = merged.dropna(subset=["BETA", "Z_STAT", "P", "BETA_ref", "Z_ref", "P_ref"])
    assert not merged.empty

    np.testing.assert_allclose(merged["BETA"].to_numpy(), merged["BETA_ref"].to_numpy(), rtol=3e-2, atol=5e-4)
    np.testing.assert_allclose(merged["Z_STAT"].to_numpy(), merged["Z_ref"].to_numpy(), rtol=4e-2, atol=5e-4)
    np.testing.assert_allclose(merged["P"].to_numpy(), merged["P_ref"].to_numpy(), rtol=8e-2, atol=1e-8)

    for col in ("#CHROM", "POS", "END"):
        unadjusted[col] = pd.to_numeric(unadjusted[col], errors="coerce").astype(int)
    unadjusted["BETA"] = pd.to_numeric(unadjusted["BETA"], errors="coerce")
    diff = adjusted.merge(
        unadjusted[["#CHROM", "POS", "END", "ANCESTRY", "BETA"]],
        on=["#CHROM", "POS", "END", "ANCESTRY"],
        how="inner",
        suffixes=("_adj", "_base"),
    )
    diff = diff.dropna(subset=["BETA_adj", "BETA_base"])
    assert not diff.empty
    assert np.mean(np.abs(diff["BETA_adj"] - diff["BETA_base"]) > 1e-6) > 0.20


def test_logistic_covariate_streaming_consistency(tmp_path: Path):
    (
        sample_ids,
        y_binary,
        _y_quant,
        lai,
        chromosomes,
        starts,
        ends,
        ancestry_map,
        covar_names,
        covar_matrix,
        _keep_ids,
        _remove_ids,
    ) = make_synthetic_dataset_with_covariates(
        n_samples=96,
        n_windows=44,
        n_covariates=3,
        seed=990,
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    covar_path = tmp_path / "toy.covar"
    _write_phe(phe_path, sample_ids, y_binary)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)
    _write_covar(covar_path, sample_ids, covar_names, covar_matrix, include_fid=True)

    baseline = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_default",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=None,
        covar_path=covar_path,
    ).copy()
    streamed = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_stream",
        phe_id="toy",
        batch_size=7,
        keep_hla=True,
        memory=2048,
        covar_path=covar_path,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ANCESTRY", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P"]
    for col in ("#CHROM", "POS", "END"):
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce").astype(int)
        streamed[col] = pd.to_numeric(streamed[col], errors="coerce").astype(int)
    for col in num_cols:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
        streamed[col] = pd.to_numeric(streamed[col], errors="coerce")

    baseline = baseline.sort_values(key_cols).reset_index(drop=True)
    streamed = streamed.sort_values(key_cols).reset_index(drop=True)
    assert len(baseline) == len(streamed)
    np.testing.assert_allclose(
        baseline[num_cols].to_numpy(),
        streamed[num_cols].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_ci_columns_logistic_and_quantitative(tmp_path: Path):
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_dataset(
        n_samples=90, n_windows=36, seed=612
    )
    phe_path = tmp_path / "bin.phe"
    msp_path = tmp_path / "bin.msp"
    _write_phe(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    logistic = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_bin",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
        ci=0.95,
    ).copy()
    assert "L95" in logistic.columns
    assert "U95" in logistic.columns
    logistic["BETA"] = pd.to_numeric(logistic["BETA"], errors="coerce")
    logistic["LOG(OR)_SE"] = pd.to_numeric(logistic["LOG(OR)_SE"], errors="coerce")
    logistic["L95"] = pd.to_numeric(logistic["L95"], errors="coerce")
    logistic["U95"] = pd.to_numeric(logistic["U95"], errors="coerce")
    mask = np.isfinite(logistic["BETA"]) & np.isfinite(logistic["LOG(OR)_SE"])
    z_crit = stats.norm.ppf(0.975)
    expected_l = np.exp(logistic.loc[mask, "BETA"] - z_crit * logistic.loc[mask, "LOG(OR)_SE"])
    expected_u = np.exp(logistic.loc[mask, "BETA"] + z_crit * logistic.loc[mask, "LOG(OR)_SE"])
    np.testing.assert_allclose(logistic.loc[mask, "L95"].to_numpy(), expected_l.to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(logistic.loc[mask, "U95"].to_numpy(), expected_u.to_numpy(), rtol=1e-10, atol=1e-12)

    (
        sample_ids_q,
        _y_binary_q,
        y_quant,
        lai_q,
        chromosomes_q,
        starts_q,
        ends_q,
        ancestry_map_q,
        covar_names_q,
        covar_matrix_q,
        _keep_q,
        _remove_q,
    ) = make_synthetic_dataset_with_covariates(
        n_samples=100,
        n_windows=40,
        n_covariates=3,
        seed=613,
    )
    phe_q_path = tmp_path / "quant.phe"
    msp_q_path = tmp_path / "quant.msp"
    covar_q_path = tmp_path / "quant.covar"
    _write_phe_quantitative(phe_q_path, sample_ids_q, y_quant)
    write_msp(msp_q_path, sample_ids_q, lai_q, chromosomes_q, starts_q, ends_q, ancestry_map_q)
    _write_covar(covar_q_path, sample_ids_q, covar_names_q, covar_matrix_q, include_fid=True)

    linear = run_admixture_mapping(
        phe_path=phe_q_path,
        msp_path=msp_q_path,
        results_path=tmp_path / "out_quant",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
        covar_path=covar_q_path,
        ci=0.95,
    ).copy()
    assert "L95" in linear.columns
    assert "U95" in linear.columns
    linear["BETA"] = pd.to_numeric(linear["BETA"], errors="coerce")
    linear["SE"] = pd.to_numeric(linear["SE"], errors="coerce")
    linear["L95"] = pd.to_numeric(linear["L95"], errors="coerce")
    linear["U95"] = pd.to_numeric(linear["U95"], errors="coerce")
    mask = np.isfinite(linear["BETA"]) & np.isfinite(linear["SE"])
    df = len(sample_ids_q) - (2 + covar_matrix_q.shape[1])
    t_crit = stats.t.ppf(0.975, df=df)
    expected_l = linear.loc[mask, "BETA"] - t_crit * linear.loc[mask, "SE"]
    expected_u = linear.loc[mask, "BETA"] + t_crit * linear.loc[mask, "SE"]
    np.testing.assert_allclose(linear.loc[mask, "L95"].to_numpy(), expected_l.to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(linear.loc[mask, "U95"].to_numpy(), expected_u.to_numpy(), rtol=1e-10, atol=1e-12)


def test_adjustment_columns_match_reference(tmp_path: Path):
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_dataset(
        n_samples=88, n_windows=34, seed=414
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    _write_phe(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    adjusted = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
        adjust=True,
    ).copy()

    assert "BONF" in adjusted.columns
    assert "FDR_BH" in adjusted.columns
    p = pd.to_numeric(adjusted["P"], errors="coerce").to_numpy(dtype=float)
    bonf = pd.to_numeric(adjusted["BONF"], errors="coerce").to_numpy(dtype=float)
    fdr = pd.to_numeric(adjusted["FDR_BH"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(p) & (p >= 0.0) & (p <= 1.0)
    m = int(np.sum(valid))
    bonf_ref = np.full_like(p, np.nan, dtype=float)
    fdr_ref = np.full_like(p, np.nan, dtype=float)
    if m > 0:
        p_valid = p[valid]
        bonf_ref[valid] = np.minimum(p_valid * m, 1.0)
        order = np.argsort(p_valid, kind="mergesort")
        sorted_p = p_valid[order]
        ranks = np.arange(1, m + 1, dtype=float)
        bh = sorted_p * (m / ranks)
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.clip(bh, 0.0, 1.0)
        fdr_valid = np.empty(m, dtype=float)
        fdr_valid[order] = bh
        fdr_ref[valid] = fdr_valid

    np.testing.assert_allclose(bonf, bonf_ref, rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(fdr, fdr_ref, rtol=0.0, atol=0.0, equal_nan=True)


def test_keep_remove_filtering_matches_prefiltered_inputs(tmp_path: Path):
    (
        sample_ids,
        y_binary,
        _y_quant,
        lai,
        chromosomes,
        starts,
        ends,
        ancestry_map,
        covar_names,
        covar_matrix,
        _keep_ids,
        _remove_ids,
    ) = make_synthetic_dataset_with_covariates(
        n_samples=110,
        n_windows=42,
        n_covariates=3,
        seed=515,
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    covar_path = tmp_path / "toy.covar"
    keep_path = tmp_path / "keep.txt"
    remove_path = tmp_path / "remove.txt"

    _write_phe(phe_path, sample_ids, y_binary)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)
    _write_covar(covar_path, sample_ids, covar_names, covar_matrix, include_fid=True)

    keep_ids = sample_ids[:80]
    remove_ids = sample_ids[20:30]
    expected_ids: Set[str] = set(keep_ids) - set(remove_ids)
    assert len(expected_ids) > 0
    _write_sample_list(keep_path, keep_ids)
    _write_sample_list(remove_path, remove_ids)

    filtered = run_admixture_mapping(
        phe_path=phe_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_filtered",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
        covar_path=covar_path,
        keep_path=keep_path,
        remove_path=remove_path,
    ).copy()
    obs_ct_vals = pd.to_numeric(filtered["OBS_CT"], errors="coerce").dropna().unique().tolist()
    assert obs_ct_vals == [len(expected_ids)]

    id_to_y = {sid: int(val) for sid, val in zip(sample_ids, y_binary)}
    prefiltered_ids = [sid for sid in sample_ids if sid in expected_ids]
    prefiltered_y = np.asarray([id_to_y[sid] for sid in prefiltered_ids], dtype=np.int8)
    phe_pref_path = tmp_path / "pref.phe"
    _write_phe(phe_pref_path, prefiltered_ids, prefiltered_y)

    reference = run_admixture_mapping(
        phe_path=phe_pref_path,
        msp_path=msp_path,
        results_path=tmp_path / "out_reference",
        phe_id="toy",
        batch_size=64,
        keep_hla=True,
        memory=2048,
        covar_path=covar_path,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ANCESTRY", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P"]
    for col in ("#CHROM", "POS", "END"):
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce").astype(int)
        reference[col] = pd.to_numeric(reference[col], errors="coerce").astype(int)
    for col in num_cols:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
        reference[col] = pd.to_numeric(reference[col], errors="coerce")

    filtered = filtered.sort_values(key_cols).reset_index(drop=True)
    reference = reference.sort_values(key_cols).reset_index(drop=True)
    assert len(filtered) == len(reference)
    np.testing.assert_allclose(
        filtered[num_cols].to_numpy(),
        reference[num_cols].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
