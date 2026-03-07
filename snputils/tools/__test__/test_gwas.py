from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

from snputils.snp.genobj import SNPObject
from snputils.tools.gwas import run_gwas


def _write_vcf(
    path: Path,
    sample_ids: Sequence[str],
    dosage: np.ndarray,
    chromosomes: Sequence[object],
    positions: Sequence[int],
    variant_ids: Sequence[str],
    refs: Sequence[str],
    alts: Sequence[str],
) -> None:
    gt_map = {0: "0|0", 1: "0|1", 2: "1|1"}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(sample_ids))
        handle.write("\n")

        n_variants = int(dosage.shape[0])
        for vidx in range(n_variants):
            genotypes = [gt_map[int(g)] for g in dosage[vidx].tolist()]
            fields = [
                str(chromosomes[vidx]),
                str(int(positions[vidx])),
                str(variant_ids[vidx]),
                str(refs[vidx]),
                str(alts[vidx]),
                ".",
                "PASS",
                ".",
                "GT",
                *genotypes,
            ]
            handle.write("\t".join(fields))
            handle.write("\n")


def _write_binary_phe(path: Path, sample_ids: Sequence[str], y_binary: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID PHENO\n")
        for sid, yi in zip(sample_ids, y_binary):
            status = 2 if int(yi) == 1 else 1
            handle.write(f"{sid} {sid} {status}\n")


def _write_quantitative_phe(path: Path, sample_ids: Sequence[str], y: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID PHENO\n")
        for sid, yi in zip(sample_ids, y):
            handle.write(f"{sid} {sid} {float(yi):.12g}\n")


def _write_multi_phe(
    path: Path,
    sample_ids: Sequence[str],
    names: Sequence[str],
    columns: Sequence[np.ndarray],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID " + " ".join(names) + "\n")
        for row_idx, sid in enumerate(sample_ids):
            values = []
            for column in columns:
                value = column[row_idx]
                if np.issubdtype(np.asarray(column).dtype, np.integer):
                    values.append(str(int(value)))
                else:
                    values.append(f"{float(value):.12g}")
            handle.write(f"{sid} {sid} {' '.join(values)}\n")


def _write_covar(path: Path, sample_ids: Sequence[str], names: Sequence[str], covar: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID " + " ".join(names) + "\n")
        for sid, row in zip(sample_ids, covar):
            values = " ".join(f"{float(v):.12g}" for v in row.tolist())
            handle.write(f"{sid} {sid} {values}\n")


def _write_sample_list(path: Path, sample_ids: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for sid in sample_ids:
            handle.write(f"{sid} {sid}\n")


def _write_variant_list(path: Path, selectors: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for selector in selectors:
            handle.write(f"{selector}\n")


def _make_snpobj_from_dosage(
    sample_ids: Sequence[str],
    dosage: np.ndarray,
    chromosomes: Sequence[object],
    positions: Sequence[int],
    variant_ids: Sequence[str],
    refs: Sequence[str],
    alts: Sequence[str],
) -> SNPObject:
    maternal = (dosage == 2).astype(np.int8)
    paternal = (dosage >= 1).astype(np.int8)
    gt = np.stack([maternal, paternal], axis=2).astype(np.int8, copy=False)
    n_variants = int(dosage.shape[0])
    return SNPObject(
        calldata_gt=gt,
        samples=np.asarray(sample_ids, dtype=str),
        variants_ref=np.asarray(refs, dtype=str),
        variants_alt=np.asarray(alts, dtype=str),
        variants_chrom=np.asarray(chromosomes, dtype=str),
        variants_filter_pass=np.ones(n_variants, dtype=bool),
        variants_id=np.asarray(variant_ids, dtype=str),
        variants_pos=np.asarray(positions, dtype=np.int64),
    )


def _write_raw_gt_vcf(
    path: Path,
    sample_ids: Sequence[str],
    gt_rows: Sequence[Sequence[str]],
    chromosomes: Sequence[object],
    positions: Sequence[int],
    variant_ids: Sequence[str],
    refs: Sequence[str],
    alts: Sequence[str],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(sample_ids))
        handle.write("\n")
        for idx, gt_row in enumerate(gt_rows):
            fields = [
                str(chromosomes[idx]),
                str(int(positions[idx])),
                str(variant_ids[idx]),
                str(refs[idx]),
                str(alts[idx]),
                ".",
                "PASS",
                ".",
                "GT",
                *[str(gt) for gt in gt_row],
            ]
            handle.write("\t".join(fields))
            handle.write("\n")


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


def _build_synthetic_gwas(
    n_samples: int,
    n_variants: int,
    seed: int,
) -> Tuple[Sequence[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_ids = [f"s{i}" for i in range(n_samples)]

    maf = rng.uniform(0.08, 0.42, size=n_variants)
    dosage = np.empty((n_variants, n_samples), dtype=np.int8)
    for vidx in range(n_variants):
        while True:
            g = rng.binomial(2, maf[vidx], size=n_samples).astype(np.int8)
            if np.unique(g).size > 1:
                dosage[vidx] = g
                break

    chromosomes = np.array([(vidx % 2) + 1 for vidx in range(n_variants)], dtype=object)
    positions = np.arange(1, n_variants + 1, dtype=np.int64) * 10_000
    variant_ids = np.array([f"rs{vidx + 1}" for vidx in range(n_variants)], dtype=object)
    refs = np.array(["A" if vidx % 2 == 0 else "C" for vidx in range(n_variants)], dtype=object)
    alts = np.array(["G" if vidx % 2 == 0 else "T" for vidx in range(n_variants)], dtype=object)
    return sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts


def test_gwas_binary_matches_reference_logistic(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=160,
        n_variants=40,
        seed=303,
    )
    rng = np.random.default_rng(304)

    linpred = (
        -0.35
        + 0.55 * dosage[2].astype(np.float64)
        - 0.45 * dosage[11].astype(np.float64)
        + 0.30 * dosage[19].astype(np.float64)
    )
    prob = expit(linpred)
    y_binary = rng.binomial(1, prob).astype(np.int8)
    assert int(np.sum(y_binary)) > 0
    assert int(np.sum(y_binary)) < y_binary.size

    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_path, sample_ids, y_binary)

    results = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out",
        phe_id="toy",
        batch_size=7,
        memory=2048,
    ).copy()
    results = results[(results["ERRCODE"] == ".") & (results["TEST"] == "ADD")].copy()
    results["BETA"] = pd.to_numeric(results["BETA"], errors="coerce")
    results["Z_STAT"] = pd.to_numeric(results["Z_STAT"], errors="coerce")
    results["P"] = pd.to_numeric(results["P"], errors="coerce")
    results = results.dropna(subset=["BETA", "Z_STAT", "P"])

    reference_rows = []
    for vidx in range(dosage.shape[0]):
        ref = _reference_logistic_from_dosage(dosage[vidx], y_binary)
        if ref is None:
            continue
        beta, z_stat, p_val = ref
        reference_rows.append(
            {
                "ID": str(variant_ids[vidx]),
                "BETA_ref": beta,
                "Z_ref": z_stat,
                "P_ref": p_val,
            }
        )
    reference = pd.DataFrame(reference_rows)
    merged = results.merge(reference, on="ID", how="inner")

    assert not merged.empty
    np.testing.assert_allclose(merged["BETA"].to_numpy(), merged["BETA_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["Z_STAT"].to_numpy(), merged["Z_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["P"].to_numpy(), merged["P_ref"].to_numpy(), rtol=3e-2, atol=1e-8)


def test_gwas_quantitative_covariates_matches_reference(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=150,
        n_variants=36,
        seed=404,
    )
    rng = np.random.default_rng(405)

    covar_names = ["PC1", "PC2", "AGE"]
    covar_matrix = rng.normal(size=(len(sample_ids), len(covar_names))).astype(np.float64)
    y = (
        0.65 * dosage[4].astype(np.float64)
        - 0.45 * dosage[13].astype(np.float64)
        + 0.55 * covar_matrix[:, 0]
        - 0.35 * covar_matrix[:, 1]
        + rng.normal(scale=0.9, size=len(sample_ids))
    )

    vcf_path = tmp_path / "toy_quant.vcf"
    phe_path = tmp_path / "toy_quant.phe"
    covar_path = tmp_path / "toy_quant.covar"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_quantitative_phe(phe_path, sample_ids, y)
    _write_covar(covar_path, sample_ids, covar_names, covar_matrix)

    results = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out_quant",
        phe_id="toy",
        batch_size=6,
        memory=2048,
        covar_path=covar_path,
        covar_col_nums="1-3",
        ci=0.95,
        adjust=True,
    ).copy()
    assert "L95" in results.columns
    assert "U95" in results.columns
    assert "BONF" in results.columns
    assert "FDR_BH" in results.columns

    results = results[results["ERRCODE"] == "."].copy()
    results["BETA"] = pd.to_numeric(results["BETA"], errors="coerce")
    results["T_STAT"] = pd.to_numeric(results["T_STAT"], errors="coerce")
    results["P"] = pd.to_numeric(results["P"], errors="coerce")
    results = results.dropna(subset=["BETA", "T_STAT", "P"])

    reference_rows = []
    for vidx in range(dosage.shape[0]):
        ref = _reference_linear_with_covariates(dosage[vidx], y, covar_matrix)
        if ref is None:
            continue
        beta, t_stat, p_val = ref
        reference_rows.append(
            {
                "ID": str(variant_ids[vidx]),
                "BETA_ref": beta,
                "T_ref": t_stat,
                "P_ref": p_val,
            }
        )
    reference = pd.DataFrame(reference_rows)
    merged = results.merge(reference, on="ID", how="inner")

    assert not merged.empty
    np.testing.assert_allclose(merged["BETA"].to_numpy(), merged["BETA_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["T_STAT"].to_numpy(), merged["T_ref"].to_numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(merged["P"].to_numpy(), merged["P_ref"].to_numpy(), rtol=3e-2, atol=1e-8)


@pytest.mark.parametrize("ext", ["bed", "pgen"])
def test_gwas_bed_and_pgen_match_vcf_results(tmp_path: Path, ext: str):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=96,
        n_variants=28,
        seed=501,
    )
    rng = np.random.default_rng(502)
    linpred = -0.25 + 0.7 * dosage[5].astype(np.float64) - 0.5 * dosage[17].astype(np.float64)
    y_binary = rng.binomial(1, expit(linpred)).astype(np.int8)
    assert 0 < int(np.sum(y_binary)) < y_binary.size

    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_path, sample_ids, y_binary)

    snpobj = _make_snpobj_from_dosage(sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    if ext == "bed":
        snp_path = tmp_path / "toy.bed"
        snpobj.save_bed(snp_path)
    else:
        snp_path = tmp_path / "toy.pgen"
        snpobj.save_pgen(snp_path)

    baseline = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out_vcf",
        phe_id="toy",
        batch_size=6,
        memory=2048,
    ).copy()
    alt_input = run_gwas(
        phe_path=phe_path,
        snp_path=snp_path,
        results_path=tmp_path / f"out_{ext}",
        phe_id="toy",
        batch_size=6,
        memory=2048,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P"]
    for col in ("POS", "END"):
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce").astype(int)
        alt_input[col] = pd.to_numeric(alt_input[col], errors="coerce").astype(int)
    baseline["#CHROM"] = baseline["#CHROM"].astype(str)
    alt_input["#CHROM"] = alt_input["#CHROM"].astype(str)
    for col in num_cols:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
        alt_input[col] = pd.to_numeric(alt_input[col], errors="coerce")

    baseline = baseline.sort_values(key_cols).reset_index(drop=True)
    alt_input = alt_input.sort_values(key_cols).reset_index(drop=True)
    assert len(baseline) == len(alt_input)
    np.testing.assert_allclose(
        baseline[num_cols].to_numpy(),
        alt_input[num_cols].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_gwas_keep_remove_filtering_matches_prefiltered_inputs(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=110,
        n_variants=24,
        seed=610,
    )
    rng = np.random.default_rng(611)
    y_binary = rng.binomial(
        1,
        expit(-0.15 + 0.55 * dosage[3].astype(np.float64) - 0.35 * dosage[12].astype(np.float64)),
    ).astype(np.int8)
    assert 0 < int(np.sum(y_binary)) < y_binary.size

    covar = rng.normal(size=(len(sample_ids), 2)).astype(np.float64)
    covar_names = ["PC1", "PC2"]

    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    covar_path = tmp_path / "toy.covar"
    keep_path = tmp_path / "keep.txt"
    remove_path = tmp_path / "remove.txt"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_path, sample_ids, y_binary)
    _write_covar(covar_path, sample_ids, covar_names, covar)

    keep_ids = sample_ids[:84]
    remove_ids = sample_ids[26:36]
    expected_ids = set(keep_ids) - set(remove_ids)
    _write_sample_list(keep_path, keep_ids)
    _write_sample_list(remove_path, remove_ids)

    filtered = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out_filtered",
        phe_id="toy",
        batch_size=8,
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
    _write_binary_phe(phe_pref_path, prefiltered_ids, prefiltered_y)

    reference = run_gwas(
        phe_path=phe_pref_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out_reference",
        phe_id="toy",
        batch_size=8,
        memory=2048,
        covar_path=covar_path,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P"]
    for col in ("POS", "END"):
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce").astype(int)
        reference[col] = pd.to_numeric(reference[col], errors="coerce").astype(int)
    filtered["#CHROM"] = filtered["#CHROM"].astype(str)
    reference["#CHROM"] = reference["#CHROM"].astype(str)
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


def test_gwas_selects_requested_phenotype_from_multi_phenotype_file(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=96,
        n_variants=20,
        seed=640,
    )
    rng = np.random.default_rng(641)
    y1 = rng.binomial(
        1,
        expit(-0.25 + 0.70 * dosage[2].astype(np.float64) - 0.25 * dosage[9].astype(np.float64)),
    ).astype(np.int8)
    y2 = rng.binomial(
        1,
        expit(0.15 - 0.60 * dosage[5].astype(np.float64) + 0.45 * dosage[13].astype(np.float64)),
    ).astype(np.int8)

    vcf_path = tmp_path / "toy.vcf"
    phe_multi = tmp_path / "toy_multi.phe"
    phe_single = tmp_path / "toy_single.phe"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_multi_phe(phe_multi, sample_ids, ["Y1", "Y2"], [y1, y2])
    _write_binary_phe(phe_single, sample_ids, y2)

    selected = run_gwas(
        phe_path=phe_multi,
        snp_path=vcf_path,
        results_path=tmp_path / "out_selected",
        phe_id="Y2",
        batch_size=6,
        memory=2048,
    ).copy()
    reference = run_gwas(
        phe_path=phe_single,
        snp_path=vcf_path,
        results_path=tmp_path / "out_reference",
        phe_id="toy",
        batch_size=6,
        memory=2048,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P"]
    for col in ("POS", "END"):
        selected[col] = pd.to_numeric(selected[col], errors="coerce").astype(int)
        reference[col] = pd.to_numeric(reference[col], errors="coerce").astype(int)
    selected["#CHROM"] = selected["#CHROM"].astype(str)
    reference["#CHROM"] = reference["#CHROM"].astype(str)
    for col in num_cols:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
        reference[col] = pd.to_numeric(reference[col], errors="coerce")

    selected = selected.sort_values(key_cols).reset_index(drop=True)
    reference = reference.sort_values(key_cols).reset_index(drop=True)
    assert len(selected) == len(reference)
    np.testing.assert_allclose(
        selected[num_cols].to_numpy(),
        reference[num_cols].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_gwas_errors_when_requested_phenotype_missing_from_multi_phenotype_file(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=20,
        n_variants=8,
        seed=650,
    )
    y1 = np.array([0, 1] * 10, dtype=np.int8)
    y2 = np.array([1, 0] * 10, dtype=np.int8)

    vcf_path = tmp_path / "toy.vcf"
    phe_multi = tmp_path / "toy_multi.phe"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_multi_phe(phe_multi, sample_ids, ["Y1", "Y2"], [y1, y2])

    with pytest.raises(ValueError, match="Phenotype column 'toy' not found"):
        run_gwas(
            phe_path=phe_multi,
            snp_path=vcf_path,
            results_path=tmp_path / "out",
            phe_id="toy",
            batch_size=4,
            memory=2048,
        )


def test_gwas_variant_exclusion_matches_prefiltered_variants(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=102,
        n_variants=28,
        seed=660,
    )
    rng = np.random.default_rng(661)
    y_binary = rng.binomial(
        1,
        expit(-0.18 + 0.52 * dosage[4].astype(np.float64) - 0.31 * dosage[15].astype(np.float64)),
    ).astype(np.int8)

    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    exclude_path = tmp_path / "exclude.txt"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_path, sample_ids, y_binary)

    excluded_indices = [3, 11, 19]
    selectors = [
        str(variant_ids[excluded_indices[0]]),
        f"{chromosomes[excluded_indices[1]]}:{int(positions[excluded_indices[1]])}",
        str(int(positions[excluded_indices[2]])),
    ]
    _write_variant_list(exclude_path, selectors)

    filtered = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out_filtered",
        phe_id="toy",
        batch_size=7,
        memory=2048,
        exclude_path=exclude_path,
    ).copy()

    keep_mask = np.ones(len(variant_ids), dtype=bool)
    keep_mask[excluded_indices] = False
    vcf_prefiltered = tmp_path / "toy_prefiltered.vcf"
    _write_vcf(
        vcf_prefiltered,
        sample_ids,
        dosage[keep_mask],
        np.asarray(chromosomes, dtype=object)[keep_mask],
        np.asarray(positions, dtype=np.int64)[keep_mask],
        np.asarray(variant_ids, dtype=object)[keep_mask],
        np.asarray(refs, dtype=object)[keep_mask],
        np.asarray(alts, dtype=object)[keep_mask],
    )

    reference = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_prefiltered,
        results_path=tmp_path / "out_reference",
        phe_id="toy",
        batch_size=7,
        memory=2048,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P"]
    for col in ("POS", "END"):
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce").astype(int)
        reference[col] = pd.to_numeric(reference[col], errors="coerce").astype(int)
    filtered["#CHROM"] = filtered["#CHROM"].astype(str)
    reference["#CHROM"] = reference["#CHROM"].astype(str)
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


def test_gwas_ci_columns_logistic_and_quantitative(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=100,
        n_variants=30,
        seed=712,
    )
    rng = np.random.default_rng(713)

    y_binary = rng.binomial(
        1,
        expit(-0.2 + 0.6 * dosage[1].astype(np.float64) - 0.45 * dosage[7].astype(np.float64)),
    ).astype(np.int8)
    vcf_bin = tmp_path / "bin.vcf"
    phe_bin = tmp_path / "bin.phe"
    _write_vcf(vcf_bin, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_bin, sample_ids, y_binary)

    logistic = run_gwas(
        phe_path=phe_bin,
        snp_path=vcf_bin,
        results_path=tmp_path / "out_bin",
        phe_id="toy",
        batch_size=5,
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

    covar_names = ["PC1", "PC2", "AGE"]
    covar = rng.normal(size=(len(sample_ids), 3)).astype(np.float64)
    y_quant = (
        0.5 * dosage[4].astype(np.float64)
        - 0.4 * dosage[16].astype(np.float64)
        + 0.45 * covar[:, 0]
        - 0.3 * covar[:, 1]
        + rng.normal(scale=0.8, size=len(sample_ids))
    )
    phe_q = tmp_path / "quant.phe"
    covar_q = tmp_path / "quant.covar"
    _write_quantitative_phe(phe_q, sample_ids, y_quant)
    _write_covar(covar_q, sample_ids, covar_names, covar)

    linear = run_gwas(
        phe_path=phe_q,
        snp_path=vcf_bin,
        results_path=tmp_path / "out_quant",
        phe_id="toy",
        batch_size=5,
        memory=2048,
        covar_path=covar_q,
        ci=0.95,
    ).copy()
    assert "L95" in linear.columns
    assert "U95" in linear.columns
    linear["BETA"] = pd.to_numeric(linear["BETA"], errors="coerce")
    linear["SE"] = pd.to_numeric(linear["SE"], errors="coerce")
    linear["L95"] = pd.to_numeric(linear["L95"], errors="coerce")
    linear["U95"] = pd.to_numeric(linear["U95"], errors="coerce")
    mask = np.isfinite(linear["BETA"]) & np.isfinite(linear["SE"])
    obs_ct = int(pd.to_numeric(linear["OBS_CT"], errors="coerce").dropna().iloc[0])
    df = obs_ct - (2 + covar.shape[1])
    t_crit = stats.t.ppf(0.975, df=df)
    expected_l = linear.loc[mask, "BETA"] - t_crit * linear.loc[mask, "SE"]
    expected_u = linear.loc[mask, "BETA"] + t_crit * linear.loc[mask, "SE"]
    np.testing.assert_allclose(linear.loc[mask, "L95"].to_numpy(), expected_l.to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(linear.loc[mask, "U95"].to_numpy(), expected_u.to_numpy(), rtol=1e-10, atol=1e-12)


def test_gwas_adjustment_columns_match_reference(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=94,
        n_variants=22,
        seed=815,
    )
    rng = np.random.default_rng(816)
    y_binary = rng.binomial(
        1,
        expit(-0.22 + 0.62 * dosage[2].astype(np.float64) - 0.28 * dosage[14].astype(np.float64)),
    ).astype(np.int8)
    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_path, sample_ids, y_binary)

    adjusted = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out",
        phe_id="toy",
        batch_size=7,
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


def test_gwas_errors_on_missing_genotypes(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2"]
    gt_rows = [
        ["0|0", "0|1", "./."],
        ["0|1", "1|1", "0|0"],
    ]
    chromosomes = [1, 1]
    positions = [100, 200]
    variant_ids = ["rs1", "rs2"]
    refs = ["A", "C"]
    alts = ["G", "T"]
    vcf_path = tmp_path / "missing.vcf"
    phe_path = tmp_path / "missing.phe"
    _write_raw_gt_vcf(vcf_path, sample_ids, gt_rows, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_path, sample_ids, np.array([0, 1, 0], dtype=np.int8))

    with pytest.raises(ValueError, match="requires diploid dosages encoded as 0/1/2"):
        run_gwas(
            phe_path=phe_path,
            snp_path=vcf_path,
            results_path=tmp_path / "out",
            phe_id="toy",
            batch_size=4,
            memory=2048,
        )


def test_gwas_errors_when_no_sample_overlap(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=8,
        n_variants=4,
        seed=910,
    )
    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    other_ids = [f"x{i}" for i in range(8)]
    _write_binary_phe(phe_path, other_ids, np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8))

    with pytest.raises(ValueError, match="No overlapping samples between phenotype and SNP input"):
        run_gwas(
            phe_path=phe_path,
            snp_path=vcf_path,
            results_path=tmp_path / "out",
            phe_id="toy",
            batch_size=4,
            memory=2048,
        )


@pytest.mark.parametrize(
    "keep_ids, expected_error",
    [
        (["s2", "s3"], "No controls after SNP/PHE sample intersection"),
        (["s0", "s1"], "No cases after SNP/PHE sample intersection"),
    ],
)
def test_gwas_errors_for_case_control_collapse_after_filtering(
    tmp_path: Path, keep_ids: Sequence[str], expected_error: str
):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=4,
        n_variants=4,
        seed=915,
    )
    y_binary = np.array([0, 0, 1, 1], dtype=np.int8)
    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    keep_path = tmp_path / "keep.txt"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)
    _write_binary_phe(phe_path, sample_ids, y_binary)
    _write_sample_list(keep_path, keep_ids)

    with pytest.raises(ValueError, match=expected_error):
        run_gwas(
            phe_path=phe_path,
            snp_path=vcf_path,
            results_path=tmp_path / "out",
            phe_id="toy",
            batch_size=4,
            memory=2048,
            keep_path=keep_path,
        )


def test_gwas_alignment_is_invariant_to_phenotype_and_covar_order(tmp_path: Path):
    sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts = _build_synthetic_gwas(
        n_samples=90,
        n_variants=26,
        seed=1001,
    )
    rng = np.random.default_rng(1002)
    covar_names = ["PC1", "PC2", "AGE"]
    covar = rng.normal(size=(len(sample_ids), 3)).astype(np.float64)
    y_quant = (
        0.55 * dosage[3].astype(np.float64)
        - 0.35 * dosage[15].astype(np.float64)
        + 0.50 * covar[:, 0]
        - 0.25 * covar[:, 1]
        + rng.normal(scale=0.8, size=len(sample_ids))
    )

    vcf_path = tmp_path / "toy.vcf"
    _write_vcf(vcf_path, sample_ids, dosage, chromosomes, positions, variant_ids, refs, alts)

    phe_ordered = tmp_path / "ordered.phe"
    covar_ordered = tmp_path / "ordered.covar"
    _write_quantitative_phe(phe_ordered, sample_ids, y_quant)
    _write_covar(covar_ordered, sample_ids, covar_names, covar)

    perm = rng.permutation(len(sample_ids))
    shuffled_ids = [sample_ids[int(i)] for i in perm.tolist()]
    y_shuffled = y_quant[perm]
    covar_shuffled = covar[perm]
    phe_shuffled = tmp_path / "shuffled.phe"
    covar_shuffled_path = tmp_path / "shuffled.covar"
    _write_quantitative_phe(phe_shuffled, shuffled_ids, y_shuffled)
    _write_covar(covar_shuffled_path, shuffled_ids, covar_names, covar_shuffled)

    ordered = run_gwas(
        phe_path=phe_ordered,
        snp_path=vcf_path,
        results_path=tmp_path / "out_ordered",
        phe_id="toy",
        batch_size=6,
        memory=2048,
        covar_path=covar_ordered,
    ).copy()
    shuffled = run_gwas(
        phe_path=phe_shuffled,
        snp_path=vcf_path,
        results_path=tmp_path / "out_shuffled",
        phe_id="toy",
        batch_size=6,
        memory=2048,
        covar_path=covar_shuffled_path,
    ).copy()

    key_cols = ["#CHROM", "POS", "END", "ID", "TEST", "ERRCODE"]
    num_cols = ["BETA", "SE", "T_STAT", "P"]
    for col in ("POS", "END"):
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce").astype(int)
        shuffled[col] = pd.to_numeric(shuffled[col], errors="coerce").astype(int)
    ordered["#CHROM"] = ordered["#CHROM"].astype(str)
    shuffled["#CHROM"] = shuffled["#CHROM"].astype(str)
    for col in num_cols:
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
        shuffled[col] = pd.to_numeric(shuffled[col], errors="coerce")

    ordered = ordered.sort_values(key_cols).reset_index(drop=True)
    shuffled = shuffled.sort_values(key_cols).reset_index(drop=True)
    assert len(ordered) == len(shuffled)
    np.testing.assert_allclose(
        ordered[num_cols].to_numpy(),
        shuffled[num_cols].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
