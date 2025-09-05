import numpy as np
import pandas as pd

from snputils.stats import f2, f3, f4, d_stat, f4_ratio, fst


def _toy_data():
    # Build simple AFs
    afs = np.array([
        [0.1, 0.1, 0.9, 0.9],
        [0.2, 0.2, 0.8, 0.8],
        [0.3, 0.32, 0.7, 0.7],
        [0.4, 0.48, 0.6, 0.6],
        [0.5, 0.5, 0.5, 0.5],
        [0.6, 0.55, 0.4, 0.35],
    ])
    counts = np.full_like(afs, 20)
    pops = ["A", "B", "C", "D"]
    return afs, counts, pops


def test_f2_basic():
    afs, counts, pops = _toy_data()
    res = f2((afs, counts, pops), block_size=2)
    # Symmetry on diagonal pairs
    ab = res[(res.pop1 == "A") & (res.pop2 == "B")].iloc[0]
    assert np.isfinite(ab.est)
    assert ab.n_blocks == 3
    assert ab.n_snps == 6


def test_f3_basic():
    afs, counts, pops = _toy_data()
    res = f3((afs, counts, pops), target=["A"], ref1=["B"], ref2=["C"], block_size=3)
    assert res.shape[0] == 1
    row = res.iloc[0]
    assert row.target == "A"
    assert np.isfinite(row.est)


def test_f4_and_d_basic():
    afs, counts, pops = _toy_data()
    quads = dict(a=["A"], b=["B"], c=["C"], d=["D"])
    res4 = f4((afs, counts, pops), **quads, block_size=2)
    resd = d_stat((afs, counts, pops), **quads, block_size=2)
    assert res4.shape[0] == 1
    assert resd.shape[0] == 1
    assert np.isfinite(res4.est.iloc[0])
    assert np.isfinite(resd.est.iloc[0])
    

def test_f4_ratio_basic():
    afs, counts, pops = _toy_data()
    num = [("A", "B", "C", "D")]
    den = [("A", "C", "B", "D")]
    res = f4_ratio((afs, counts, pops), num=num, den=den, block_size=2)
    assert res.shape[0] == 1
    assert np.isfinite(res.est.iloc[0])


def test_f4_identities_symmetry_and_signs():
    afs, counts, pops = _toy_data()
    base = f4((afs, counts, pops), a=["A"], b=["B"], c=["C"], d=["D"], block_size=2).est.iloc[0]
    swapped_pairs = f4((afs, counts, pops), a=["C"], b=["D"], c=["A"], d=["B"], block_size=2).est.iloc[0]
    swapped_ab = f4((afs, counts, pops), a=["B"], b=["A"], c=["C"], d=["D"], block_size=2).est.iloc[0]
    swapped_cd = f4((afs, counts, pops), a=["A"], b=["B"], c=["D"], d=["C"], block_size=2).est.iloc[0]

    assert np.isclose(base, swapped_pairs, atol=1e-12)
    assert np.isclose(base, -swapped_ab, atol=1e-12)
    assert np.isclose(base, -swapped_cd, atol=1e-12)


def test_f4_identity_additivity():
    afs, counts, pops = _toy_data()
    ab_cd = f4((afs, counts, pops), a=["A"], b=["B"], c=["C"], d=["D"], block_size=2).est.iloc[0]
    ac_bd = f4((afs, counts, pops), a=["A"], b=["C"], c=["B"], d=["D"], block_size=2).est.iloc[0]
    ad_cb = f4((afs, counts, pops), a=["A"], b=["D"], c=["C"], d=["B"], block_size=2).est.iloc[0]

    assert np.isclose(ab_cd, ac_bd + ad_cb, atol=1e-12)


def test_f2_equals_f4_self_definition():
    afs, counts, pops = _toy_data()
    f2_ab = f2((afs, counts, pops), pop1=["A"], pop2=["B"], apply_correction=False, block_size=2).est.iloc[0]
    f4_abab = f4((afs, counts, pops), a=["A"], b=["B"], c=["A"], d=["B"], block_size=2).est.iloc[0]
    assert np.isclose(f2_ab, f4_abab, atol=1e-12)


def test_f3_equals_f4_cross_definition():
    afs, counts, pops = _toy_data()
    f3_a_bc = f3((afs, counts, pops), target=["A"], ref1=["B"], ref2=["C"], apply_correction=False, block_size=2).est.iloc[0]
    f4_abac = f4((afs, counts, pops), a=["A"], b=["B"], c=["A"], d=["C"], block_size=2).est.iloc[0]
    assert np.isclose(f3_a_bc, f4_abac, atol=1e-12)
    

def test_f3_corrected_equals_f4_minus_target_term():
    afs, counts, pops = _toy_data()
    t, a, b = "A", "B", "C"
    res_unc = f3((afs, counts, pops), target=[t], ref1=[a], ref2=[b],
                 apply_correction=False, block_size=2).est.iloc[0]
    res_cor = f3((afs, counts, pops), target=[t], ref1=[a], ref2=[b],
                 apply_correction=True, block_size=2).est.iloc[0]
    # All counts are 20 in _toy_data, so the mean correction is mean[p_t(1-p_t)/(20-1)]
    ti = pops.index(t)
    corr = np.mean(afs[:, ti] * (1 - afs[:, ti]) / (counts[:, ti] - 1))
    assert np.isclose(res_unc - corr, res_cor, atol=1e-12)


def test_f4_admixture_linearity():
    # Construct synthetic data with an admixed population Cmix = alpha*C1 + (1-alpha)*C2
    alpha = 0.3
    # 8 SNPs, 6 populations: A, B, C1, C2, Cmix, D
    A = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    B = np.array([0.12, 0.18, 0.31, 0.39, 0.52, 0.58, 0.69, 0.79])
    C1 = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    C2 = np.array([0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25])
    D = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20])
    Cmix = alpha * C1 + (1 - alpha) * C2
    afs = np.vstack([A, B, C1, C2, Cmix, D]).T  # shape (8, 6)
    counts = np.full_like(afs, 20)
    pops = ["A", "B", "C1", "C2", "Cmix", "D"]

    f4_cmix = f4((afs, counts, pops), a=["A"], b=["B"], c=["Cmix"], d=["D"], block_size=2).est.iloc[0]
    f4_c1 = f4((afs, counts, pops), a=["A"], b=["B"], c=["C1"], d=["D"], block_size=2).est.iloc[0]
    f4_c2 = f4((afs, counts, pops), a=["A"], b=["B"], c=["C2"], d=["D"], block_size=2).est.iloc[0]

    rhs = alpha * f4_c1 + (1 - alpha) * f4_c2
    assert np.isclose(f4_cmix, rhs, atol=1e-12)


def test_f4_as_sum_of_f2s():
    # Verify f4(A,B;C,D) = 1/2 [ f2(A,D) + f2(B,C) - f2(A,C) - f2(B,D) ]
    # Use uncorrected f2 to match algebraic identity on allele frequencies.
    afs, counts, pops = _toy_data()

    f4_ab_cd = f4((afs, counts, pops), a=["A"], b=["B"], c=["C"], d=["D"], block_size=2).est.iloc[0]

    f2_ad = f2((afs, counts, pops), pop1=["A"], pop2=["D"], apply_correction=False, block_size=2).est.iloc[0]
    f2_bc = f2((afs, counts, pops), pop1=["B"], pop2=["C"], apply_correction=False, block_size=2).est.iloc[0]
    f2_ac = f2((afs, counts, pops), pop1=["A"], pop2=["C"], apply_correction=False, block_size=2).est.iloc[0]
    f2_bd = f2((afs, counts, pops), pop1=["B"], pop2=["D"], apply_correction=False, block_size=2).est.iloc[0]

    rhs = 0.5 * (f2_ad + f2_bc - f2_ac - f2_bd)
    assert np.isclose(f4_ab_cd, rhs, atol=1e-12)


def test_f2_diagonal_behavior():
    afs, counts, pops = _toy_data()
    # Diagonal with no correction should be exactly zero
    res0 = f2((afs, counts, pops), pop1=["A"], pop2=["A"], apply_correction=False, block_size=2)
    assert np.isclose(res0.est.iloc[0], 0.0, atol=1e-15)
    # With correction, diagonal is not defined in the same sense; just ensure it is finite
    res1 = f2((afs, counts, pops), pop1=["A"], pop2=["A"], apply_correction=True, block_size=2)
    assert np.isfinite(res1.est.iloc[0])


def test_f2_correction_filters_n_le_1():
    afs, counts, pops = _toy_data()
    counts2 = counts.copy()
    counts2[0, 0] = 1  # SNP 0 pop A has n=1
    counts2[1, 1] = 1  # SNP 1 pop B has n=1
    r0 = f2((afs, counts2, pops), pop1=["A"], pop2=["B"], apply_correction=False, block_size=2).iloc[0]
    r1 = f2((afs, counts2, pops), pop1=["A"], pop2=["B"], apply_correction=True, block_size=2).iloc[0]
    assert r1.n_snps < r0.n_snps
    assert np.isfinite(r1.est)


def test_f3_correction_filters_target_n_le_1():
    afs, counts, pops = _toy_data()
    counts2 = counts.copy()
    counts2[0, pops.index("A")] = 1  # target A has n=1 at SNP 0
    r0 = f3((afs, counts2, pops), target=["A"], ref1=["B"], ref2=["C"], apply_correction=False, block_size=2).iloc[0]
    r1 = f3((afs, counts2, pops), target=["A"], ref1=["B"], ref2=["C"], apply_correction=True, block_size=2).iloc[0]
    assert r1.n_snps < r0.n_snps


def test_f4_trivial_zero_and_sign():
    afs, counts, pops = _toy_data()
    # If A==B or C==D, f4 should be 0
    z1 = f4((afs, counts, pops), a=["A"], b=["A"], c=["C"], d=["D"], block_size=2).est.iloc[0]
    z2 = f4((afs, counts, pops), a=["A"], b=["B"], c=["C"], d=["C"], block_size=2).est.iloc[0]
    assert np.isclose(z1, 0.0, atol=1e-15)
    assert np.isclose(z2, 0.0, atol=1e-15)
    # Swap within a pair flips the sign
    base = f4((afs, counts, pops), a=["A"], b=['B'], c=['C'], d=['D'], block_size=2).est.iloc[0]
    flip = f4((afs, counts, pops), a=["B"], b=['A'], c=['C'], d=['D'], block_size=2).est.iloc[0]
    assert np.isclose(base, -flip, atol=1e-15)


def test_f4_ratio_negative_denominator():
    afs, counts, pops = _toy_data()
    # Make denominator f4 likely negative by swapping references
    num = [("A", "B", "C", "D")]
    den = [("A", "B", "D", "C")]
    res = f4_ratio((afs, counts, pops), num=num, den=den, block_size=2).iloc[0]
    assert np.isfinite(res.est)
    # Denominator should be near -numerator in this toy, so ratio near -1
    assert res.est < 0


def test_empty_block_is_dropped():
    afs, counts, pops = _toy_data()
    # Make SNPs 0-1 invalid for A or B to wipe out block 0 when block_size=2
    counts2 = counts.copy()
    counts2[0:2, pops.index("A")] = 0
    res = f2((afs, counts2, pops), pop1=["A"], pop2=["B"], apply_correction=False, block_size=2).iloc[0]
    # Originally 3 blocks. Now first block is empty, so contributing blocks should be 2
    assert res.n_blocks == 2


def test_results_invariant_to_snp_order_with_labels():
    afs, counts, pops = _toy_data()
    n = afs.shape[0]
    labels = np.array([0,0,1,1,2,2])
    base = f4((afs, counts, pops), a=["A"], b=["B"], c=["C"], d=["D"], blocks=labels).est.iloc[0]
    perm = np.random.permutation(n)
    base2 = f4((afs[perm], counts[perm], pops), a=["A"], b=["B"], c=["C"], d=["D"], blocks=labels[perm]).est.iloc[0]
    assert np.isclose(base, base2, atol=1e-15)


def test_missing_afs_and_zero_counts_are_ignored():
    afs, counts, pops = _toy_data()
    afs2 = afs.copy().astype(float)
    afs2[3, pops.index("C")] = np.nan
    counts2 = counts.copy()
    counts2[4, pops.index("D")] = 0
    res = f4((afs2, counts2, pops), a=["A"], b=["B"], c=["C"], d=["D"], block_size=2).iloc[0]
    assert np.isfinite(res.est)
    # Should have fewer SNPs than the original 6
    assert res.n_snps < 6


def test_d_stat_zero_when_pairs_equal():
    afs, counts, pops = _toy_data()
    # If C==D, numerator is zero for every SNP
    res = d_stat((afs, counts, pops), a=["A"], b=["B"], c=["C"], d=["C"], block_size=2).iloc[0]
    assert np.isclose(res.est, 0.0, atol=1e-15)


def test_all_snps_invalid_returns_nan():
    afs, counts, pops = _toy_data()
    counts2 = counts.copy()
    counts2[:, :] = 0
    res = f4((afs, counts2, pops), a=["A"], b=["B"], c=["C"], d=["D"]).iloc[0]
    assert np.isnan(res.est) and np.isnan(res.se) and np.isnan(res.z) and np.isnan(res.p)
    assert res.n_blocks == 0 and res.n_snps == 0


def test_fst_hudson_basic():
    afs, counts, pops = _toy_data()
    res = fst((afs, counts, pops), pop1=["A"], pop2=["B"], method="hudson", block_size=2)
    assert res.shape[0] == 1
    row = res.iloc[0]
    assert row.method == "hudson"
    assert np.isfinite(row.est)
    assert row.n_blocks == 3
    assert row.n_snps == 6


def test_fst_weir_cockerham_basic():
    afs, counts, pops = _toy_data()
    res = fst((afs, counts, pops), pop1=["A"], pop2=["B"], method="weir_cockerham", block_size=2)
    assert res.shape[0] == 1
    row = res.iloc[0]
    assert row.method == "weir_cockerham"
    assert np.isfinite(row.est)
    assert row.n_blocks == 3
    assert row.n_snps == 6


def test_fst_all_pairs_default():
    afs, counts, pops = _toy_data()
    res = fst((afs, counts, pops), method="hudson", block_size=2)
    # 4 pops → 6 unordered pairs
    assert res.shape[0] == 6
    assert set(res.method.unique()) == {"hudson"}


def test_fst_invalid_method_raises():
    afs, counts, pops = _toy_data()
    try:
        fst((afs, counts, pops), pop1=["A"], pop2=["B"], method="not-a-method")
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_fst_ignores_snp_with_too_small_n():
    afs, counts, pops = _toy_data()
    counts2 = counts.copy()
    # Make first SNP invalid in A (n=1) so it is dropped for both methods
    counts2[0, pops.index("A")] = 1
    r_h = fst((afs, counts2, pops), pop1=["A"], pop2=["B"], method="hudson", block_size=2).iloc[0]
    r_w = fst((afs, counts2, pops), pop1=["A"], pop2=["B"], method="weir_cockerham", block_size=2).iloc[0]
    assert r_h.n_snps < 6 and r_w.n_snps < 6
    assert np.isfinite(r_h.est) and np.isfinite(r_w.est)


def test_fst_blocks_with_labels_invariant():
    afs, counts, pops = _toy_data()
    n = afs.shape[0]
    labels = np.array([0,0,1,1,2,2])
    base = fst((afs, counts, pops), pop1=["A"], pop2=["B"], method="hudson", blocks=labels).est.iloc[0]
    perm = np.random.permutation(n)
    alt  = fst((afs[perm], counts[perm], pops), pop1=["A"], pop2=["B"], method="hudson", blocks=labels[perm]).est.iloc[0]
    assert np.isclose(base, alt, atol=1e-15)

def test_fst_weir_cockerham_matches_manual_ratio_of_sums():
    afs = np.array([[0.10, 0.90],
                    [0.25, 0.75],
                    [0.40, 0.60]], dtype=float)
    counts = np.array([[40, 30],
                       [40, 30],
                       [40, 30]], dtype=float)
    pops = ["X", "Y"]

    p1, p2 = afs[:, 0], afs[:, 1]
    n1, n2 = counts[:, 0], counts[:, 1]
    n = n1 + n2
    n_bar = n / 2.0
    p_bar = (n1 * p1 + n2 * p2) / n
    s2 = (n1 * (p1 - p_bar) ** 2 + n2 * (p2 - p_bar) ** 2) / n_bar
    h1 = 2 * p1 * (1 - p1)
    h2 = 2 * p2 * (1 - p2)
    h_bar = 0.5 * (h1 + h2)
    n_c = n - (n1 * n1 + n2 * n2) / n  # == 2*n1*n2/n for r=2

    a = (n_bar / n_c) * (s2 - (p_bar * (1 - p_bar) - 0.5 * s2 - 0.25 * h_bar) / (n_bar - 1))
    b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar) - 0.5 * s2 - ((2 * n_bar - 1) / (4 * n_bar)) * h_bar)
    c = 0.5 * h_bar

    expected = np.nansum(a) / np.nansum(a + b + c)

    res = fst((afs, counts, pops), pop1=["X"], pop2=["Y"], method="weir_cockerham", block_size=3).est.iloc[0]
    assert np.isclose(res, expected, rtol=1e-12, atol=1e-12)

def test_fst_hudson_identical_populations_expected_bias():
    # Two identical pops across SNPs; equal haplotype counts per pop (n=40)
    afs = np.array([[0.1, 0.1],
                    [0.3, 0.3],
                    [0.7, 0.7]], dtype=float)
    counts = np.full_like(afs, 40.0)  # haplotype counts, equal across SNPs
    pops = ["X", "Y"]

    est = fst((afs, counts, pops), pop1=["X"], pop2=["Y"], method="hudson", block_size=2).est.iloc[0]
    expected = -1.0 / (40.0 - 1.0)   # = -1/39
    assert np.isclose(est, expected, atol=1e-12)


def test_fst_weir_cockerham_identical_populations_expected_bias():
    # Same setup; WC has half the finite-sample bias of Hudson in this scenario
    afs = np.array([[0.1, 0.1],
                    [0.3, 0.3],
                    [0.7, 0.7]], dtype=float)
    counts = np.full_like(afs, 40.0)
    pops = ["X", "Y"]

    est = fst((afs, counts, pops), pop1=["X"], pop2=["Y"], method="weir_cockerham", block_size=2).est.iloc[0]
    expected = -0.5 / (40.0 - 1.0)   # = -1/(2*39)
    assert np.isclose(est, expected, atol=1e-12)


def test_fst_hudson_equals_ratio_of_sums_of_f2_and_within_hets():
    afs, counts, pops = _toy_data()
    # Use A and B from toy; all counts are 20 so masks align
    pop1 = ["A"]
    pop2 = ["B"]
    # Fst (Hudson)
    fst_row = fst((afs, counts, pops), pop1=pop1, pop2=pop2, method="hudson", block_size=2).iloc[0]
    # f2 sums: corrected (numerator) and uncorrected (part of denominator)
    f2_corr_row = f2((afs, counts, pops), pop1=pop1, pop2=pop2, apply_correction=True, block_size=2).iloc[0]
    f2_unc_row = f2((afs, counts, pops), pop1=pop1, pop2=pop2, apply_correction=False, block_size=2).iloc[0]
    num_sum = f2_corr_row.est * f2_corr_row.n_snps
    f2_unc_sum = f2_unc_row.est * f2_unc_row.n_snps
    # Within-pop expected heterozygosities (per-snp p(1-p) terms sum to 0.5*(2p(1-p)) per pop)
    i = pops.index(pop1[0])
    j = pops.index(pop2[0])
    p_i = afs[:, i]
    p_j = afs[:, j]
    within_sum = np.sum(p_i * (1.0 - p_i) + p_j * (1.0 - p_j))
    den_sum = f2_unc_sum + within_sum
    expected = num_sum / den_sum
    assert np.isclose(fst_row.est, expected, rtol=1e-12, atol=1e-12)
