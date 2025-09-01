from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class BlockJackknifeResult:
    est: float
    se: float
    z: float
    p: float
    n_blocks: int
    n_snps: int


def _compute_block_indices(
    n_snps: int,
    size: int = 5000,
    block_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build SNP-to-block assignments using fixed SNP counts per block.

    Returns:
        block_ids (int array of shape (n_snps,)): Block index for each SNP (0..B-1)
        block_lengths (int array of shape (n_blocks,)): Number of SNPs per block
    """
    if block_labels is not None:
        block_labels = np.asarray(block_labels)
        if block_labels.shape[0] != n_snps:
            raise ValueError("'block_labels' must have length n_snps")
        # Reindex to 0..B-1 preserving first appearance order
        codes, uniques = pd.factorize(block_labels, sort=False)
        counts = np.bincount(codes, minlength=len(uniques))
        return codes.astype(np.int32), counts.astype(np.int32)

    size = max(1, int(size))
    block_ids = np.arange(n_snps) // size
    # block lengths: last block may be shorter
    n_blocks = int(block_ids.max()) + 1
    block_lengths = np.bincount(block_ids, minlength=n_blocks)
    return block_ids.astype(np.int32), block_lengths.astype(np.int32)


def _jackknife_ratio_from_block_sums(
    num_block_sums: np.ndarray,
    den_block_sums: np.ndarray,
) -> BlockJackknifeResult:
    """
    Compute statistic = sum(num) / sum(den) and its block-jackknife SE using delete-one blocks.
    """
    num_block_sums = np.asarray(num_block_sums, dtype=float)
    den_block_sums = np.asarray(den_block_sums, dtype=float)

    # Keep negative denominators, drop only non-finite and exactly-zero denominators
    mask = (den_block_sums != 0) & np.isfinite(num_block_sums) & np.isfinite(den_block_sums)
    if not np.any(mask):
        return BlockJackknifeResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)

    nb = int(mask.sum())
    num_b = num_block_sums[mask]
    den_b = den_block_sums[mask]
    num_total = float(np.sum(num_b))
    den_total = float(np.sum(den_b))
    if den_total == 0:
        return BlockJackknifeResult(float("nan"), float("nan"), float("nan"), float("nan"), nb, int(np.sum(den_b)))

    est = num_total / den_total
    # delete-one estimates
    with np.errstate(divide="ignore", invalid="ignore"):
        est_loo = (num_total - num_b) / (den_total - den_b)
    # Remove degenerate cases where denominator becomes zero after deletion
    valid_loo = np.isfinite(est_loo)
    est_loo = est_loo[valid_loo]
    nb2 = est_loo.size
    if nb2 == 0:
        return BlockJackknifeResult(est, 0.0, float("nan"), float("nan"), nb, int(np.sum(den_b)))
    # standard jackknife variance
    se = float(np.sqrt((nb2 - 1) / nb2 * np.sum((est_loo - est) ** 2)))
    z = float(est / se) if se > 0 else float("nan")
    p = float(math.erfc(abs(z) / math.sqrt(2))) if np.isfinite(z) else float("nan")
    return BlockJackknifeResult(est, se, z, p, nb, int(np.sum(den_b)))


def _aggregate_to_pop_allele_freq(
    calldata_gt: np.ndarray,
    sample_labels: Sequence[str],
    *,
    ancestry: Optional[Union[str, int]] = None,
    snpobj: Optional[Any] = None,
    laiobj: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert sample-level genotypes into per-population allele frequency and count matrices.

    Genotype encoding supported:
        - 3D (n_snps, n_samples, 2): haplotype calls in {0,1}, missing as negative or NaN
        - 2D (n_snps, n_samples): diploid dosages in {0,1,2} or haploid in {0,1}; missing as negative or NaN

    Returns:
        afs: float array (n_snps, n_pops)
        counts: int array (n_snps, n_pops) of called haplotypes per SNP and population
        pops: list of population names in column order
    """
    gt = np.asarray(calldata_gt)
    if gt.ndim not in (2, 3):
        raise ValueError("'calldata_gt' must be 2D or 3D array")

    sample_labels = np.asarray(sample_labels)
    pops, pop_indices = np.unique(sample_labels, return_inverse=True)
    n_snps = gt.shape[0]
    n_pops = pops.size

    # Optional ancestry mask (requires haplotype-level genotypes)
    if ancestry is not None:
        laj = None
        if snpobj is not None and getattr(snpobj, "calldata_lai", None) is not None:
            laj = snpobj.calldata_lai
        if laiobj is not None and laj is None:
            try:
                # Prefer 3D LAI to mask haplotypes
                laj = laiobj.convert_to_snp_level(snpobject=snpobj, lai_format="3D").calldata_lai
            except Exception:
                pass
        if laj is None:
            raise ValueError(
                "Ancestry-specific masking requires SNP-level LAI (provide a LocalAncestryObject via 'laiobj' or ensure 'snpobj.calldata_lai' is set)."
            )
        if gt.ndim != 3:
            raise ValueError("Ancestry-specific masking requires 3D genotype array (n_snps, n_samples, 2).")
        # Normalize LAI to 3D
        laj = np.asarray(laj)
        if laj.ndim == 2:
            n_samples = gt.shape[1]
            try:
                laj = laj.reshape(n_snps, n_samples, 2)
            except Exception:
                raise ValueError("LAI shape is incompatible with genotypes. Expected (n_snps, n_samples*2) to reshape.")
        if laj.ndim != 3:
            raise ValueError("LAI must be 3D (n_snps, n_samples, 2) or 2D (n_snps, n_samples*2).")
        # Apply mask: keep entries matching ancestry; others set to NaN
        mask = (laj.astype(str) == str(ancestry))
        gt = gt.astype(float)
        gt[~mask] = np.nan

    # Compute alt allele counts and haplotype counts per SNP and sample
    if gt.ndim == 3:
        # (n_snps, n_samples, 2)
        g = gt.astype(float)
        # treat values < 0 as missing
        missing = g < 0
        g[missing] = np.nan
        alt_counts_per_sample = np.nansum(g, axis=2)  # sum haplotypes per sample
        hap_count_per_sample = 2 - np.sum(np.isnan(g), axis=2)
    else:
        # (n_snps, n_samples)
        g = gt.astype(float)
        missing = g < 0
        g[missing] = np.nan
        all_nan = np.all(np.isnan(g))
        max_val = np.nan if all_nan else np.nanmax(g)
        if all_nan:
            hap_count_per_sample = np.zeros_like(g)
            alt_counts_per_sample = np.zeros_like(g)
        elif max_val <= 1:
            hap_count_per_sample = np.where(np.isnan(g), 0.0, 1.0)
            alt_counts_per_sample = np.where(np.isnan(g), 0.0, g)
        else:
            hap_count_per_sample = np.where(np.isnan(g), 0.0, 2.0)
            alt_counts_per_sample = np.where(np.isnan(g), 0.0, g)

    # Aggregate per population
    afs = np.zeros((n_snps, n_pops), dtype=float)
    counts = np.zeros((n_snps, n_pops), dtype=float)
    for k in range(n_pops):
        cols = np.where(pop_indices == k)[0]
        if cols.size == 0:
            continue
        counts[:, k] = np.sum(hap_count_per_sample[:, cols], axis=1)
        alt_sum = np.sum(alt_counts_per_sample[:, cols], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            afs[:, k] = np.where(counts[:, k] > 0, alt_sum / counts[:, k], np.nan)
    return afs, counts.astype(int), pops.tolist()


def _prepare_inputs(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    sample_labels: Optional[Sequence[str]] = None,
    *,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Normalize inputs to allele frequencies, counts, and optional variant metadata.
    """
    try:
        from snputils.snp.genobj.snpobj import SNPObject  # type: ignore
        is_snpobj = isinstance(data, SNPObject)
    except Exception:
        is_snpobj = False

    if is_snpobj:
        snpobj = data
        if sample_labels is None:
            if snpobj.samples is None:
                raise ValueError("sample_labels must be provided when SNPObject.samples is None")
            sample_labels = snpobj.samples
        afs, counts, pops = _aggregate_to_pop_allele_freq(
            snpobj.calldata_gt,
            sample_labels,
            ancestry=ancestry,
            snpobj=snpobj,
            laiobj=laiobj,
        )
        return afs, counts, pops

    if isinstance(data, tuple) and len(data) == 3:
        afs, counts, pops = data
        return np.asarray(afs), np.asarray(counts), list(pops)

    raise ValueError(
        "data must be either a SNPObject or a tuple (afs, counts, pops) where afs/counts have shape (n_snps, n_pops)"
    )


def _build_blocks(
    n_snps: int,
    blocks: Optional[np.ndarray],
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if blocks is not None:
        blocks = np.asarray(blocks)
        if blocks.shape[0] != n_snps:
            raise ValueError("'blocks' must have length n_snps")
        return _compute_block_indices(n_snps, block_labels=blocks)
    return _compute_block_indices(n_snps=n_snps, size=block_size)


def f2(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    pop1: Optional[Sequence[str]] = None,
    pop2: Optional[Sequence[str]] = None,
    sample_labels: Optional[Sequence[str]] = None,
    apply_correction: bool = True,
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
    include_self: bool = False,
) -> pd.DataFrame:
    """
    Compute f2-statistics with block-jackknife standard errors.

    Args:
        data: Either a SNPObject or a tuple (afs, counts, pops), where `afs` and `counts` are arrays of
              shape (n_snps, n_pops). If a SNPObject is provided, `sample_labels` are used to aggregate samples to populations.
        pop1, pop2: Populations to compute f2 for. 
            If None:
                - with include_self=False (default), compute only off-diagonal pairs i<j.
                - with include_self=True, compute all pairs including diagonals (i<=j).
        sample_labels: Population label per sample (aligned with SNPObject.samples) when `data` is a SNPObject.
        apply_correction: Apply small-sample correction p*(1-p)/(n-1) per population.
            When True, SNPs with n<=1 in either population are excluded at that SNP.
        block_size: Number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
        blocks: Optional explicit block id per SNP. If provided, overrides `block_size`.
        ancestry: Optional ancestry code to mask genotypes to a specific ancestry before aggregation. Requires LAI.
        laiobj: Optional `LocalAncestryObject` used to derive SNP-level LAI if it is not already present on the SNPObject.

    Returns:
        Pandas DataFrame with columns: pop1, pop2, est, se, z, p, n_blocks, n_snps
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj)
    n_snps, n_pops = afs.shape
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    n_blocks = block_lengths.size

    if pop1 is None and pop2 is None:
        if include_self:
            pop_pairs = [(pops[i], pops[j]) for i in range(n_pops) for j in range(i, n_pops)]
        else:
            pop_pairs = [(pops[i], pops[j]) for i in range(n_pops) for j in range(i + 1, n_pops)]
    else:
        if pop1 is None or pop2 is None:
            raise ValueError("Both pop1 and pop2 must be provided when one is provided")
        if len(pop1) != len(pop2):
            raise ValueError("pop1 and pop2 must have the same length")
        pop_pairs = list(zip(pop1, pop2))

    name_to_idx = {p: i for i, p in enumerate(pops)}

    rows: List[Dict[str, Union[str, float, int]]] = []
    # Precompute per-block index lists for efficiency
    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]

    for p1, p2 in pop_pairs:
        i = name_to_idx[p1]
        j = name_to_idx[p2]
        p_i = afs[:, i]
        p_j = afs[:, j]
        n_i = counts[:, i].astype(float)
        n_j = counts[:, j].astype(float)

        # per-SNP f2 with optional small-sample correction.
        # If apply_correction is True, SNPs with n<=1 for either population are excluded.
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (p_i - p_j) ** 2
            if apply_correction:
                corr_i = np.where(n_i > 1, (p_i * (1.0 - p_i)) / (n_i - 1.0), np.nan)
                corr_j = np.where(n_j > 1, (p_j * (1.0 - p_j)) / (n_j - 1.0), np.nan)
                num = num - corr_i - corr_j
        # SNPs contribute only if the correction was defined (finite) and AFs present
        if apply_correction:
            snp_mask = np.isfinite(num)
        else:
            snp_mask = np.isfinite(num) & (n_i > 0) & (n_j > 0)

        # Aggregate per block: sums of num and counts. Use NaN to mark empty blocks.
        num_block_sums = np.full(n_blocks, np.nan, dtype=float)
        den_block_sums = np.full(n_blocks, np.nan, dtype=float)
        for b, idx in enumerate(block_bins):
            if idx.size == 0:
                continue
            idx2 = idx[snp_mask[idx]]
            if idx2.size == 0:
                continue
            num_block_sums[b] = float(np.nansum(num[idx2]))
            den_block_sums[b] = float(idx2.size)

        res = _jackknife_ratio_from_block_sums(num_block_sums, den_block_sums)
        rows.append(
            {
                "pop1": p1,
                "pop2": p2,
                "est": res.est,
                "se": res.se,
                "z": res.z,
                "p": res.p,
                "n_blocks": res.n_blocks,
                "n_snps": res.n_snps,
            }
        )

    return pd.DataFrame(rows)


def f3(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    target: Optional[Sequence[str]] = None,
    ref1: Optional[Sequence[str]] = None,
    ref2: Optional[Sequence[str]] = None,
    sample_labels: Optional[Sequence[str]] = None,
    apply_correction: bool = True,
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Compute f3-statistics f3(target; ref1, ref2) with block-jackknife SE.

    - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
    - If `target`, `ref1`, and `ref2` are all None, compute f3 for all combinations where each role can be any population.
    - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
    - If `apply_correction` is True, subtract the finite sample term p_t*(1-p_t)/(n_t-1) from the per-SNP product.
        When True, SNPs with n_t<=1 are excluded.
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj)
    n_snps, _ = afs.shape
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    n_blocks = block_lengths.size

    if target is None and ref1 is None and ref2 is None:
        triples = [(a, b, c) for a in pops for b in pops for c in pops]
    else:
        if target is None or ref1 is None or ref2 is None:
            raise ValueError("target, ref1, and ref2 must all be provided if any is provided")
        if not (len(target) == len(ref1) == len(ref2)):
            raise ValueError("target, ref1, ref2 must have the same length")
        triples = list(zip(target, ref1, ref2))

    name_to_idx = {p: i for i, p in enumerate(pops)}
    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]
    rows: List[Dict[str, Union[str, float, int]]] = []

    for t, r1, r2 in triples:
        it = name_to_idx[t]
        i1 = name_to_idx[r1]
        i2 = name_to_idx[r2]
        pt = afs[:, it]
        p1 = afs[:, i1]
        p2 = afs[:, i2]
        nt = counts[:, it].astype(float)
        n1 = counts[:, i1].astype(float)
        n2 = counts[:, i2].astype(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            num = (pt - p1) * (pt - p2)
            if apply_correction:
                corr_t = np.where(nt > 1, (pt * (1.0 - pt)) / (nt - 1.0), np.nan)
                num = num - corr_t
        if apply_correction:
            # Require a valid correction on the target, references can be n>0
            snp_mask = np.isfinite(num) & (n1 > 0) & (n2 > 0)
        else:
            snp_mask = np.isfinite(num) & (nt > 0) & (n1 > 0) & (n2 > 0)

        num_block_sums = np.full(n_blocks, np.nan, dtype=float)
        den_block_sums = np.full(n_blocks, np.nan, dtype=float)
        for b, idx in enumerate(block_bins):
            if idx.size == 0:
                continue
            idx2 = idx[snp_mask[idx]]
            if idx2.size == 0:
                continue
            num_block_sums[b] = float(np.nansum(num[idx2]))
            den_block_sums[b] = float(idx2.size)

        res = _jackknife_ratio_from_block_sums(num_block_sums, den_block_sums)
        rows.append(
            {
                "target": t,
                "ref1": r1,
                "ref2": r2,
                "est": res.est,
                "se": res.se,
                "z": res.z,
                "p": res.p,
                "n_blocks": res.n_blocks,
                "n_snps": res.n_snps,
            }
        )

    return pd.DataFrame(rows)


def f4(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    a: Optional[Sequence[str]] = None,
    b: Optional[Sequence[str]] = None,
    c: Optional[Sequence[str]] = None,
    d: Optional[Sequence[str]] = None,
    sample_labels: Optional[Sequence[str]] = None,
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Compute f4-statistics f4(a, b; c, d) with block-jackknife SE.

    - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
    - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj)
    n_snps, _ = afs.shape
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    n_blocks = block_lengths.size

    if a is None and b is None and c is None and d is None:
        quads = [(w, x, y, z) for w in pops for x in pops for y in pops for z in pops]
    else:
        if a is None or b is None or c is None or d is None:
            raise ValueError("a, b, c, d must all be provided if any is provided")
        if not (len(a) == len(b) == len(c) == len(d)):
            raise ValueError("a, b, c, d must have the same length")
        quads = list(zip(a, b, c, d))

    name_to_idx = {p: i for i, p in enumerate(pops)}
    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]
    rows: List[Dict[str, Union[str, float, int]]] = []

    for pa, pb, pc, dpop in quads:
        ia = name_to_idx[pa]
        ib = name_to_idx[pb]
        ic = name_to_idx[pc]
        id_ = name_to_idx[dpop]

        A = afs[:, ia]
        B = afs[:, ib]
        C = afs[:, ic]
        D = afs[:, id_]
        na = counts[:, ia].astype(float)
        nb = counts[:, ib].astype(float)
        nc = counts[:, ic].astype(float)
        nd = counts[:, id_].astype(float)

        with np.errstate(invalid="ignore"):
            num = (A - B) * (C - D)
        snp_mask = np.isfinite(num) & (na > 0) & (nb > 0) & (nc > 0) & (nd > 0)

        num_block_sums = np.full(n_blocks, np.nan, dtype=float)
        den_block_sums = np.full(n_blocks, np.nan, dtype=float)
        for b_idx, idx in enumerate(block_bins):
            if idx.size == 0:
                continue
            idx2 = idx[snp_mask[idx]]
            if idx2.size == 0:
                continue
            num_block_sums[b_idx] = float(np.nansum(num[idx2]))
            den_block_sums[b_idx] = float(idx2.size)

        res = _jackknife_ratio_from_block_sums(num_block_sums, den_block_sums)
        rows.append(
            {
                "a": pa,
                "b": pb,
                "c": pc,
                "d": dpop,
                "est": res.est,
                "se": res.se,
                "z": res.z,
                "p": res.p,
                "n_blocks": res.n_blocks,
                "n_snps": res.n_snps,
            }
        )

    return pd.DataFrame(rows)


def d_stat(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    a: Optional[Sequence[str]] = None,
    b: Optional[Sequence[str]] = None,
    c: Optional[Sequence[str]] = None,
    d: Optional[Sequence[str]] = None,
    sample_labels: Optional[Sequence[str]] = None,
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Compute D-statistics D(a, b; c, d) as ratio of sums:
        D = sum_l (A-B)(C-D)  /  sum_l (A+B-2AB)(C+D-2CD)
    with delete-one block jackknife SE.

    - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
    - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj)
    n_snps, _ = afs.shape
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    n_blocks = block_lengths.size

    if a is None and b is None and c is None and d is None:
        quads = [(w, x, y, z) for w in pops for x in pops for y in pops for z in pops]
    else:
        if a is None or b is None or c is None or d is None:
            raise ValueError("a, b, c, d must all be provided if any is provided")
        if not (len(a) == len(b) == len(c) == len(d)):
            raise ValueError("a, b, c, d must have the same length")
        quads = list(zip(a, b, c, d))

    name_to_idx = {p: i for i, p in enumerate(pops)}
    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]
    rows: List[Dict[str, Union[str, float, int]]] = []

    for pa, pb, pc, dpop in quads:
        ia = name_to_idx[pa]
        ib = name_to_idx[pb]
        ic = name_to_idx[pc]
        id_ = name_to_idx[dpop]

        A = afs[:, ia]
        B = afs[:, ib]
        C = afs[:, ic]
        D = afs[:, id_]
        na = counts[:, ia].astype(float)
        nb = counts[:, ib].astype(float)
        nc = counts[:, ic].astype(float)
        nd = counts[:, id_].astype(float)

        with np.errstate(invalid="ignore"):
            num = (A - B) * (C - D)
            den = (A + B - 2 * A * B) * (C + D - 2 * C * D)

        snp_mask = np.isfinite(num) & np.isfinite(den) & (na > 0) & (nb > 0) & (nc > 0) & (nd > 0)

        num_block_sums = np.full(n_blocks, np.nan, dtype=float)
        den_block_sums = np.full(n_blocks, np.nan, dtype=float)
        ct_block_sums = np.zeros(n_blocks, dtype=int)
        for b_idx, idx in enumerate(block_bins):
            if idx.size == 0:
                continue
            idx2 = idx[snp_mask[idx]]
            if idx2.size == 0:
                continue
            num_block_sums[b_idx] = float(np.nansum(num[idx2]))
            # drop near-zero denominators within a block for stability
            block_den = float(np.nansum(den[idx2]))
            den_block_sums[b_idx] = block_den if abs(block_den) > 1e-12 else np.nan
            ct_block_sums[b_idx] = int(idx2.size)
            
        res = _jackknife_ratio_from_block_sums(num_block_sums, den_block_sums)
        rows.append(
            {
                "a": pa,
                "b": pb,
                "c": pc,
                "d": dpop,
                "est": res.est,
                "se": res.se,
                "z": res.z,
                "p": res.p,
                "n_blocks": res.n_blocks,
                "n_snps": int(ct_block_sums.sum()),
            }
        )

    return pd.DataFrame(rows)


def f4_ratio(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    num: Sequence[Tuple[str, str, str, str]],
    den: Sequence[Tuple[str, str, str, str]],
    sample_labels: Optional[Sequence[str]] = None,
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Compute f4-ratio statistics as ratio of two f4-statistics with block-jackknife SE.

    Args:
        num: Sequence of quadruples (a, b, c, d) for numerator f4(a, b; c, d)
        den: Sequence of quadruples (a, b, c, d) for denominator f4(a, b; c, d)

    Notes:
        - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
        - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
    """
    if len(num) != len(den):
        raise ValueError("'num' and 'den' must have the same length")

    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj)
    n_snps, _ = afs.shape
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    n_blocks = block_lengths.size
    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]
    name_to_idx = {p: i for i, p in enumerate(pops)}

    rows: List[Dict[str, Union[str, float, int]]] = []
    for (na, nb, nc, nd), (da, db, dc, dd) in zip(num, den):
        ia, ib, ic, id_ = name_to_idx[na], name_to_idx[nb], name_to_idx[nc], name_to_idx[nd]
        ja, jb, jc, jd = name_to_idx[da], name_to_idx[db], name_to_idx[dc], name_to_idx[dd]

        A, B, C, D = afs[:, ia], afs[:, ib], afs[:, ic], afs[:, id_]
        E, F, G, H = afs[:, ja], afs[:, jb], afs[:, jc], afs[:, jd]
        nA, nB, nC, nD = counts[:, ia], counts[:, ib], counts[:, ic], counts[:, id_]
        nE, nF, nG, nH = counts[:, ja], counts[:, jb], counts[:, jc], counts[:, jd]

        with np.errstate(invalid="ignore"):
            num_snp = (A - B) * (C - D)
            den_snp = (E - F) * (G - H)
        mask_num = np.isfinite(num_snp) & (nA > 0) & (nB > 0) & (nC > 0) & (nD > 0)
        mask_den = np.isfinite(den_snp) & (nE > 0) & (nF > 0) & (nG > 0) & (nH > 0)
        mask_both = mask_num & mask_den

        num_block_sums = np.full(n_blocks, np.nan, dtype=float)
        den_block_sums = np.full(n_blocks, np.nan, dtype=float)
        ct_block_sums = np.zeros(n_blocks, dtype=int)
        for b_idx, idx in enumerate(block_bins):
            if idx.size == 0:
                continue
            idx2 = idx[mask_both[idx]]
            if idx2.size == 0:
                continue
            num_block_sums[b_idx] = float(np.nansum(num_snp[idx2]))
            block_den = float(np.nansum(den_snp[idx2]))
            den_block_sums[b_idx] = block_den if abs(block_den) > 1e-12 else np.nan
            ct_block_sums[b_idx] = int(idx2.size)
            
        res = _jackknife_ratio_from_block_sums(num_block_sums, den_block_sums)
        rows.append(
            {
                "num": f"({na},{nb};{nc},{nd})",
                "den": f"({da},{db};{dc},{dd})",
                "est": res.est,
                "se": res.se,
                "z": res.z,
                "p": res.p,
                "n_blocks": res.n_blocks,
                "n_snps": int(ct_block_sums.sum()),
            }
        )

    return pd.DataFrame(rows)


def fst(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    pop1: Optional[Sequence[str]] = None,
    pop2: Optional[Sequence[str]] = None,
    *,
    method: str = "hudson",
    sample_labels: Optional[Sequence[str]] = None,
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
    include_self: bool = False,
) -> pd.DataFrame:
    """
    Pairwise F_ST with delete-one block jackknife SE.

    Methods:
      - 'hudson' (a.k.a. "ratio of averages" following Hudson 1992 / Bhatia 2013):
            per-SNP num = d_xy - 0.5*(pi_x + pi_y)
            per-SNP den = d_xy
        where d_xy = p_x*(1-p_y) + p_y*(1-p_x) and
              pi_x = 2*p_x*(1-p_x) * n_x/(n_x - 1) (unbiased within-pop diversity on haplotypes).
      - 'weir_cockerham' (Weir & Cockerham 1984; θ):
            compute per-SNP variance components a,b,c (for r=2) from allele freqs and haplotype counts:
                n1, n2 = haplotype counts; p1, p2 = allele freqs
                n  = n1 + n2
                n_bar = n / 2
                p_bar = (n1*p1 + n2*p2) / n
                s2    = (n1*(p1 - p_bar)**2 + n2*(p2 - p_bar)**2) / n_bar
                h1, h2 = 2*p1*(1-p1), 2*p2*(1-p2)
                h_bar = 0.5*(h1 + h2)
                n_c   = (n - (n1**2 + n2**2)/n)  # equals 2*n1*n2/n for r=2
                a = (n_bar / n_c) * (s2 - (p_bar*(1 - p_bar) - 0.5*s2 - 0.25*h_bar) / (n_bar - 1))
                b = (n_bar / (n_bar - 1)) * (p_bar*(1 - p_bar) - 0.5*s2 - ((2*n_bar - 1)/(4*n_bar))*h_bar)
                c = 0.5 * h_bar
            and use num=a, den=(a+b+c), then ratio-of-sums jackknife.

    Notes:
      * Inputs are the same as f2/f3/f4: either SNPObject or (afs, counts, pops).
      * For WC we use expected heterozygosity h_i = 2 p_i (1 - p_i) from allele freqs.
      * SNPs with n<=1 in either pop or with invalid denominators are ignored.
    """
    method = str(method).strip().lower().replace(" ", "_").replace("-", "_")
    if method in ("wc", "weir", "weir_cockerham", "weir-cockerham"):
        method = "weir_cockerham"
    elif method in ("h", "hudson", "bhatia", "ratio", "ratio_of_averages", "ratio-of-averages"):
        method = "hudson"
    elif method not in ("hudson", "weir_cockerham"):
        # only raise if it's neither a known alias nor a canonical name
        raise ValueError(f"Unknown method for fst: {method!r}")

    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj)
    n_snps, n_pops = afs.shape
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    n_blocks = block_lengths.size

    if pop1 is None and pop2 is None:
        if include_self:
            pairs = [(pops[i], pops[j]) for i in range(n_pops) for j in range(i, n_pops)]
        else:
            pairs = [(pops[i], pops[j]) for i in range(n_pops) for j in range(i + 1, n_pops)]
    else:
        if pop1 is None or pop2 is None or len(pop1) != len(pop2):
            raise ValueError("pop1 and pop2 must both be provided and of equal length")
        pairs = list(zip(pop1, pop2))

    name_to_idx = {p: i for i, p in enumerate(pops)}
    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]

    out_rows: List[Dict[str, Union[str, float, int]]] = []

    for pA, pB in pairs:
        i = name_to_idx[pA]
        j = name_to_idx[pB]
        p1 = afs[:, i].astype(float)
        p2 = afs[:, j].astype(float)
        n1 = counts[:, i].astype(float)
        n2 = counts[:, j].astype(float)

        valid = np.isfinite(p1) & np.isfinite(p2)

        if method == "hudson":
            # d_xy and within-pop diversities (unbiased, haplotype-based)
            d_xy = p1 * (1.0 - p2) + p2 * (1.0 - p1)
            with np.errstate(divide="ignore", invalid="ignore"):
                pi1 = 2.0 * p1 * (1.0 - p1) * (n1 / (n1 - 1.0))
                pi2 = 2.0 * p2 * (1.0 - p2) * (n2 / (n2 - 1.0))
            num_snp = d_xy - 0.5 * (pi1 + pi2)
            den_snp = d_xy
            snp_mask = valid & (n1 > 1) & (n2 > 1) & np.isfinite(num_snp) & np.isfinite(den_snp)
        else:
            # Weir-Cockerham θ components (r=2)
            n = n1 + n2
            with np.errstate(divide="ignore", invalid="ignore"):
                n_bar = n / 2.0
                p_bar = np.where(n > 0, (n1 * p1 + n2 * p2) / n, np.nan)
                s2 = (n1 * (p1 - p_bar) ** 2 + n2 * (p2 - p_bar) ** 2) / n_bar
                h1 = 2.0 * p1 * (1.0 - p1)
                h2 = 2.0 * p2 * (1.0 - p2)
                h_bar = 0.5 * (h1 + h2)
                n_c = n - (n1 * n1 + n2 * n2) / np.where(n > 0, n, np.nan)  # == 2*n1*n2/n
                # components
                a = (n_bar / n_c) * (s2 - (p_bar * (1.0 - p_bar) - 0.5 * s2 - 0.25 * h_bar) / (n_bar - 1.0))
                b = (n_bar / (n_bar - 1.0)) * (p_bar * (1.0 - p_bar) - 0.5 * s2 - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar)
                c = 0.5 * h_bar
                num_snp = a
                den_snp = a + b + c
            # Need at least 2 haplotypes per pop and well-defined denominators
            snp_mask = valid & (n1 > 1) & (n2 > 1) & np.isfinite(num_snp) & np.isfinite(den_snp)

        # Aggregate by blocks
        num_block_sums = np.full(n_blocks, np.nan, dtype=float)
        den_block_sums = np.full(n_blocks, np.nan, dtype=float)
        ct_block_sums = np.zeros(n_blocks, dtype=int)

        for b_idx, idx in enumerate(block_bins):
            if idx.size == 0:
                continue
            idx2 = idx[snp_mask[idx]]
            if idx2.size == 0:
                continue
            ns = float(np.nansum(num_snp[idx2]))
            ds = float(np.nansum(den_snp[idx2]))
            # drop numerically-zero denominators for stability
            den_block_sums[b_idx] = ds if abs(ds) > 1e-12 else np.nan
            num_block_sums[b_idx] = ns
            ct_block_sums[b_idx] = int(idx2.size)

        res = _jackknife_ratio_from_block_sums(num_block_sums, den_block_sums)
        out_rows.append(
            {
                "pop1": pA,
                "pop2": pB,
                "method": method,
                "est": res.est,
                "se": res.se,
                "z": res.z,
                "p": res.p,
                "n_blocks": res.n_blocks,
                "n_snps": int(ct_block_sums.sum()),
            }
        )

    return pd.DataFrame(out_rows)



__all__ = [
    "f2",
    "f3",
    "f4",
    "d_stat",
    "f4_ratio",
    "fst",
]


