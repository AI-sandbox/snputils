from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from snputils._utils.allele_freq import aggregate_pop_allele_freq


def genomic_block_labels(
    chrom: Sequence[Any],
    pos: Sequence[int],
    block_size_bp: int,
) -> np.ndarray:
    """
    Build genomic block labels from chromosome and base-pair position arrays.

    A new block starts when the chromosome changes or when the current variant is at least
    ``block_size_bp`` bases away from the start of the current block on that chromosome.
    """
    if block_size_bp <= 0:
        raise ValueError("block_size_bp must be positive.")
    chrom_arr = np.asarray(chrom)
    pos_arr = np.asarray(pos)
    if chrom_arr.shape[0] != pos_arr.shape[0]:
        raise ValueError("'chrom' and 'pos' must have the same length.")

    labels: List[str] = []
    current_chrom: Optional[str] = None
    block_start = -1
    block_index = -1
    for c, p in zip(chrom_arr, pos_arr):
        c = str(c)
        p = int(p)
        if current_chrom != c or p - block_start >= block_size_bp:
            current_chrom = c
            block_start = p
            block_index += 1
        labels.append(f"{c}:{block_index}")
    return np.asarray(labels, dtype=object)


def genomic_block_labels(
    chrom: Sequence[Any],
    pos: Sequence[int],
    block_size_bp: int,
) -> np.ndarray:
    """
    Build genomic block labels from chromosome and base-pair position arrays.

    A new block starts when the chromosome changes or when the current variant is at least
    ``block_size_bp`` bases away from the start of the current block on that chromosome.
    """
    if block_size_bp <= 0:
        raise ValueError("block_size_bp must be positive.")
    chrom_arr = np.asarray(chrom)
    pos_arr = np.asarray(pos)
    if chrom_arr.shape[0] != pos_arr.shape[0]:
        raise ValueError("'chrom' and 'pos' must have the same length.")

    labels: List[str] = []
    current_chrom: Optional[str] = None
    block_start = -1
    block_index = -1
    for c, p in zip(chrom_arr, pos_arr):
        c = str(c)
        p = int(p)
        if current_chrom != c or p - block_start >= block_size_bp:
            current_chrom = c
            block_start = p
            block_index += 1
        labels.append(f"{c}:{block_index}")
    return np.asarray(labels, dtype=object)


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


def _weighted_jackknife_from_block_estimates(
    block_estimates: np.ndarray,
    block_lengths: np.ndarray,
) -> BlockJackknifeResult:
    block_estimates = np.asarray(block_estimates, dtype=float)
    block_lengths = np.asarray(block_lengths, dtype=float)

    valid = (
        np.isfinite(block_estimates)
        & np.isfinite(block_lengths)
        & (block_lengths > 0)
    )
    if not np.any(valid):
        return BlockJackknifeResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)

    x = block_estimates[valid]
    w = block_lengths[valid]
    n_snps = int(np.sum(w))
    nb = int(w.size)

    if nb < 2:
        est = float(np.average(x, weights=w))
        return BlockJackknifeResult(est, float("nan"), float("nan"), float("nan"), nb, n_snps)

    weight_sum = float(np.sum(w))
    tot = float(np.average(x, weights=w))
    rel = w / weight_sum

    with np.errstate(divide="ignore", invalid="ignore"):
        loo = (tot - x * rel) / (1.0 - rel)

    finite = np.isfinite(loo)
    if not np.any(finite):
        return BlockJackknifeResult(tot, float("nan"), float("nan"), float("nan"), nb, n_snps)

    loo = loo[finite]
    w = w[finite]
    weight_sum = float(np.sum(w))
    nb = int(w.size)

    h = weight_sum / w
    est = float(np.average(loo, weights=(1.0 - 1.0 / h)))
    se = float(np.sqrt(np.mean((est - loo) ** 2 * (h - 1.0))))
    z = float(est / se) if se > 0 else float("nan")
    p = float(math.erfc(abs(z) / math.sqrt(2))) if np.isfinite(z) else float("nan")

    return BlockJackknifeResult(est, se, z, p, nb, int(weight_sum))



def _jackknife_block_ratio_estimates(
    num_block_sums: np.ndarray,
    den_block_sums: np.ndarray,
    block_lengths: np.ndarray,
    *,
    min_abs_den: float = 1e-12,
) -> BlockJackknifeResult:
    den_block_sums = np.asarray(den_block_sums, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        block_estimates = np.asarray(num_block_sums, dtype=float) / den_block_sums
    block_estimates = np.where(np.abs(den_block_sums) > min_abs_den, block_estimates, np.nan)
    return _weighted_jackknife_from_block_estimates(block_estimates, block_lengths)

def _weighted_jackknife_ratio_from_block_sums(
    num_block_sums: np.ndarray,
    den_block_sums: np.ndarray,
    block_lengths: np.ndarray,
    *,
    denominator_est_threshold: float = 1e-6,
) -> BlockJackknifeResult:
    """
    Compute a weighted delete-one block jackknife for a ratio of block means.
    """
    num_block_sums = np.asarray(num_block_sums, dtype=float)
    den_block_sums = np.asarray(den_block_sums, dtype=float)
    block_lengths = np.asarray(block_lengths, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        num_blocks = num_block_sums / block_lengths
        den_blocks = den_block_sums / block_lengths

    valid = (
        np.isfinite(num_blocks)
        & np.isfinite(den_blocks)
        & np.isfinite(block_lengths)
        & (block_lengths > 0)
        & (np.abs(den_blocks) >= denominator_est_threshold)
    )
    if not np.any(valid):
        return BlockJackknifeResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)

    num_b = num_blocks[valid]
    den_b = den_blocks[valid]
    weights = block_lengths[valid]
    n_snps = int(np.sum(weights))
    nb = int(weights.size)

    weight_sum = float(np.sum(weights))
    tot_num = float(np.sum(num_b * weights) / weight_sum)
    tot_den = float(np.sum(den_b * weights) / weight_sum)
    if tot_den == 0:
        return BlockJackknifeResult(float("nan"), float("nan"), float("nan"), float("nan"), nb, n_snps)

    tot = tot_num / tot_den
    rel = weights / weight_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        loo_num = (tot_num - num_b * rel) / (1.0 - rel)
        loo_den = (tot_den - den_b * rel) / (1.0 - rel)
        loo_ratio = loo_num / loo_den

    finite = np.isfinite(loo_ratio)
    if not np.any(finite):
        return BlockJackknifeResult(tot, 0.0, float("nan"), float("nan"), nb, n_snps)

    loo_ratio = loo_ratio[finite]
    weights = weights[finite]
    nb2 = int(weights.size)
    weight_sum = float(np.sum(weights))
    est = float(np.mean(tot - loo_ratio) * nb2 + np.sum(loo_ratio * weights) / weight_sum)
    h = weight_sum / weights
    tau = h * tot - (h - 1.0) * loo_ratio
    se = float(np.sqrt(np.mean((tau - est) ** 2 / (h - 1.0))))
    z = float(est / se) if se > 0 else float("nan")
    p = float(math.erfc(abs(z) / math.sqrt(2))) if np.isfinite(z) else float("nan")
    return BlockJackknifeResult(est, se, z, p, nb2, n_snps)


def _aggregate_to_pop_allele_freq(
    genotypes: np.ndarray,
    sample_labels: Sequence[str],
    *,
    ancestry: Optional[Union[str, int]] = None,
    snpobj: Optional[Any] = None,
    laiobj: Optional[Any] = None,
    pseudohaploid: Union[bool, int] = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert sample-level genotypes into per-population allele frequency and count matrices.
    """
    calldata_lai = None
    if ancestry is not None:
        if snpobj is not None and getattr(snpobj, "calldata_lai", None) is not None:
            calldata_lai = snpobj.calldata_lai
        elif laiobj is not None:
            try:
                converted_lai = laiobj.convert_to_snp_level(snpobject=snpobj, lai_format="3D")
                calldata_lai = getattr(converted_lai, "calldata_lai", None)
            except Exception:
                calldata_lai = None

        if calldata_lai is None:
            raise ValueError(
                "Ancestry-specific masking requires SNP-level LAI "
                "(provide a LocalAncestryObject via 'laiobj' or ensure 'snpobj.calldata_lai' is set)."
            )

    afs, counts, pops = aggregate_pop_allele_freq(
        genotypes=genotypes,
        sample_labels=sample_labels,
        ancestry=ancestry,
        calldata_lai=calldata_lai,
        pseudohaploid=pseudohaploid,
    )
    return afs, counts, pops


def _default_sample_labels_from_snpobj(snpobj: Any) -> List[str]:
    """
    Default per-sample group labels for SNPObject inputs.

    If ``sample_fid`` is set on the object and differs from ``samples`` for at least
    one individual (PLINK FID vs IID), use ``sample_fid``. Otherwise use ``samples`` (IID) as the label
    for each individual.
    """
    if snpobj.samples is None:
        raise ValueError("sample_labels must be provided when SNPObject.samples is None")
    samples = np.asarray(snpobj.samples, dtype=object)
    fid = getattr(snpobj, "sample_fid", None)
    if fid is not None:
        fid = np.asarray(fid, dtype=object)
        if fid.shape[0] != samples.shape[0]:
            raise ValueError("'sample_fid' must have the same length as 'samples'")
        if np.any(fid.astype(str) != samples.astype(str)):
            return [str(x) for x in fid]
    return [str(x) for x in samples]


def _prepare_inputs(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    sample_labels: Optional[Sequence[str]] = None,
    *,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
    pseudohaploid: Union[bool, int] = False,
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
            sample_labels = _default_sample_labels_from_snpobj(snpobj)
        afs, counts, pops = _aggregate_to_pop_allele_freq(
            snpobj.genotypes,
            sample_labels,
            ancestry=ancestry,
            snpobj=snpobj,
            laiobj=laiobj,
            pseudohaploid=pseudohaploid,
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


def _block_ids_are_contiguous(block_ids: np.ndarray, block_lengths: np.ndarray) -> bool:
    if block_ids.size == 0:
        return True
    if int(block_lengths.sum()) != block_ids.size:
        return False
    if np.any(np.diff(block_ids) < 0):
        return False
    counts = np.bincount(block_ids, minlength=block_lengths.size)
    return bool(np.array_equal(counts[: block_lengths.size], block_lengths))


def _complete_block_product_sums(
    afs: np.ndarray,
    counts: np.ndarray,
    block_ids: np.ndarray,
    block_lengths: np.ndarray,
    *,
    require_counts_gt: int,
    need_correction: bool,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
    """
    Precompute block sums of p_i * p_j for complete-data f-stat calculations.

    This fast path is exact for cases where every requested SNP has valid allele
    frequencies and sufficient haplotype counts for all populations. Missing or
    low-count data falls back to the per-statistic masking logic.
    """
    afs = np.asarray(afs, dtype=float)
    counts = np.asarray(counts)
    if afs.ndim != 2 or counts.shape != afs.shape:
        return None
    if not np.isfinite(afs).all() or not np.all(counts > require_counts_gt):
        return None

    n_snps, n_pops = afs.shape
    n_blocks = block_lengths.size
    den_block_sums = block_lengths.astype(float, copy=False)
    pair_sums = np.empty((n_blocks, n_pops, n_pops), dtype=float)
    corr_sums = np.empty((n_blocks, n_pops), dtype=float) if need_correction else None

    if _block_ids_are_contiguous(block_ids, block_lengths):
        starts = np.r_[0, np.cumsum(block_lengths[:-1])]
        for b_idx, start in enumerate(starts):
            end = int(start + block_lengths[b_idx])
            block_afs = afs[int(start):end]
            pair_sums[b_idx] = block_afs.T @ block_afs
            if need_correction and corr_sums is not None:
                block_counts = counts[int(start):end].astype(float, copy=False)
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr_sums[b_idx] = np.sum(block_afs * (1.0 - block_afs) / (block_counts - 1.0), axis=0)
    else:
        for i in range(n_pops):
            ai = afs[:, i]
            for j in range(n_pops):
                pair_sums[:, i, j] = np.bincount(
                    block_ids,
                    weights=ai * afs[:, j],
                    minlength=n_blocks,
                )
            if need_correction and corr_sums is not None:
                ci = counts[:, i].astype(float, copy=False)
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = ai * (1.0 - ai) / (ci - 1.0)
                corr_sums[:, i] = np.bincount(block_ids, weights=corr, minlength=n_blocks)

    return pair_sums, corr_sums, den_block_sums


def _complete_block_hudson_fst_sums(
    afs: np.ndarray,
    counts: np.ndarray,
    block_ids: np.ndarray,
    block_lengths: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Precompute complete-data block sums needed by Hudson F_ST.
    """
    afs = np.asarray(afs, dtype=float)
    counts = np.asarray(counts)
    if afs.ndim != 2 or counts.shape != afs.shape:
        return None
    if not np.isfinite(afs).all() or not np.all(counts > 1):
        return None

    n_snps, n_pops = afs.shape
    n_blocks = block_lengths.size
    af_sums = np.empty((n_blocks, n_pops), dtype=float)
    pair_sums = np.empty((n_blocks, n_pops, n_pops), dtype=float)
    pi_sums = np.empty((n_blocks, n_pops), dtype=float)

    if _block_ids_are_contiguous(block_ids, block_lengths):
        starts = np.r_[0, np.cumsum(block_lengths[:-1])]
        for b_idx, start in enumerate(starts):
            end = int(start + block_lengths[b_idx])
            block_afs = afs[int(start):end]
            block_counts = counts[int(start):end].astype(float, copy=False)
            af_sums[b_idx] = np.sum(block_afs, axis=0)
            pair_sums[b_idx] = block_afs.T @ block_afs
            with np.errstate(divide="ignore", invalid="ignore"):
                pi_sums[b_idx] = np.sum(
                    2.0 * block_afs * (1.0 - block_afs) * (block_counts / (block_counts - 1.0)),
                    axis=0,
                )
    else:
        for i in range(n_pops):
            ai = afs[:, i]
            af_sums[:, i] = np.bincount(block_ids, weights=ai, minlength=n_blocks)
            with np.errstate(divide="ignore", invalid="ignore"):
                pi = 2.0 * ai * (1.0 - ai) * (counts[:, i].astype(float, copy=False) / (counts[:, i] - 1.0))
            pi_sums[:, i] = np.bincount(block_ids, weights=pi, minlength=n_blocks)
            for j in range(n_pops):
                pair_sums[:, i, j] = np.bincount(
                    block_ids,
                    weights=ai * afs[:, j],
                    minlength=n_blocks,
                )

    return af_sums, pair_sums, pi_sums


def _complete_block_pair_het_product_sums(
    afs: np.ndarray,
    counts: np.ndarray,
    block_ids: np.ndarray,
    block_lengths: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
) -> Optional[Tuple[np.ndarray, Dict[Tuple[int, int], int]]]:
    """
    Precompute block sums of h_ij * h_kl, where h_ij = p_i + p_j - 2 p_i p_j.
    """
    afs = np.asarray(afs, dtype=float)
    counts = np.asarray(counts)
    if afs.ndim != 2 or counts.shape != afs.shape:
        return None
    if not np.isfinite(afs).all() or not np.all(counts > 0):
        return None

    norm_pairs = sorted({(min(i, j), max(i, j)) for i, j in pairs})
    if not norm_pairs:
        return None

    pair_to_idx = {pair: idx for idx, pair in enumerate(norm_pairs)}
    n_blocks = block_lengths.size
    n_pairs = len(norm_pairs)
    out = np.empty((n_blocks, n_pairs, n_pairs), dtype=float)

    if _block_ids_are_contiguous(block_ids, block_lengths):
        starts = np.r_[0, np.cumsum(block_lengths[:-1])]
        for b_idx, start in enumerate(starts):
            end = int(start + block_lengths[b_idx])
            block_afs = afs[int(start):end]
            h = np.empty((block_afs.shape[0], n_pairs), dtype=float)
            for pair_idx, (i, j) in enumerate(norm_pairs):
                pi = block_afs[:, i]
                pj = block_afs[:, j]
                h[:, pair_idx] = pi + pj - 2.0 * pi * pj
            out[b_idx] = h.T @ h
    else:
        block_ids = np.asarray(block_ids, dtype=np.int64)
        for b_idx in range(n_blocks):
            block_afs = afs[block_ids == b_idx]
            h = np.empty((block_afs.shape[0], n_pairs), dtype=float)
            for pair_idx, (i, j) in enumerate(norm_pairs):
                pi = block_afs[:, i]
                pj = block_afs[:, j]
                h[:, pair_idx] = pi + pj - 2.0 * pi * pj
            out[b_idx] = h.T @ h

    return out, pair_to_idx

def _tsallis_entropy_bernoulli(p: np.ndarray, q: float) -> np.ndarray:
    """
    Tsallis q-entropy for Bernoulli(p).

    For q -> 1, returns Shannon entropy:
        -p log p - (1-p) log(1-p)

    For q != 1:
        S_q(p) = (1 - (p^q + (1-p)^q)) / (q - 1)
    """
    q = float(q)
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be a finite positive number")

    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)

    if np.isclose(q, 1.0):
        out = np.zeros_like(p, dtype=float)
        mask = (p > 0.0) & (p < 1.0)
        out[mask] = -(
            p[mask] * np.log(p[mask])
            + (1.0 - p[mask]) * np.log1p(-p[mask])
        )
        return out

    return (1.0 - (np.power(p, q) + np.power(1.0 - p, q))) / (q - 1.0)

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
    pseudohaploid: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Compute f2-statistics with block-jackknife standard errors.

    Args:
        data: Either a SNPObject or a tuple (afs, counts, pops), where `afs` and `counts` are arrays of
              shape (n_snps, n_pops). If a SNPObject is provided, `sample_labels` are used to aggregate samples to populations.
        pop1, pop2: Populations to compute f2 for. If both are None, `include_self=False`
            computes only off-diagonal pairs i<j; `include_self=True` computes all pairs
            including diagonals i<=j.
        sample_labels: Population label per sample (aligned with SNPObject.samples) when `data` is a SNPObject.
            If omitted, labels are inferred from ``SNPObject.sample_fid`` when it differs from ``samples`` (PLINK FID);
            otherwise each ``samples`` value is used as its own group label.
        apply_correction: Apply small-sample correction p*(1-p)/(n-1) per population.
            When True, SNPs with n<=1 in either population are excluded at that SNP.
        block_size: Number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
        blocks: Optional explicit block id per SNP. If provided, overrides `block_size`.
        ancestry: Optional ancestry code to mask genotypes to a specific ancestry before aggregation. Requires LAI.
        laiobj: Optional `LocalAncestryObject` used to derive SNP-level LAI if it is not already present on the SNPObject.
        pseudohaploid: If True, detects pseudo-haploid samples (samples with no heterozygotes in the first 1000 SNPs)
                       and treats them as haploid. If an integer `n` is provided, checks the first `n` SNPs.
                       If False, treats all samples as diploid.

    Returns:
        Pandas DataFrame with columns: pop1, pop2, est, se, z, p, n_blocks, n_snps
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj, pseudohaploid=pseudohaploid)
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
    fast = _complete_block_product_sums(
        afs,
        counts,
        block_ids,
        block_lengths,
        require_counts_gt=1 if apply_correction else 0,
        need_correction=apply_correction,
    )
    if fast is not None:
        pair_sums, corr_sums, den_block_sums = fast
        for p1, p2 in pop_pairs:
            i = name_to_idx[p1]
            j = name_to_idx[p2]
            num_block_sums = pair_sums[:, i, i] + pair_sums[:, j, j] - 2.0 * pair_sums[:, i, j]
            if apply_correction:
                assert corr_sums is not None
                num_block_sums = num_block_sums - corr_sums[:, i] - corr_sums[:, j]
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
    pseudohaploid: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Compute f3-statistics f3(target; ref1, ref2) with block-jackknife SE.

    - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
    - If `target`, `ref1`, and `ref2` are all None, compute f3 for all combinations where each role can be any population.
    - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
    - If `apply_correction` is True, subtract the finite sample term p_t*(1-p_t)/(n_t-1) from the per-SNP product.
        When True, SNPs with n_t<=1 are excluded.
    - If `sample_labels` is omitted for a SNPObject, defaults match ``f2`` (PLINK ``sample_fid`` when FID differs from IID).
    - `pseudohaploid`: If True, detects and treats pseudo-haploid samples as haploid. If int `n`, checks first `n` SNPs. If False, treats all as diploid.
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj, pseudohaploid=pseudohaploid)
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
    rows: List[Dict[str, Union[str, float, int]]] = []

    fast = _complete_block_product_sums(
        afs,
        counts,
        block_ids,
        block_lengths,
        require_counts_gt=1 if apply_correction else 0,
        need_correction=apply_correction,
    )
    if fast is not None:
        pair_sums, corr_sums, den_block_sums = fast
        for t, r1, r2 in triples:
            it = name_to_idx[t]
            i1 = name_to_idx[r1]
            i2 = name_to_idx[r2]
            num_block_sums = (
                pair_sums[:, it, it]
                - pair_sums[:, it, i1]
                - pair_sums[:, it, i2]
                + pair_sums[:, i1, i2]
            )
            if apply_correction:
                assert corr_sums is not None
                num_block_sums = num_block_sums - corr_sums[:, it]
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

    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]

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
    pseudohaploid: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Compute f4-statistics f4(a, b; c, d) with block-jackknife SE.

    - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
    - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
    - `pseudohaploid`: If True, detects and treats pseudo-haploid samples as haploid. If int `n`, checks first `n` SNPs. If False, treats all as diploid.
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj, pseudohaploid=pseudohaploid)
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
    rows: List[Dict[str, Union[str, float, int]]] = []

    fast = _complete_block_product_sums(
        afs,
        counts,
        block_ids,
        block_lengths,
        require_counts_gt=0,
        need_correction=False,
    )
    if fast is not None:
        pair_sums, _, den_block_sums = fast
        for pa, pb, pc, dpop in quads:
            ia = name_to_idx[pa]
            ib = name_to_idx[pb]
            ic = name_to_idx[pc]
            id_ = name_to_idx[dpop]
            num_block_sums = (
                pair_sums[:, ia, ic]
                - pair_sums[:, ia, id_]
                - pair_sums[:, ib, ic]
                + pair_sums[:, ib, id_]
            )
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

    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]

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
    pseudohaploid: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Compute D-statistics D(a, b; c, d) as ratio of sums:

    `D = sum_l (A-B)(C-D) / sum_l (A+B-2AB)(C+D-2CD)`

    with delete-one block jackknife SE.

    - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
    - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
    - `pseudohaploid`: If True, detects and treats pseudo-haploid samples as haploid. If int `n`, checks first `n` SNPs. If False, treats all as diploid.
    """
    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj, pseudohaploid=pseudohaploid)
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
    bids = np.asarray(block_ids, dtype=np.int64)
    rows: List[Dict[str, Union[str, float, int]]] = []
    indexed_quads = [
        (pa, pb, pc, dpop, name_to_idx[pa], name_to_idx[pb], name_to_idx[pc], name_to_idx[dpop])
        for pa, pb, pc, dpop in quads
    ]

    fast_products = _complete_block_product_sums(
        afs,
        counts,
        block_ids,
        block_lengths,
        require_counts_gt=0,
        need_correction=False,
    )
    den_pairs = [(ia, ib) for _, _, _, _, ia, ib, _, _ in indexed_quads]
    den_pairs.extend((ic, id_) for _, _, _, _, _, _, ic, id_ in indexed_quads)
    fast_den = _complete_block_pair_het_product_sums(afs, counts, block_ids, block_lengths, den_pairs)
    if fast_products is not None and fast_den is not None:
        pair_sums, _, _ = fast_products
        hetprod_sums, pair_to_idx = fast_den
        n_snps_used = int(block_lengths.sum())
        for pa, pb, pc, dpop, ia, ib, ic, id_ in indexed_quads:
            num_block_sums = (
                pair_sums[:, ia, ic]
                - pair_sums[:, ia, id_]
                - pair_sums[:, ib, ic]
                + pair_sums[:, ib, id_]
            )
            p_ab = pair_to_idx[(min(ia, ib), max(ia, ib))]
            p_cd = pair_to_idx[(min(ic, id_), max(ic, id_))]
            den_block_sums = hetprod_sums[:, p_ab, p_cd].copy()
            den_block_sums[np.isfinite(den_block_sums) & (np.abs(den_block_sums) <= 1e-12)] = np.nan
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
                    "n_snps": n_snps_used,
                }
            )
        return pd.DataFrame(rows)

    for pa, pb, pc, dpop, ia, ib, ic, id_ in indexed_quads:
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

        num_sel = np.where(snp_mask, num.astype(np.float64, copy=False), 0.0)
        den_sel = np.where(snp_mask, den.astype(np.float64, copy=False), 0.0)
        num_block_sums = np.bincount(bids, weights=num_sel, minlength=n_blocks).astype(np.float64, copy=False)
        den_raw = np.bincount(bids, weights=den_sel, minlength=n_blocks).astype(np.float64, copy=False)
        ct_block_sums = np.bincount(bids, weights=snp_mask.astype(np.int64), minlength=n_blocks)

        empty = ct_block_sums == 0
        num_block_sums[empty] = np.nan
        den_raw[empty] = np.nan
        den_raw[np.isfinite(den_raw) & (np.abs(den_raw) <= 1e-12)] = np.nan
        den_block_sums = den_raw
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
    pseudohaploid: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Compute f4-ratio statistics as ratio of two f4-statistics with block-jackknife SE.

    Args:
        num: Sequence of quadruples (a, b, c, d) for numerator f4(a, b; c, d)
        den: Sequence of quadruples (a, b, c, d) for denominator f4(a, b; c, d)

    Notes:
        - `block_size` is the number of SNPs per jackknife block (default 5000 SNPs). Ignored if `blocks` is provided.
        - If `ancestry` is provided, genotypes will be masked to the specified ancestry using LAI before aggregation.
        - `pseudohaploid`: If True, detects and treats pseudo-haploid samples as haploid. If int `n`, checks first `n` SNPs. If False, treats all as diploid.
    """
    if len(num) != len(den):
        raise ValueError("'num' and 'den' must have the same length")

    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj, pseudohaploid=pseudohaploid)
    n_snps, _ = afs.shape
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    n_blocks = block_lengths.size
    name_to_idx = {p: i for i, p in enumerate(pops)}

    rows: List[Dict[str, Union[str, float, int]]] = []
    fast = _complete_block_product_sums(
        afs,
        counts,
        block_ids,
        block_lengths,
        require_counts_gt=0,
        need_correction=False,
    )
    if fast is not None:
        pair_sums, _, _ = fast
        for (na, nb, nc, nd), (da, db, dc, dd) in zip(num, den):
            ia, ib, ic, id_ = name_to_idx[na], name_to_idx[nb], name_to_idx[nc], name_to_idx[nd]
            ja, jb, jc, jd = name_to_idx[da], name_to_idx[db], name_to_idx[dc], name_to_idx[dd]

            num_block_sums = (
                pair_sums[:, ia, ic]
                - pair_sums[:, ia, id_]
                - pair_sums[:, ib, ic]
                + pair_sums[:, ib, id_]
            )
            den_block_sums = (
                pair_sums[:, ja, jc]
                - pair_sums[:, ja, jd]
                - pair_sums[:, jb, jc]
                + pair_sums[:, jb, jd]
            )
            res = _weighted_jackknife_ratio_from_block_sums(
                num_block_sums,
                den_block_sums,
                block_lengths,
            )
            rows.append(
                {
                    "num": f"({na},{nb};{nc},{nd})",
                    "den": f"({da},{db};{dc},{dd})",
                    "est": res.est,
                    "se": res.se,
                    "z": res.z,
                    "p": res.p,
                    "n_blocks": res.n_blocks,
                    "n_snps": res.n_snps,
                }
            )
        return pd.DataFrame(rows)

    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]
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
            den_block_sums[b_idx] = float(np.nansum(den_snp[idx2]))
            ct_block_sums[b_idx] = int(idx2.size)
            
        res = _weighted_jackknife_ratio_from_block_sums(
            num_block_sums,
            den_block_sums,
            ct_block_sums,
        )
        rows.append(
            {
                "num": f"({na},{nb};{nc},{nd})",
                "den": f"({da},{db};{dc},{dd})",
                "est": res.est,
                "se": res.se,
                "z": res.z,
                "p": res.p,
                "n_blocks": res.n_blocks,
                "n_snps": res.n_snps,
            }
        )

    return pd.DataFrame(rows)


def fst(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    pop1: Optional[Sequence[str]] = None,
    pop2: Optional[Sequence[str]] = None,
    *,
    method: str = "hudson",
    q: float = 2.0,
    tsallis_weights: str = "equal",
    sample_labels: Optional[Sequence[str]] = None,
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
    include_self: bool = False,
    pseudohaploid: Union[bool, int] = False,
) -> pd.DataFrame:
    """
    Pairwise F_ST with delete-one block jackknife SE.

    Methods:
        `hudson`:
            Ratio-of-averages following Hudson 1992 / Bhatia 2013. Uses
            `num = d_xy - 0.5*(pi_x + pi_y)` and `den = d_xy`, where
            `d_xy = p_x*(1-p_y) + p_y*(1-p_x)` and `pi_x = 2*p_x*(1-p_x)*n_x/(n_x-1)`.
        `weir_cockerham`:
            Weir and Cockerham's theta for two populations. Computes per-SNP
            variance components `a`, `b`, and `c`, then uses a ratio-of-sums
            jackknife with `num = a` and `den = a + b + c`.
        `tsallis`:
            Tsallis q-entropy F-statistic. For two populations, computes
            per-SNP total entropy S_q(Bern(p_bar)) and within entropy
            w1*S_q(Bern(p1)) + w2*S_q(Bern(p2)), then returns the
            genome-wide micro-average:
                sum_l [S_total(l) - S_within(l)] / sum_l S_total(l)

            With q=2, this equals the classical heterozygosity-based F_ST:
                (H_T - H_S) / H_T

            With q=1, this is the Shannon entropy / normalized mutual
            information analogue.

            `tsallis_weights="equal"` uses w1=w2=0.5, for OVR equal-group weighting.
            `tsallis_weights="sample_size"` uses per-SNP haplotype count weights.

    Notes:
      * Inputs are the same as f2/f3/f4: either SNPObject or (afs, counts, pops).
      * For WC we use expected heterozygosity h_i = 2 p_i (1 - p_i) from allele freqs.
      * SNPs with n<=1 in either pop or with invalid denominators are ignored.
      * `pseudohaploid`: If True, detects and treats pseudo-haploid samples as haploid. If int `n`, checks first `n` SNPs. If False, treats all as diploid.
    """
    method = str(method).strip().lower().replace("-", "_")
    if method not in {"hudson", "weir_cockerham", "tsallis"}:
        raise ValueError("method must be 'hudson', 'weir_cockerham', or 'tsallis'")

    afs, counts, pops = _prepare_inputs(data, sample_labels, ancestry=ancestry, laiobj=laiobj, pseudohaploid=pseudohaploid)
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
    out_rows: List[Dict[str, Union[str, float, int]]] = []

    if method == "hudson":
        fast = _complete_block_hudson_fst_sums(afs, counts, block_ids, block_lengths)
        if fast is not None:
            af_sums, pair_sums, pi_sums = fast
            for pA, pB in pairs:
                i = name_to_idx[pA]
                j = name_to_idx[pB]
                den_block_sums = af_sums[:, i] + af_sums[:, j] - 2.0 * pair_sums[:, i, j]
                num_block_sums = den_block_sums - 0.5 * (pi_sums[:, i] + pi_sums[:, j])
                res = _jackknife_block_ratio_estimates(
                    num_block_sums,
                    den_block_sums,
                    block_lengths,
                )
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
                        "n_snps": n_snps,
                    }
                )
            return pd.DataFrame(out_rows)

    if method == "tsallis":
        q = float(q)
        if not np.isfinite(q) or q <= 0:
            raise ValueError("q must be a finite positive number")
        tsallis_weights = str(tsallis_weights).strip().lower().replace("-", "_")
        if tsallis_weights not in {"equal", "sample_size"}:
            raise ValueError("tsallis_weights must be 'equal' or 'sample_size'")

    block_bins = [np.where(block_ids == b)[0] for b in range(n_blocks)]

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
        elif method == "weir_cockerham":
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
        elif method == "tsallis":
            # Tsallis q-entropy F-statistic.
            #
            # S_total(l)  = S_q(Bern(p_bar_l))
            # S_within(l) = w1*S_q(Bern(p1_l)) + w2*S_q(Bern(p2_l))
            # F_q         = sum_l [S_total(l) - S_within(l)] / sum_l S_total(l)
            #
            # q=2 recovers heterozygosity-based F_ST.
            if tsallis_weights == "equal":
                w1 = np.full_like(p1, 0.5, dtype=float)
                w2 = np.full_like(p2, 0.5, dtype=float)
                count_mask = (n1 > 0) & (n2 > 0)
            else:
                n = n1 + n2
                with np.errstate(divide="ignore", invalid="ignore"):
                    w1 = np.where(n > 0, n1 / n, np.nan)
                    w2 = np.where(n > 0, n2 / n, np.nan)
                count_mask = (n1 > 0) & (n2 > 0) & (n > 0)

            with np.errstate(divide="ignore", invalid="ignore"):
                p_bar = w1 * p1 + w2 * p2

                s_total = _tsallis_entropy_bernoulli(p_bar, q)
                s_within = (
                    w1 * _tsallis_entropy_bernoulli(p1, q)
                    + w2 * _tsallis_entropy_bernoulli(p2, q)
                )

                num_snp = s_total - s_within
                den_snp = s_total

            snp_mask = (
                valid
                & count_mask
                & np.isfinite(num_snp)
                & np.isfinite(den_snp)
                & (den_snp > 1e-12)
            )

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

        if method == "hudson":
            res = _jackknife_block_ratio_estimates(num_block_sums, den_block_sums, ct_block_sums)
        else:
            res = _jackknife_ratio_from_block_sums(num_block_sums, den_block_sums)
        
        row = {
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
        if method == "tsallis":
            row["q"] = q
            row["tsallis_weights"] = tsallis_weights
        
        out_rows.append(row)

    return pd.DataFrame(out_rows)



__all__ = [
    "f2",
    "f3",
    "f4",
    "d_stat",
    "f4_ratio",
    "fst",
]
