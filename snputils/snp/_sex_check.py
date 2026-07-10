"""Reference-free genetic sex checking from chromosome-X zygosity distributions.

The inference model is the degree-6 distilled polynomial published with Zigo:
Molina-Sedano, Mas Montserrat, and Ioannidis (2026),
https://doi.org/10.64898/2026.03.15.711924.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import resources
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

from snputils.snp.genobj import SNPObject


_DEFAULT_X_CHROMOSOMES = ("X", "chrX", "23")
_POLY_TOKEN_RE = re.compile(r"x(\d+)(?:\^(\d+))?")


def _normalize_chromosome(value: Any) -> str:
    text = str(value).strip().lower()
    for prefix in ("chrom", "chr", "chm"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text


def _x_variant_mask(
    snpobj: SNPObject,
    n_variants: int,
    *,
    x_chromosomes: Union[str, Sequence[Union[str, int]]],
    assume_x: bool,
) -> np.ndarray:
    if assume_x:
        return np.ones(n_variants, dtype=bool)

    chromosomes = snpobj.variants_chrom
    if chromosomes is None or (n_variants > 0 and np.asarray(chromosomes).size == 0):
        raise ValueError(
            "Sex checking requires chromosome metadata to select chromosome X. "
            "Pass `assume_x=True` only when every variant in the SNPObject is from chromosome X."
        )

    chromosomes = np.asarray(chromosomes)
    if chromosomes.ndim != 1 or chromosomes.shape[0] != n_variants:
        raise ValueError(
            "`variants_chrom` must be one-dimensional and aligned with the genotype variant axis."
        )

    requested = [x_chromosomes] if isinstance(x_chromosomes, str) else list(x_chromosomes)
    normalized_x = {_normalize_chromosome(value) for value in requested}
    mask = np.fromiter(
        (_normalize_chromosome(value) in normalized_x for value in chromosomes),
        dtype=bool,
        count=n_variants,
    )
    if not np.any(mask):
        available = list(dict.fromkeys(str(value) for value in chromosomes))
        preview = available[:10]
        suffix = "..." if len(available) > len(preview) else ""
        raise ValueError(
            "No chromosome-X variants were found. "
            f"Observed chromosome labels: {preview}{suffix}."
        )
    return mask


def _zygosity_distribution(genotypes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized Zigo features and callable genotype counts per sample."""
    gt = np.asarray(genotypes)
    if gt.ndim not in (2, 3):
        raise ValueError("Sex checking requires a 2D dosage or 3D allele-call genotype array.")
    if gt.ndim == 3 and gt.shape[2] != 2:
        raise ValueError("Three-dimensional genotype arrays must have a final allele axis of length 2.")

    try:
        finite = np.isfinite(gt)
    except TypeError as exc:
        raise ValueError("Sex checking requires numeric hard-call genotypes.") from exc

    n_samples = gt.shape[1]
    counts = np.zeros((n_samples, 3), dtype=np.float64)

    if gt.ndim == 2:
        valid = finite & ((gt == 0) | (gt == 1) | (gt == 2))
        for genotype_class in range(3):
            counts[:, genotype_class] = np.sum(
                valid & (gt == genotype_class), axis=0, dtype=np.int64
            )
    else:
        valid_alleles = finite & ((gt == 0) | (gt == 1))
        invalid_nonmissing = finite & (gt >= 0) & ~valid_alleles
        first_valid = valid_alleles[:, :, 0]
        second_valid = valid_alleles[:, :, 1]
        both_called = first_valid & second_valid
        one_called = (first_valid ^ second_valid) & ~np.any(invalid_nonmissing, axis=2)

        dosage = np.where(valid_alleles, gt, 0).sum(axis=2)
        counts[:, 0] = np.sum(
            (both_called & (dosage == 0)) | (one_called & (dosage == 0)),
            axis=0,
            dtype=np.int64,
        )
        counts[:, 1] = np.sum(
            (both_called & (dosage == 1)) | (one_called & (dosage == 1)),
            axis=0,
            dtype=np.int64,
        )
        counts[:, 2] = np.sum(
            both_called & (dosage == 2), axis=0, dtype=np.int64
        )

    n_called = counts.sum(axis=1).astype(np.int64, copy=False)
    frequencies = np.zeros_like(counts)
    np.divide(
        counts,
        n_called[:, None],
        out=frequencies,
        where=n_called[:, None] > 0,
    )

    # Zigo makes the features invariant to reference/alternate allele orientation,
    # except when genotype-0 calls are absent in a single-sample callset.
    haploid_encoded = frequencies[:, 2] == 0
    swap_01 = haploid_encoded & (frequencies[:, 1] > frequencies[:, 0])
    if np.any(swap_01):
        old_0 = frequencies[swap_01, 0].copy()
        frequencies[swap_01, 0] = frequencies[swap_01, 1]
        frequencies[swap_01, 1] = old_0

    swap_02 = (
        ~haploid_encoded
        & (frequencies[:, 2] > frequencies[:, 0])
        & (frequencies[:, 0] != 0)
    )
    if np.any(swap_02):
        old_0 = frequencies[swap_02, 0].copy()
        frequencies[swap_02, 0] = frequencies[swap_02, 2]
        frequencies[swap_02, 2] = old_0

    return frequencies, n_called


def _feature_exponents(feature_names: Sequence[str]) -> np.ndarray:
    exponents = np.zeros((len(feature_names), 3), dtype=np.int8)
    for row, feature_name in enumerate(feature_names):
        name = str(feature_name).strip()
        if name == "1":
            continue
        for token in name.split():
            match = _POLY_TOKEN_RE.fullmatch(token)
            if match is None:
                raise ValueError(f"Unrecognized Zigo polynomial feature token: {token!r}.")
            index = int(match.group(1))
            if index >= 3:
                raise ValueError(f"Zigo polynomial feature index out of range: x{index}.")
            exponents[row, index] += int(match.group(2) or 1)
    return exponents


@lru_cache(maxsize=1)
def _load_zigo_model() -> Tuple[float, np.ndarray, np.ndarray]:
    model_resource = resources.files("snputils.snp").joinpath("models", "zigo.json")
    with model_resource.open("r", encoding="utf-8") as handle:
        model = json.load(handle)

    required = {"intercept", "coefs", "feature_names"}
    if not required.issubset(model):
        raise ValueError("The bundled Zigo model is missing required polynomial parameters.")

    coefficients = np.asarray(model["coefs"], dtype=np.float64)
    feature_names = list(model["feature_names"])
    if coefficients.ndim != 1 or coefficients.shape[0] != len(feature_names):
        raise ValueError("The bundled Zigo model has inconsistent coefficients and features.")
    return float(model["intercept"]), coefficients, _feature_exponents(feature_names)


def _predict_zigo(frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    intercept, coefficients, exponents = _load_zigo_model()
    x = np.asarray(frequencies, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("Zigo inference requires a feature matrix with shape (n_samples, 3).")

    basis = np.prod(
        np.power(x[:, None, :], exponents[None, :, :], dtype=np.float64),
        axis=2,
    )
    logits = intercept + basis @ coefficients

    probabilities_male = np.empty(logits.shape[0], dtype=np.float64)
    nonnegative = logits >= 0
    probabilities_male[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    probabilities_male[~nonnegative] = exp_logits / (1.0 + exp_logits)
    probabilities_male = np.clip(probabilities_male, 1e-6, 1.0 - 1e-6)

    # Preserve Zigo's exact behavior for the all-genotype-2 simplex vertex.
    all_genotype_2 = (x[:, 0] == 0) & (x[:, 1] == 0) & (x[:, 2] > 0)
    probabilities_male[all_genotype_2] = 1.0
    probabilities_female = 1.0 - probabilities_male
    return probabilities_male, probabilities_female


def _normalize_reported_sex(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    normalized = str(value).strip().lower()
    if normalized in {"1", "m", "male"}:
        return "male"
    if normalized in {"2", "f", "female"}:
        return "female"
    return None


def _reported_sex_values(
    snpobj: SNPObject,
    sample_ids: np.ndarray,
    reported_sex: Optional[Union[Mapping[str, Any], Sequence[Any], np.ndarray]],
) -> np.ndarray:
    source: Any = snpobj.sample_sex if reported_sex is None else reported_sex
    if source is None:
        return np.full(sample_ids.shape[0], None, dtype=object)

    if isinstance(source, Mapping):
        values = [source.get(str(sample)) for sample in sample_ids]
    else:
        if isinstance(source, (str, bytes)):
            raise TypeError("`reported_sex` must be a sample-aligned sequence or mapping, not a string.")
        values = np.asarray(source, dtype=object).ravel()
        if values.shape[0] != sample_ids.shape[0]:
            raise ValueError(
                f"`reported_sex` has {values.shape[0]} entries, expected {sample_ids.shape[0]}."
            )

    return np.asarray([_normalize_reported_sex(value) for value in values], dtype=object)


def sex_check(
    snpobj: SNPObject,
    reported_sex: Optional[Union[Mapping[str, Any], Sequence[Any], np.ndarray]] = None,
    *,
    x_chromosomes: Union[str, Sequence[Union[str, int]]] = _DEFAULT_X_CHROMOSOMES,
    assume_x: bool = False,
    low_information_threshold: int = 500,
) -> pd.DataFrame:
    """Infer genetic sex from chromosome-X zygosity distributions using Zigo.

    The function consumes hard-call genotypes already represented by a
    :class:`~snputils.SNPObject`; file parsing remains the responsibility of
    snputils readers. Chromosome-X variants are selected automatically from
    ``variants_chrom``. The three model features are the normalized frequencies
    of genotype classes 0, 1, and 2 after Zigo's allele-orientation normalization.

    Args:
        snpobj:
            SNP data with 2D dosage calls ``(variants, samples)`` or 3D biallelic
            allele calls ``(variants, samples, 2)``.
        reported_sex:
            Optional sample-aligned sex values or mapping from sample ID to sex.
            Values ``1``/``M``/``male`` and ``2``/``F``/``female`` are recognized.
            If omitted, ``snpobj.sample_sex`` is used when available.
        x_chromosomes:
            Chromosome labels treated as X. Defaults to ``X``, ``chrX``, and the
            PLINK numeric label ``23``.
        assume_x:
            Treat every variant as chromosome X. Use only when chromosome metadata
            is unavailable and the object is known to contain X variants exclusively.
        low_information_threshold:
            Callable genotype count below which a non-empty sample is marked
            ``low_information``. The prediction is retained. Set to 0 to disable
            this flag.

    Returns:
        A sample-level DataFrame containing reported and inferred sex, comparison
        status, male/female probabilities, callable counts, normalized Zigo
        features, and a QC status.

    Raises:
        TypeError: If ``snpobj`` is not an SNPObject.
        ValueError: If hard calls or chromosome-X variants are unavailable.

    Notes:
        This integrates the distilled model from *Sex checking by zygosity
        distributions* (Molina-Sedano et al., 2026),
        https://doi.org/10.64898/2026.03.15.711924. The model was not designed to
        diagnose sex-chromosome aneuploidies.
    """
    if not isinstance(snpobj, SNPObject):
        raise TypeError("`snpobj` must be an SNPObject.")
    if (
        isinstance(low_information_threshold, bool)
        or not isinstance(low_information_threshold, (int, np.integer))
        or int(low_information_threshold) < 0
    ):
        raise ValueError("`low_information_threshold` must be a non-negative integer.")
    low_information_threshold = int(low_information_threshold)

    if snpobj.genotypes is None:
        if snpobj.calldata_gp is not None:
            raise ValueError(
                "Sex checking requires hard-call genotypes; genotype probabilities must be hard-called first."
            )
        raise ValueError("Sex checking requires genotype data in `SNPObject.genotypes`.")

    genotypes = np.asarray(snpobj.genotypes)
    if genotypes.ndim not in (2, 3):
        raise ValueError("Sex checking requires a 2D dosage or 3D allele-call genotype array.")
    n_variants, n_samples = genotypes.shape[:2]
    x_mask = _x_variant_mask(
        snpobj,
        n_variants,
        x_chromosomes=x_chromosomes,
        assume_x=assume_x,
    )
    frequencies, n_called = _zygosity_distribution(genotypes[x_mask])
    probabilities_male, probabilities_female = _predict_zigo(frequencies)

    no_data = n_called == 0
    probabilities_male[no_data] = np.nan
    probabilities_female[no_data] = np.nan

    inferred_sex = np.full(n_samples, None, dtype=object)
    informative = ~no_data
    inferred_sex[informative] = np.where(
        probabilities_male[informative] > 0.5, "male", "female"
    )

    if snpobj.samples is None:
        sample_ids = np.asarray([f"sample{i + 1}" for i in range(n_samples)], dtype=object)
    else:
        sample_ids = np.asarray(snpobj.samples, dtype=object).ravel()
        if sample_ids.shape[0] != n_samples:
            raise ValueError("`samples` must be aligned with the genotype sample axis.")

    if snpobj.sample_fid is None:
        family_ids = sample_ids.copy()
    else:
        family_ids = np.asarray(snpobj.sample_fid, dtype=object).ravel()
        if family_ids.shape[0] != n_samples:
            raise ValueError("`sample_fid` must be aligned with the genotype sample axis.")

    reported = _reported_sex_values(snpobj, sample_ids, reported_sex)
    status = np.full(n_samples, "not_compared", dtype=object)
    status[no_data] = "unknown"
    has_reported_sex = np.fromiter(
        (value is not None for value in reported), dtype=bool, count=n_samples
    )
    comparable = informative & has_reported_sex
    status[comparable] = np.where(
        reported[comparable] == inferred_sex[comparable], "match", "mismatch"
    )

    qc_status = np.full(n_samples, "pass", dtype=object)
    qc_status[no_data] = "no_data"
    if low_information_threshold > 0:
        qc_status[(n_called > 0) & (n_called < low_information_threshold)] = "low_information"

    return pd.DataFrame(
        {
            "family_id": family_ids.astype(str),
            "sample": sample_ids.astype(str),
            "reported_sex": pd.array(reported, dtype="string"),
            "inferred_sex": pd.array(inferred_sex, dtype="string"),
            "status": status,
            "p_male": probabilities_male,
            "p_female": probabilities_female,
            "n_called": n_called,
            "genotype_0_frequency": frequencies[:, 0],
            "genotype_1_frequency": frequencies[:, 1],
            "genotype_2_frequency": frequencies[:, 2],
            "qc_status": qc_status,
        }
    )


__all__ = ["sex_check"]
