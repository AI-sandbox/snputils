from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.phenotype.genobj import (
    CovariateObject,
    MultiPhenotypeObject,
    PhenotypeObject,
)
from snputils.snp.genobj.snpobj import SNPObject
from snputils.visualization.constants import CHROM_SIZES

if TYPE_CHECKING:
    from snputils.snp.genobj.grgobj import GRGObject


DEFAULT_SYNTHETIC_ANCESTRY_MAP = {"0": "AFR", "1": "EUR", "2": "EAS"}


_POPULATION_COORDS = {
    "AFR_ref": np.array([1.0, 0.0], dtype=np.float64),
    "EUR_ref": np.array([-0.5, 0.866], dtype=np.float64),
    "EAS_ref": np.array([-0.5, -0.866], dtype=np.float64),
    "ADMIXED_AFR_EUR": np.array([0.25, 0.433], dtype=np.float64),
    "ADMIXED_EUR_EAS": np.array([-0.5, 0.0], dtype=np.float64),
}

_LABEL_ANCESTRY_SOURCE_WEIGHTS = {
    "AFR_ref": {"AFR": np.asarray([0.75, 0.20, 0.05], dtype=np.float64)},
    "EUR_ref": {"EUR": np.asarray([0.70, 0.20, 0.10], dtype=np.float64)},
    "EAS_ref": {"EAS": np.asarray([0.72, 0.18, 0.10], dtype=np.float64)},
    "ADMIXED_AFR_EUR": {
        "AFR": np.asarray([0.20, 0.65, 0.15], dtype=np.float64),
        "EUR": np.asarray([0.12, 0.58, 0.30], dtype=np.float64),
    },
    "ADMIXED_EUR_EAS": {
        "EUR": np.asarray([0.18, 0.52, 0.30], dtype=np.float64),
        "EAS": np.asarray([0.15, 0.62, 0.23], dtype=np.float64),
    },
}


def _balanced_labels(n_samples: int, labels: Sequence[str]) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if len(labels) == 0:
        raise ValueError("At least one label is required.")
    return np.asarray([labels[idx % len(labels)] for idx in range(n_samples)], dtype=object)


def _sample_latent_coordinates(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coords = np.vstack([_POPULATION_COORDS[str(label)] for label in labels.astype(str)])
    coords = coords + rng.normal(0.0, 0.08, size=coords.shape)
    return coords.astype(np.float64, copy=False)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - float(np.mean(values))
    sd = float(np.std(centered, ddof=0))
    if sd <= 0.0 or not np.isfinite(sd):
        return centered
    return centered / sd


def _allele_probabilities_from_coordinates(
    coords: np.ndarray,
    n_snps: int,
    rng: np.random.Generator,
    *,
    load_scale: float = 1.15,
    noise_scale: float = 0.10,
) -> np.ndarray:
    intercept = rng.normal(0.0, 0.45, size=n_snps)
    loadings = rng.normal(0.0, load_scale, size=(2, n_snps))
    logits = intercept[None, :] + coords @ loadings + rng.normal(0.0, noise_scale, size=(coords.shape[0], n_snps))
    return np.clip(1.0 / (1.0 + np.exp(-logits)), 0.02, 0.98)


def build_synthetic_snp_dataset(
    n_samples: int = 12,
    n_snps: int = 100,
    seed: int | None = 0,
    *,
    n_populations: int = 3,
    missing_rate: float = 0.0,
    phased: bool = True,
    chromosome: str = "1",
    sample_prefix: str = "S",
    variant_prefix: str = "rs_syn",
) -> SNPObject:
    """Build a small SNPObject with realistic metadata and population structure.

    Population labels are stored in ``sample_fid``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_snps <= 0:
        raise ValueError("n_snps must be positive.")
    if n_populations <= 0:
        raise ValueError("n_populations must be positive.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in the interval [0, 1).")

    rng = np.random.default_rng(seed)
    samples = np.asarray([f"{sample_prefix}{idx + 1:02d}" for idx in range(n_samples)], dtype=object)
    population_names = [f"POP_{chr(ord('A') + idx)}" for idx in range(n_populations)]
    sample_fid = _balanced_labels(n_samples, population_names)
    sample_sex = np.asarray([(idx % 2) + 1 for idx in range(n_samples)], dtype=np.int8)

    base_af = rng.beta(0.8, 0.8, size=n_snps)
    pop_offsets = rng.normal(0.0, 0.12, size=(n_populations, n_snps))
    pop_af = np.clip(base_af[None, :] + pop_offsets, 0.02, 0.98)
    pop_index = {pop: idx for idx, pop in enumerate(population_names)}

    genotypes = np.empty((n_snps, n_samples, 2), dtype=np.int8)
    for sample_idx, pop in enumerate(sample_fid.tolist()):
        probs = pop_af[pop_index[str(pop)]]
        genotypes[:, sample_idx, :] = rng.binomial(1, probs[:, None], size=(n_snps, 2))

    if missing_rate > 0:
        missing = rng.random(genotypes.shape) < missing_rate
        genotypes[missing] = -1

    refs = np.asarray(["A", "C", "G", "T"], dtype=object)
    alts = np.asarray(["G", "T", "A", "C"], dtype=object)
    variants_ref = np.resize(refs, n_snps)
    variants_alt = np.resize(alts, n_snps)
    variants_chrom = np.repeat(str(chromosome), n_snps).astype(object)
    variants_pos = np.arange(1, n_snps + 1, dtype=np.int64) * 1_000
    variants_id = np.asarray([f"{variant_prefix}_{idx + 1:05d}" for idx in range(n_snps)], dtype=object)

    if not phased:
        genotype_matrix = genotypes.sum(axis=2, dtype=np.int8)
        genotype_matrix[np.any(genotypes < 0, axis=2)] = -1
    else:
        genotype_matrix = genotypes

    return SNPObject(
        genotypes=genotype_matrix,
        samples=samples,
        sample_fid=sample_fid,
        sample_sex=sample_sex,
        variants_ref=variants_ref,
        variants_alt=variants_alt,
        variants_chrom=variants_chrom,
        variants_id=variants_id,
        variants_pos=variants_pos,
    )


def build_synthetic_phenotype_dataset(
    n_samples: int = 24,
    n_snps: int = 200,
    seed: int | None = 0,
    *,
    snpobj: SNPObject | None = None,
    missing_rate: float = 0.0,
) -> dict[str, object]:
    """Build a sample-aligned SNP/phenotype/covariate cohort for tutorials.

    The returned objects are intentionally small and self-contained so docs and
    tests can demonstrate phenotype handling, file readers, and association
    workflows without downloading external data.
    """
    rng = np.random.default_rng(seed)
    if snpobj is None:
        snpobj = build_synthetic_snp_dataset(
            n_samples=n_samples,
            n_snps=n_snps,
            seed=seed,
            missing_rate=missing_rate,
            phased=True,
        )
    elif not isinstance(snpobj, SNPObject):
        raise TypeError("snpobj must be a SNPObject.")

    sample_ids = np.asarray(snpobj.samples, dtype=object).astype(str)
    if sample_ids.size == 0:
        raise ValueError("snpobj must contain at least one sample.")

    genotypes = np.asarray(snpobj.genotypes)
    if genotypes.ndim == 3:
        dosages = genotypes.sum(axis=2, dtype=np.int16).astype(np.float64)
        dosages[np.any(genotypes < 0, axis=2)] = np.nan
    elif genotypes.ndim == 2:
        dosages = genotypes.astype(np.float64, copy=False)
        dosages[dosages < 0] = np.nan
    else:
        raise ValueError("Synthetic phenotype generation expects 2D or 3D genotypes.")

    n_snps_actual, n_samples_actual = dosages.shape
    if n_snps_actual == 0:
        raise ValueError("snpobj must contain at least one variant.")

    effect_count = min(3, n_snps_actual)
    effect_indexes = np.linspace(0, n_snps_actual - 1, num=effect_count, dtype=np.int64)
    effect_weights = np.array([0.95, -0.70, 0.45], dtype=np.float64)[:effect_count]

    dosage_matrix = dosages[effect_indexes].T.copy()
    col_means = np.nanmean(dosage_matrix, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 1.0)
    missing = np.isnan(dosage_matrix)
    if np.any(missing):
        dosage_matrix[missing] = np.take(col_means, np.where(missing)[1])
    dosage_signal = _standardize(dosage_matrix @ effect_weights)

    age = rng.integers(24, 71, size=n_samples_actual, endpoint=False).astype(np.int16)
    batch = ((np.arange(n_samples_actual) % 2) + 1).astype(np.int8)
    sex = np.asarray(getattr(snpobj, "sample_sex", ((np.arange(n_samples_actual) % 2) + 1)), dtype=np.int8)

    sample_fid = getattr(snpobj, "sample_fid", None)
    if sample_fid is None:
        ancestry_shift = np.zeros(n_samples_actual, dtype=np.float64)
    else:
        labels = np.asarray(sample_fid, dtype=object).astype(str)
        unique_labels = sorted(set(labels.tolist()))
        shift_map = {
            label: offset
            for label, offset in zip(
                unique_labels,
                np.linspace(-0.4, 0.4, num=len(unique_labels), dtype=np.float64),
            )
        }
        ancestry_shift = np.asarray([shift_map[label] for label in labels], dtype=np.float64)

    quantitative_values = (
        0.05 * (age.astype(np.float64) - float(np.mean(age)))
        + 0.30 * (batch == 2).astype(np.float64)
        - 0.22 * (sex == 2).astype(np.float64)
        + 0.85 * dosage_signal
        + ancestry_shift
        + rng.normal(0.0, 0.55, size=n_samples_actual)
    )
    quantitative_values = _standardize(quantitative_values)

    liability = (
        0.70 * dosage_signal
        + 0.35 * _standardize(age)
        + 0.25 * (batch == 2).astype(np.float64)
        - 0.20 * (sex == 2).astype(np.float64)
        + 0.35 * ancestry_shift
        + rng.normal(0.0, 0.65, size=n_samples_actual)
    )
    binary_01 = (liability > np.median(liability)).astype(np.int8)
    if binary_01.sum() == 0 or binary_01.sum() == binary_01.size:
        order = np.argsort(liability)
        binary_01 = np.zeros(n_samples_actual, dtype=np.int8)
        binary_01[order[n_samples_actual // 2 :]] = 1
    binary_12 = (binary_01 + 1).astype(np.int8)

    phenotype_table = pd.DataFrame(
        {
            "IID": sample_ids,
            "trait_quantitative": np.round(quantitative_values, 4),
            "trait_binary_01": binary_01,
            "trait_binary_12": binary_12,
            "age": age.astype(np.int64),
            "batch": batch.astype(np.int64),
            "sex": sex.astype(np.int64),
        }
    )

    quantitative = PhenotypeObject(
        samples=sample_ids,
        values=phenotype_table["trait_quantitative"].to_numpy(),
        phenotype_name="TRAIT_Q",
        quantitative=True,
    )
    binary = PhenotypeObject(
        samples=sample_ids,
        values=phenotype_table["trait_binary_12"].to_numpy(),
        phenotype_name="TRAIT_BIN",
        quantitative=None,
    )
    covariates = CovariateObject(
        samples=sample_ids,
        values=phenotype_table[["age", "batch", "sex"]].to_numpy(),
        covariate_names=["age", "batch", "sex"],
    )

    return {
        "snpobj": snpobj,
        "phen_df": phenotype_table,
        "multi_phenotype": MultiPhenotypeObject(phenotype_table.copy(), sample_column="IID"),
        "quantitative": quantitative,
        "binary": binary,
        "covariates": covariates,
        "effect_variant_ids": np.asarray(snpobj.variants_id, dtype=object)[effect_indexes].astype(str).tolist(),
        "shuffled_phen_df": phenotype_table.sample(frac=1.0, random_state=seed).reset_index(drop=True),
    }


def _make_structured_snpobject(
    *,
    samples: np.ndarray,
    labels: np.ndarray,
    variant_ids: np.ndarray,
    variant_positions: np.ndarray,
    seed: int | None,
    chromosome: str = "1",
    missing_rate: float = 0.0,
) -> SNPObject:
    rng = np.random.default_rng(seed)
    n_snps = len(variant_ids)
    n_samples = len(samples)
    coords = _sample_latent_coordinates(labels, rng)
    sample_af = _allele_probabilities_from_coordinates(coords, n_snps, rng)

    genotypes = np.empty((n_snps, n_samples, 2), dtype=np.int8)
    for sample_idx in range(n_samples):
        probs = sample_af[sample_idx]
        genotypes[:, sample_idx, :] = rng.binomial(1, probs[:, None], size=(n_snps, 2))
    if missing_rate > 0:
        missing = rng.random(genotypes.shape) < missing_rate
        genotypes[missing] = -1

    refs = np.asarray(["A", "C", "G", "T"], dtype=object)
    alts = np.asarray(["G", "T", "A", "C"], dtype=object)
    return SNPObject(
        genotypes=genotypes,
        samples=samples.astype(object),
        sample_fid=labels.astype(object),
        sample_sex=np.asarray([(idx % 2) + 1 for idx in range(n_samples)], dtype=np.int8),
        variants_ref=np.resize(refs, n_snps),
        variants_alt=np.resize(alts, n_snps),
        variants_chrom=np.repeat(str(chromosome), n_snps).astype(object),
        variants_id=variant_ids.astype(object),
        variants_pos=variant_positions.astype(np.int64, copy=False),
    )


def _build_lai_for_snpobject(
    snpobj: SNPObject,
    labels: np.ndarray,
    ancestry_map: dict[str, str],
    seed: int | None,
    *,
    lai: np.ndarray | None = None,
) -> LocalAncestryObject:
    rng = np.random.default_rng(seed)
    n_snps = snpobj.n_snps
    n_samples = snpobj.n_samples
    n_ancestries = len(ancestry_map)
    ancestry_names = list(ancestry_map.values())
    label_to_ancestry = {
        "AFR_ref": ancestry_names.index("AFR") if "AFR" in ancestry_names else 0,
        "EUR_ref": ancestry_names.index("EUR") if "EUR" in ancestry_names else min(1, n_ancestries - 1),
        "EAS_ref": ancestry_names.index("EAS") if "EAS" in ancestry_names else min(2, n_ancestries - 1),
    }

    if lai is None:
        lai = np.empty((n_snps, n_samples * 2), dtype=np.int8)
        for sample_idx, label in enumerate(labels.astype(str).tolist()):
            if label in label_to_ancestry:
                state = label_to_ancestry[label]
                lai[:, 2 * sample_idx : 2 * sample_idx + 2] = state
            else:
                props = rng.dirichlet(np.ones(n_ancestries))
                states = rng.choice(n_ancestries, size=(n_snps, 2), p=props)
                lai[:, 2 * sample_idx : 2 * sample_idx + 2] = states
    else:
        lai = np.asarray(lai, dtype=np.int8, order="C")
        if lai.shape != (n_snps, n_samples * 2):
            raise ValueError(
                f"Provided lai has shape {lai.shape}, expected {(n_snps, n_samples * 2)}."
            )
        if np.any((lai < 0) | (lai >= n_ancestries)):
            raise ValueError("Provided lai contains ancestry codes outside ancestry_map.")

    physical_pos = np.column_stack([snpobj.variants_pos, snpobj.variants_pos + 999])
    laiobj = LocalAncestryObject(
        haplotypes=[f"{sample}.{phase}" for sample in snpobj.samples.astype(str) for phase in (0, 1)],
        lai=lai,
        samples=snpobj.samples.astype(str).tolist(),
        ancestry_map=ancestry_map,
        chromosomes=snpobj.variants_chrom.copy(),
        physical_pos=physical_pos.astype(np.int64, copy=False),
        window_sizes=np.ones(n_snps, dtype=np.int64),
    )
    snpobj.calldata_lai = lai.reshape(n_snps, n_samples, 2)
    snpobj.ancestry_map = ancestry_map
    return laiobj


def _sample_lai_matrix_for_labels(
    labels: np.ndarray,
    n_snps: int,
    ancestry_map: dict[str, str],
    rng: np.random.Generator,
) -> np.ndarray:
    ancestry_names = list(ancestry_map.values())
    n_ancestries = len(ancestry_names)
    ancestry_index = {name: idx for idx, name in enumerate(ancestry_names)}

    def _one_hot(name: str) -> np.ndarray:
        weights = np.zeros(n_ancestries, dtype=np.float64)
        weights[ancestry_index.get(name, 0)] = 1.0
        return weights

    lai = np.empty((n_snps, labels.size * 2), dtype=np.int8)
    labels_list = labels.astype(str).tolist()
    for sample_idx, label in enumerate(labels_list):
        if label == "AFR_ref":
            base_props = _one_hot("AFR")
        elif label == "EUR_ref":
            base_props = _one_hot("EUR")
        elif label == "EAS_ref":
            base_props = _one_hot("EAS")
        elif label == "ADMIXED_AFR_EUR":
            afr = ancestry_index.get("AFR", 0)
            eur = ancestry_index.get("EUR", 0)
            p_afr = float(rng.beta(2.2, 2.2))
            base_props = np.zeros(n_ancestries, dtype=np.float64)
            base_props[afr] = p_afr
            base_props[eur] = 1.0 - p_afr
        elif label == "ADMIXED_EUR_EAS":
            eur = ancestry_index.get("EUR", 0)
            eas = ancestry_index.get("EAS", 0)
            p_eur = float(rng.beta(2.2, 2.2))
            base_props = np.zeros(n_ancestries, dtype=np.float64)
            base_props[eur] = p_eur
            base_props[eas] = 1.0 - p_eur
        else:
            base_props = rng.dirichlet(np.ones(n_ancestries, dtype=np.float64))

        for hap in range(2):
            col = 2 * sample_idx + hap
            current = int(rng.choice(n_ancestries, p=base_props))
            for snp_idx in range(n_snps):
                lai[snp_idx, col] = current
                if rng.random() < (1.0 / 35.0):
                    current = int(rng.choice(n_ancestries, p=base_props))
    return lai


def build_synthetic_mdpca_dataset(
    n_samples: int = 200,
    n_snps: int = 1_000,
    seed: int | None = 0,
    *,
    ancestry_map: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build one-array SNP, LAI, and labels inputs for mdPCA examples.

    Notes:
        - This generator is intended for missing-data mdPCA tutorials.
        - Genotypes are generated from sample-level label structure, while LAI is
          generated separately. In admixed labels, haplotype ancestry states are
          not used to drive ancestry-specific allele frequencies.
        - For ancestry-specific masking demos, prefer
          :func:`build_synthetic_maasmds_dataset`, which couples haplotype
          genotypes to local ancestry states.
    """
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4 for mdPCA/maasMDS examples.")
    if n_snps <= 0:
        raise ValueError("n_snps must be positive.")

    ancestry_map = dict(DEFAULT_SYNTHETIC_ANCESTRY_MAP if ancestry_map is None else ancestry_map)
    labels_cycle = ["AFR_ref", "EUR_ref", "EAS_ref", "ADMIXED_AFR_EUR", "ADMIXED_EUR_EAS"]
    labels = _balanced_labels(n_samples, labels_cycle)
    samples = np.asarray([f"M{i + 1:04d}" for i in range(n_samples)], dtype=object)
    variant_positions = np.arange(1, n_snps + 1, dtype=np.int64) * 1_000
    variant_ids = np.asarray([f"rs_mdpca_{idx + 1:06d}" for idx in range(n_snps)], dtype=object)
    snpobj = _make_structured_snpobject(
        samples=samples,
        labels=labels,
        variant_ids=variant_ids,
        variant_positions=variant_positions,
        seed=seed,
        missing_rate=0.01,
    )
    labels_df = pd.DataFrame({"indID": snpobj.samples.astype(str), "label": labels})
    laiobj = _build_lai_for_snpobject(snpobj, labels, ancestry_map, seed)

    return {
        "snpobj": snpobj,
        "laiobj": laiobj,
        "labels": labels_df,
        "ancestry_map": ancestry_map,
    }


def build_synthetic_maasmds_dataset(
    n_samples_per_array: int = 200,
    n_snps_per_array: int = 1_000,
    n_arrays: int = 3,
    seed: int | None = 0,
    *,
    ancestry_map: dict[str, str] | None = None,
    triple_shared_fraction: float = 0.25,
    pair_shared_fraction: float = 0.20,
) -> dict[str, object]:
    """Build multi-array SNP, LAI, and labels inputs for maasMDS examples.

    With the default three arrays and 1,000 SNPs per array, each array has
    250 SNPs shared across all arrays, 200 SNPs shared with each other array,
    and 350 array-specific SNPs. Thus overlap decays from within-array to
    pairwise intersections to the three-way intersection.

    Genotypes are sampled haplotype-by-haplotype from local ancestry states.
    Within each continental ancestry we simulate multiple latent sources so
    ancestry-masked analyses still contain population-specific structure.
    """
    if n_arrays != 3:
        raise ValueError("Only n_arrays=3 is currently supported.")
    if n_samples_per_array <= 0:
        raise ValueError("n_samples_per_array must be positive.")
    if n_snps_per_array <= 0:
        raise ValueError("n_snps_per_array must be positive.")
    ancestry_map = dict(DEFAULT_SYNTHETIC_ANCESTRY_MAP if ancestry_map is None else ancestry_map)

    n_all = int(round(n_snps_per_array * triple_shared_fraction))
    n_pair = int(round(n_snps_per_array * pair_shared_fraction))
    n_unique = n_snps_per_array - n_all - 2 * n_pair
    if n_all <= 0 or n_pair <= 0 or n_unique <= 0:
        raise ValueError(
            "Overlap fractions must leave positive all-shared, pair-shared, and unique SNP counts."
        )

    next_pos = 1

    def alloc(prefix: str, count: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal next_pos
        positions = np.arange(next_pos, next_pos + count, dtype=np.int64) * 1_000
        ids = np.asarray([f"{prefix}_{idx + 1:06d}" for idx in range(count)], dtype=object)
        next_pos += count
        return ids, positions

    all_ids, all_pos = alloc("rs_all_arrays", n_all)
    pair_defs = {(0, 1): alloc("rs_array_1_2", n_pair), (0, 2): alloc("rs_array_1_3", n_pair), (1, 2): alloc("rs_array_2_3", n_pair)}
    unique_defs = [alloc(f"rs_array_{idx + 1}_only", n_unique) for idx in range(n_arrays)]

    labels_cycle = ["AFR_ref", "EUR_ref", "EAS_ref", "ADMIXED_AFR_EUR", "ADMIXED_EUR_EAS"]
    rng = np.random.default_rng(seed)
    all_variant_positions = np.concatenate(
        [all_pos]
        + [pos for _, pos in pair_defs.values()]
        + [pos for _, pos in unique_defs]
    )
    ancestry_names = [str(name) for name in ancestry_map.values()]
    fallback_centroid = np.mean(
        np.vstack([_POPULATION_COORDS["AFR_ref"], _POPULATION_COORDS["EUR_ref"], _POPULATION_COORDS["EAS_ref"]]),
        axis=0,
    )
    ancestry_centroids = {
        "AFR": _POPULATION_COORDS["AFR_ref"],
        "EUR": _POPULATION_COORDS["EUR_ref"],
        "EAS": _POPULATION_COORDS["EAS_ref"],
    }
    n_sources_per_ancestry = 3
    source_offsets = np.asarray(
        [
            [0.18, 0.00],
            [-0.14, 0.12],
            [-0.07, -0.16],
        ],
        dtype=np.float64,
    )
    source_coord_rows = []
    source_row_lookup: dict[tuple[str, int], int] = {}
    for anc in ancestry_names:
        centroid = np.asarray(ancestry_centroids.get(anc, fallback_centroid), dtype=np.float64)
        coords = centroid[None, :] + source_offsets
        for src_idx in range(n_sources_per_ancestry):
            source_row_lookup[(anc, src_idx)] = len(source_coord_rows)
            source_coord_rows.append(coords[src_idx])

    af_by_source_matrix = _allele_probabilities_from_coordinates(
        np.vstack(source_coord_rows),
        len(all_variant_positions),
        rng,
        load_scale=1.25,
        noise_scale=0.02,
    )
    af_by_ancestry_source_and_pos = {
        anc: {
            src_idx: {
                int(pos): float(af_by_source_matrix[source_row_lookup[(anc, src_idx)], pos_idx])
                for pos_idx, pos in enumerate(all_variant_positions.tolist())
            }
            for src_idx in range(n_sources_per_ancestry)
        }
        for anc in ancestry_names
    }
    source_mean_by_ancestry = {
        anc: np.mean(
            np.stack(
                [
                    np.asarray(
                        [af_by_ancestry_source_and_pos[anc][src_idx][int(pos)] for pos in all_variant_positions.tolist()],
                        dtype=np.float64,
                    )
                    for src_idx in range(n_sources_per_ancestry)
                ],
                axis=0,
            ),
            axis=0,
        )
        for anc in ancestry_names
    }
    ancestry_mean_stack = np.stack([source_mean_by_ancestry[anc] for anc in ancestry_names], axis=0)
    global_maf = np.mean(ancestry_mean_stack, axis=0)
    ancestry_informativeness = np.max(ancestry_mean_stack, axis=0) - np.min(ancestry_mean_stack, axis=0)
    snpobjs = []
    laiobjs = []
    labels_frames = []
    for array_idx in range(n_arrays):
        samples = np.asarray(
            [f"array{array_idx + 1}_sample{i + 1:04d}" for i in range(n_samples_per_array)],
            dtype=object,
        )
        labels = _balanced_labels(n_samples_per_array, labels_cycle)
        ids_parts = [all_ids]
        pos_parts = [all_pos]
        for pair, (ids, pos) in pair_defs.items():
            if array_idx in pair:
                ids_parts.append(ids)
                pos_parts.append(pos)
        ids_parts.append(unique_defs[array_idx][0])
        pos_parts.append(unique_defs[array_idx][1])
        variant_ids = np.concatenate(ids_parts)
        variant_positions = np.concatenate(pos_parts)
        order = np.argsort(variant_positions)

        variant_ids = variant_ids[order]
        variant_positions = variant_positions[order]
        n_array_snps = len(variant_ids)
        genotypes = np.empty((n_array_snps, n_samples_per_array, 2), dtype=np.int8)
        array_rng = np.random.default_rng(None if seed is None else seed + array_idx + 10)
        lai = _sample_lai_matrix_for_labels(labels, n_array_snps, ancestry_map, array_rng)
        anc_name_by_code = {idx: str(name) for idx, name in enumerate(ancestry_map.values())}
        pos_to_global_index = {int(pos): idx for idx, pos in enumerate(all_variant_positions.tolist())}
        snp_global_index = np.asarray([pos_to_global_index[int(pos)] for pos in variant_positions.tolist()], dtype=np.int64)
        array_maf = global_maf[snp_global_index]
        array_aim = ancestry_informativeness[snp_global_index]
        if array_idx == 0:
            ascertainment_delta = -0.55 * (array_maf - np.mean(array_maf))
        elif array_idx == 1:
            ascertainment_delta = 0.55 * (array_maf - np.mean(array_maf))
        else:
            ascertainment_delta = 0.95 * (array_aim - np.mean(array_aim))

        probs_by_ancestry_source = {
            anc_name: {
                src_idx: np.asarray(
                    [af_by_ancestry_source_and_pos[anc_name][src_idx][int(pos)] for pos in variant_positions.tolist()],
                    dtype=np.float64,
                )
                for src_idx in range(n_sources_per_ancestry)
            }
            for anc_name in ancestry_names
        }

        def _source_weights_for_label_ancestry(label: str, ancestry: str) -> np.ndarray:
            label_map = _LABEL_ANCESTRY_SOURCE_WEIGHTS.get(label, {})
            if ancestry in label_map:
                base = np.asarray(label_map[ancestry], dtype=np.float64)
            else:
                base = np.full(n_sources_per_ancestry, 1.0 / n_sources_per_ancestry, dtype=np.float64)
            draw = array_rng.dirichlet(24.0 * base + 0.3)
            return draw / np.sum(draw)

        for sample_idx in range(n_samples_per_array):
            label = str(labels[sample_idx])
            source_by_ancestry: dict[str, int] = {}
            for anc_name in ancestry_names:
                weights = _source_weights_for_label_ancestry(label, anc_name)
                source_by_ancestry[anc_name] = int(array_rng.choice(n_sources_per_ancestry, p=weights))
            for hap in range(2):
                codes = lai[:, 2 * sample_idx + hap]
                hap_probs = np.empty(n_array_snps, dtype=np.float64)
                for code, anc_name in anc_name_by_code.items():
                    mask = codes == code
                    if np.any(mask):
                        source_idx = source_by_ancestry[anc_name]
                        hap_probs[mask] = probs_by_ancestry_source[anc_name][source_idx][mask]
                adjusted_probs = np.clip(hap_probs + ascertainment_delta, 0.01, 0.99)
                genotypes[:, sample_idx, hap] = array_rng.binomial(1, adjusted_probs)
        missing = array_rng.random(genotypes.shape) < 0.01
        genotypes[missing] = -1
        refs = np.asarray(["A", "C", "G", "T"], dtype=object)
        alts = np.asarray(["G", "T", "A", "C"], dtype=object)
        snpobj = SNPObject(
            genotypes=genotypes,
            samples=samples.astype(object),
            sample_fid=labels.astype(object),
            sample_sex=np.asarray([(idx % 2) + 1 for idx in range(n_samples_per_array)], dtype=np.int8),
            variants_ref=np.resize(refs, len(variant_ids)),
            variants_alt=np.resize(alts, len(variant_ids)),
            variants_chrom=np.repeat("1", len(variant_ids)).astype(object),
            variants_id=variant_ids.astype(object),
            variants_pos=variant_positions.astype(np.int64, copy=False),
        )
        laiobj = _build_lai_for_snpobject(
            snpobj,
            labels,
            ancestry_map,
            None if seed is None else seed + array_idx + 100,
            lai=lai,
        )
        snpobjs.append(snpobj)
        laiobjs.append(laiobj)
        labels_frames.append(pd.DataFrame({"indID": samples.astype(str), "label": labels, "array": array_idx + 1}))

    variant_sets = [set(map(int, snp.variants_pos.tolist())) for snp in snpobjs]
    overlap_counts = {
        "array_1": len(variant_sets[0]),
        "array_2": len(variant_sets[1]),
        "array_3": len(variant_sets[2]),
        "array_1_2": len(variant_sets[0] & variant_sets[1]),
        "array_1_3": len(variant_sets[0] & variant_sets[2]),
        "array_2_3": len(variant_sets[1] & variant_sets[2]),
        "array_1_2_3": len(variant_sets[0] & variant_sets[1] & variant_sets[2]),
    }
    return {
        "snpobjs": snpobjs,
        "laiobjs": laiobjs,
        "labels": pd.concat(labels_frames, ignore_index=True),
        "ancestry_map": ancestry_map,
        "overlap_counts": overlap_counts,
    }


def build_synthetic_chromosome_painting_dataset(
    n_samples: int = 3,
    windows_per_chromosome: int = 60,
    seed: int | None = 42,
    *,
    build: str = "hg38",
    chromosomes: Sequence[str] | None = None,
    ancestry_map: dict[str, str] | None = None,
    male_samples: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build synthetic LAI data for chromosome painting examples.

    By default this simulates diploid local ancestry across chromosomes ``1``-``22``
    and ``X``. The default sample metadata marks every sample as female so the
    reported sex is consistent with the diploid-X simulation used for painting.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if windows_per_chromosome <= 0:
        raise ValueError("windows_per_chromosome must be positive.")
    if build not in CHROM_SIZES:
        raise ValueError(f"Unknown build {build!r}. Available builds: {sorted(CHROM_SIZES)}")

    ancestry_map = dict(DEFAULT_SYNTHETIC_ANCESTRY_MAP if ancestry_map is None else ancestry_map)
    supported_chroms = {str(chrom) for chrom in range(1, 23)} | {"X"}
    requested_chroms = [str(c) for c in (chromosomes or [*range(1, 23), "X"])]
    chroms = [chrom for chrom in requested_chroms if chrom in supported_chroms]
    if not chroms:
        raise ValueError(
            "No supported chromosomes selected. Pass chromosomes from '1' to '22' and/or 'X'."
        )
    rng = np.random.default_rng(seed)
    samples = [f"sample{i}" for i in range(n_samples)]
    if male_samples is None:
        male_samples = []
    male_samples = set(map(str, male_samples))

    n_windows = len(chroms) * windows_per_chromosome
    n_haplotypes = n_samples * 2
    chrom_array = np.repeat(np.asarray(chroms, dtype=object), windows_per_chromosome)
    physical_pos = np.zeros((n_windows, 2), dtype=np.int64)
    centimorgan_pos = np.zeros((n_windows, 2), dtype=np.float64)
    window_sizes = np.full(n_windows, 10, dtype=np.int32)

    for chrom in chroms:
        chrom_size = CHROM_SIZES[build][chrom]
        win_size = chrom_size // windows_per_chromosome
        idx = np.where(chrom_array == chrom)[0]
        for local_i, global_i in enumerate(idx):
            start = local_i * win_size
            end = start + win_size if local_i < windows_per_chromosome - 1 else chrom_size
            physical_pos[global_i] = [start, end]
            centimorgan_pos[global_i] = [start / 1e6, end / 1e6]

    lai = np.zeros((n_windows, n_haplotypes), dtype=np.int32)
    n_ancestries = len(ancestry_map)
    for hap in range(n_haplotypes):
        for chrom_idx in range(len(chroms)):
            start = chrom_idx * windows_per_chromosome
            stop = start + windows_per_chromosome
            pos = start
            while pos < stop:
                block_len = int(rng.integers(4, 20))
                lai[pos : min(pos + block_len, stop), hap] = int(rng.integers(0, n_ancestries))
                pos += block_len

    laiobj = LocalAncestryObject(
        haplotypes=[f"{sample}.{phase}" for sample in samples for phase in (0, 1)],
        lai=lai,
        samples=samples,
        ancestry_map=ancestry_map,
        chromosomes=chrom_array,
        physical_pos=physical_pos,
        centimorgan_pos=centimorgan_pos,
        window_sizes=window_sizes,
    )
    sample_sex = pd.DataFrame(
        {
            "sample": samples,
            "sex": ["male" if sample in male_samples else "female" for sample in samples],
        }
    )
    return {
        "laiobj": laiobj,
        "sample_sex": sample_sex,
        "build": build,
        "chromosomes": chroms,
        "ancestry_map": ancestry_map,
    }


def build_synthetic_grg() -> GRGObject:
    """Build a tiny deterministic GRGObject."""
    try:
        import pygrgl as pyg
        from snputils.snp.genobj.grgobj import GRGObject
    except ModuleNotFoundError as exc:
        if exc.name == "pygrgl":
            raise ImportError(
                "GRG support requires the optional dependency 'pygrgl'. "
                "Install it with: pip install 'snputils[grg]'"
            ) from exc
        raise

    grg = pyg.MutableGRG(6, 2, True)
    root = grg.make_node()
    left = grg.make_node()
    right = grg.make_node()

    grg.connect(root, left)
    grg.connect(root, right)
    for sample in [0, 1, 2]:
        grg.connect(left, sample)
    for sample in [3, 4, 5]:
        grg.connect(right, sample)

    grg.add_mutation(pyg.Mutation(100.0, "G", "A", 0.0), root)
    grg.add_mutation(pyg.Mutation(110.0, "T", "C", 0.0), left)
    grg.add_mutation(pyg.Mutation(120.0, "C", "G", 0.0), right)
    grg.add_mutation(pyg.Mutation(130.0, "A", "T", 0.0), 0)
    grg.add_mutation(pyg.Mutation(140.0, "G", "T", 0.0), 5)
    return GRGObject(genotypes=grg, mutable=True)
