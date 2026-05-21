from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pygrgl as pyg

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.snp.genobj.grgobj import GRGObject
from snputils.snp.genobj.snpobj import SNPObject
from snputils.visualization.constants import CHROM_SIZES


DEFAULT_SYNTHETIC_ANCESTRY_MAP = {"0": "AFR", "1": "EUR", "2": "EAS"}


_POPULATION_COORDS = {
    "AFR_ref": np.array([1.0, 0.0], dtype=np.float64),
    "EUR_ref": np.array([-0.5, 0.866], dtype=np.float64),
    "EAS_ref": np.array([-0.5, -0.866], dtype=np.float64),
    "ADMIXED_AFR_EUR": np.array([0.25, 0.433], dtype=np.float64),
    "ADMIXED_EUR_EAS": np.array([-0.5, 0.0], dtype=np.float64),
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

    lai = np.empty((n_snps, n_samples * 2), dtype=np.int8)
    for sample_idx, label in enumerate(labels.astype(str).tolist()):
        if label in label_to_ancestry:
            state = label_to_ancestry[label]
            lai[:, 2 * sample_idx : 2 * sample_idx + 2] = state
        else:
            props = rng.dirichlet(np.ones(n_ancestries))
            states = rng.choice(n_ancestries, size=(n_snps, 2), p=props)
            lai[:, 2 * sample_idx : 2 * sample_idx + 2] = states

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


def build_synthetic_mdpca_dataset(
    n_samples: int = 200,
    n_snps: int = 1_000,
    seed: int | None = 0,
    *,
    ancestry_map: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build one-array SNP, LAI, and labels inputs for mdPCA examples."""
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4 for mdPCA/maasMDS examples.")
    if n_snps <= 0:
        raise ValueError("n_snps must be positive.")

    ancestry_map = dict(DEFAULT_SYNTHETIC_ANCESTRY_MAP if ancestry_map is None else ancestry_map)
    n_ancestries = len(ancestry_map)
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
    label_coords = np.vstack([_POPULATION_COORDS[label] for label in labels_cycle])
    af_by_label_matrix = _allele_probabilities_from_coordinates(
        label_coords,
        len(all_variant_positions),
        rng,
        load_scale=1.2,
        noise_scale=0.02,
    )
    af_by_label_and_pos = {
        label: {
            int(pos): float(af_by_label_matrix[label_idx, pos_idx])
            for pos_idx, pos in enumerate(all_variant_positions.tolist())
        }
        for label_idx, label in enumerate(labels_cycle)
    }
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
        genotypes = np.empty((len(variant_ids), n_samples_per_array, 2), dtype=np.int8)
        array_rng = np.random.default_rng(None if seed is None else seed + array_idx + 10)
        for sample_idx, label in enumerate(labels.astype(str).tolist()):
            probs = np.asarray(
                [af_by_label_and_pos[label][int(pos)] for pos in variant_positions.tolist()],
                dtype=np.float64,
            )
            genotypes[:, sample_idx, :] = array_rng.binomial(
                1, probs[:, None], size=(len(variant_ids), 2)
            )
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
    """Build autosome-only LAI data for chromosome painting examples."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if windows_per_chromosome <= 0:
        raise ValueError("windows_per_chromosome must be positive.")
    if build not in CHROM_SIZES:
        raise ValueError(f"Unknown build {build!r}. Available builds: {sorted(CHROM_SIZES)}")

    ancestry_map = dict(DEFAULT_SYNTHETIC_ANCESTRY_MAP if ancestry_map is None else ancestry_map)
    autosomes = {str(chrom) for chrom in range(1, 23)}
    requested_chroms = [str(c) for c in (chromosomes or [*range(1, 23)])]
    chroms = [chrom for chrom in requested_chroms if chrom in autosomes]
    if not chroms:
        raise ValueError(
            "No autosomes selected. Pass chromosomes from '1' to '22'."
        )
    rng = np.random.default_rng(seed)
    samples = [f"sample{i}" for i in range(n_samples)]
    if male_samples is None:
        male_samples = [samples[-1]] if samples else []
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
