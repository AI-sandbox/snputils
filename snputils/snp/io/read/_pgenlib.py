import numpy as np


PHASED_ALLELE_FULL_READ_BYTES = 64 * 1024 * 1024
PHASED_ALLELE_CHUNK_BYTES = 16 * 1024 * 1024
PHASED_ALLELE_MAX_CHUNK_VARIANTS = 512


def _allele_int32_bytes(num_variants: int, num_samples: int) -> int:
    return int(num_variants) * 2 * int(num_samples) * np.dtype(np.int32).itemsize


def _phased_chunk_size(num_variants: int, num_samples: int) -> int:
    bytes_per_variant = 2 * int(num_samples) * np.dtype(np.int32).itemsize
    if bytes_per_variant == 0:
        return int(num_variants)
    return max(
        1,
        min(
            int(num_variants),
            PHASED_ALLELE_MAX_CHUNK_VARIANTS,
            PHASED_ALLELE_CHUNK_BYTES // bytes_per_variant,
        ),
    )


def estimate_separate_strands_peak_bytes(num_variants: int, num_samples: int) -> int:
    output_bytes = int(num_variants) * int(num_samples) * 2 * np.dtype(np.int8).itemsize
    int32_bytes = _allele_int32_bytes(num_variants, num_samples)
    phase_bytes = int(num_variants) * int(num_samples) * np.dtype(np.bool_).itemsize
    if int32_bytes <= PHASED_ALLELE_FULL_READ_BYTES:
        return output_bytes + int32_bytes + phase_bytes
    chunk_bytes = _allele_int32_bytes(_phased_chunk_size(num_variants, num_samples), num_samples)
    chunk_phase_bytes = (
        _phased_chunk_size(num_variants, num_samples)
        * int(num_samples)
        * np.dtype(np.bool_).itemsize
    )
    return output_bytes + chunk_bytes + chunk_phase_bytes


def _is_contiguous_variant_chunk(variant_idxs: np.ndarray) -> bool:
    return (
        variant_idxs.size > 0
        and int(variant_idxs[-1]) - int(variant_idxs[0]) + 1 == variant_idxs.size
        and np.all(np.diff(variant_idxs) == 1)
    )


def read_separate_strands(
    pgen_reader,
    variant_idxs: np.ndarray,
    num_variants: int,
    num_samples: int,
    *,
    require_phase: bool = False,
) -> np.ndarray:
    """Read diploid alleles into a compact `(variants, samples, 2)` int8 array."""
    if num_variants == 0 or num_samples == 0:
        return np.empty((num_variants, num_samples, 2), dtype=np.int8)

    variant_idxs = np.asarray(variant_idxs, dtype=np.uint32).ravel()
    allele_cols = 2 * int(num_samples)

    if _allele_int32_bytes(num_variants, num_samples) <= PHASED_ALLELE_FULL_READ_BYTES:
        genotypes = np.empty((num_variants, allele_cols), dtype=np.int32)
        if require_phase:
            phase_present = np.empty((num_variants, num_samples), dtype=np.bool_)
            pgen_reader.read_alleles_and_phasepresent_list(variant_idxs, genotypes, phase_present)
            _raise_if_unphased_heterozygote(genotypes.reshape((num_variants, num_samples, 2)), phase_present)
        else:
            pgen_reader.read_alleles_list(variant_idxs, genotypes)
        return genotypes.astype(np.int8).reshape((num_variants, num_samples, 2))

    genotypes = np.empty((num_variants, num_samples, 2), dtype=np.int8)
    chunk_size = _phased_chunk_size(num_variants, num_samples)
    allele_chunk = np.empty((chunk_size, allele_cols), dtype=np.int32)
    phase_chunk = np.empty((chunk_size, num_samples), dtype=np.bool_) if require_phase else None

    for start in range(0, num_variants, chunk_size):
        stop = min(start + chunk_size, num_variants)
        chunk_len = stop - start
        chunk_variant_idxs = variant_idxs[start:stop]
        chunk = allele_chunk[:chunk_len]
        phase = phase_chunk[:chunk_len] if phase_chunk is not None else None

        if _is_contiguous_variant_chunk(chunk_variant_idxs):
            if require_phase:
                pgen_reader.read_alleles_and_phasepresent_range(
                    int(chunk_variant_idxs[0]),
                    int(chunk_variant_idxs[-1]) + 1,
                    chunk,
                    phase,
                )
            else:
                pgen_reader.read_alleles_range(
                    int(chunk_variant_idxs[0]),
                    int(chunk_variant_idxs[-1]) + 1,
                    chunk,
                )
        else:
            if require_phase:
                pgen_reader.read_alleles_and_phasepresent_list(chunk_variant_idxs, chunk, phase)
            else:
                pgen_reader.read_alleles_list(chunk_variant_idxs, chunk)

        if require_phase:
            _raise_if_unphased_heterozygote(
                chunk.reshape((chunk_len, num_samples, 2)),
                phase,
            )

        np.copyto(
            genotypes[start:stop].reshape(chunk_len, allele_cols),
            chunk,
            casting="unsafe",
        )

    return genotypes


def _raise_if_unphased_heterozygote(alleles: np.ndarray, phase_present: np.ndarray) -> None:
    called = np.all(alleles >= 0, axis=2)
    heterozygous = alleles[:, :, 0] != alleles[:, :, 1]
    if np.any(called & heterozygous & ~phase_present):
        raise ValueError(
            "Cannot read unphased heterozygous PGEN genotypes with `sum_strands=False`; "
            "use `sum_strands=True` to load 0/1/2 genotype dosages."
        )
