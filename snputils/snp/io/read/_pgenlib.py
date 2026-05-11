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
    if int32_bytes <= PHASED_ALLELE_FULL_READ_BYTES:
        return output_bytes + int32_bytes
    chunk_bytes = _allele_int32_bytes(_phased_chunk_size(num_variants, num_samples), num_samples)
    return output_bytes + chunk_bytes


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
) -> np.ndarray:
    """Read diploid alleles into a compact `(variants, samples, 2)` int8 array."""
    if num_variants == 0 or num_samples == 0:
        return np.empty((num_variants, num_samples, 2), dtype=np.int8)

    variant_idxs = np.asarray(variant_idxs, dtype=np.uint32).ravel()
    allele_cols = 2 * int(num_samples)

    if _allele_int32_bytes(num_variants, num_samples) <= PHASED_ALLELE_FULL_READ_BYTES:
        genotypes = np.empty((num_variants, allele_cols), dtype=np.int32)
        pgen_reader.read_alleles_list(variant_idxs, genotypes)
        return genotypes.astype(np.int8).reshape((num_variants, num_samples, 2))

    genotypes = np.empty((num_variants, num_samples, 2), dtype=np.int8)
    chunk_size = _phased_chunk_size(num_variants, num_samples)
    allele_chunk = np.empty((chunk_size, allele_cols), dtype=np.int32)

    for start in range(0, num_variants, chunk_size):
        stop = min(start + chunk_size, num_variants)
        chunk_len = stop - start
        chunk_variant_idxs = variant_idxs[start:stop]
        chunk = allele_chunk[:chunk_len]

        if _is_contiguous_variant_chunk(chunk_variant_idxs):
            pgen_reader.read_alleles_range(
                int(chunk_variant_idxs[0]),
                int(chunk_variant_idxs[-1]) + 1,
                chunk,
            )
        else:
            pgen_reader.read_alleles_list(chunk_variant_idxs, chunk)

        np.copyto(
            genotypes[start:stop].reshape(chunk_len, allele_cols),
            chunk,
            casting="unsafe",
        )

    return genotypes
