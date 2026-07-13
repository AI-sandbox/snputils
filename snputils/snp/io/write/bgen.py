from __future__ import annotations

import logging
import math
import struct
import zlib
from importlib import import_module
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zstandard as zstd

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.write._genotype_encoding import validate_hardcall_values

log = logging.getLogger(__name__)

_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")

try:
    _native_bgen = import_module("snputils.snp.io._bgen")
except ImportError:  # pragma: no cover - exercised only when the extension is unavailable
    _native_bgen = None


class BGENWriter:
    """
    Write a SNPObject to BGEN format.

    ``calldata_gp`` is written directly when present. If it is absent and
    ``genotypes`` is present, hard calls are encoded as one-hot genotype
    probabilities so SNPObjects created from VCF/BED/PGEN can still be exported.
    """

    def __init__(self, snpobj: SNPObject, filename: Union[str, Path]):
        """
        Initialize the BGENWriter.

        Args:
            snpobj: SNPObject containing genotype probabilities or hard-call genotypes.
            filename: Output path. A ``.bgen`` suffix is appended if missing.
        """
        self.__snpobj = snpobj
        self.__filename = Path(filename)

    def write(
        self,
        compression: Optional[str] = "zstd",
        layout: int = 2,
        bit_depth: int = 8,
        phased: Optional[bool] = None,
        metadata: Optional[str] = None,
    ) -> None:
        """
        Write the SNPObject to a BGEN file.

        Args:
            compression: BGEN compression type. Supported by the backend: ``None``,
                ``"zlib"``, and ``"zstd"``.
            layout: BGEN layout version. The backend supports layouts 1 and 2.
            bit_depth: Number of bits used to store each probability.
            phased: Whether probabilities are phased. If None, inferred per variant
                from ``calldata_gp`` width and NaN padding when possible.
            metadata: Optional free-form BGEN metadata string.
        """
        output = self.__filename
        if output.suffix != ".bgen":
            output = output.with_suffix(".bgen")

        probabilities = self._probabilities(phased)
        if probabilities.ndim != 3:
            raise ValueError("BGEN genotype probabilities must have shape (n_snps, n_samples, n_probabilities).")

        n_variants, n_samples, _ = probabilities.shape

        samples = self._samples(n_samples)
        variants_ref = self._required_variant_column("variants_ref", n_variants)
        variants_alt = self._required_variant_column("variants_alt", n_variants)
        variants_chrom = self._required_variant_column("variants_chrom", n_variants)
        variants_pos = self._required_variant_column("variants_pos", n_variants)
        variants_id = self._variant_ids(n_variants)

        log.info(f"Writing to {output}")
        if layout != 2:
            raise NotImplementedError("Native BGENWriter currently supports BGEN layout 2 files.")

        compression_flag = self._compression_flag(compression)
        compressor = zstd.ZstdCompressor(level=3) if compression_flag == 2 else None
        with open(output, "wb") as handle:
            self._write_header(
                handle=handle,
                n_variants=n_variants,
                n_samples=n_samples,
                samples=[str(sample) for sample in samples],
                compression_flag=compression_flag,
                metadata=metadata,
            )
            for idx in range(n_variants):
                variant_probabilities = np.ascontiguousarray(probabilities[idx], dtype=np.float64)
                alleles = self._alleles(variants_ref[idx], variants_alt[idx])
                variant_probabilities = self._trim_trailing_nan_probability_columns(variant_probabilities)
                variant_phased = self._variant_phased(variant_probabilities, alleles, phased)
                ploidy = self._common_case_ploidy(variant_probabilities, alleles, variant_phased)
                if ploidy is None:
                    ploidy = self._infer_ploidy(variant_probabilities, alleles, variant_phased)
                self._write_variant(
                    handle=handle,
                    varid=str(variants_id[idx]),
                    rsid=str(variants_id[idx]),
                    chrom=str(variants_chrom[idx]),
                    pos=int(variants_pos[idx]),
                    alleles=alleles,
                    probabilities=variant_probabilities,
                    ploidy=ploidy,
                    phased=variant_phased,
                    bit_depth=int(bit_depth),
                    compression_flag=compression_flag,
                    zstd_compressor=compressor,
                )

    @staticmethod
    def _compression_flag(compression: Optional[str]) -> int:
        if compression is None:
            return 0
        if compression == "zlib":
            return 1
        if compression == "zstd":
            return 2
        raise ValueError(f"compression type {compression!r} is not one of None, 'zlib', or 'zstd'.")

    @staticmethod
    def _write_header(
        *,
        handle,
        n_variants: int,
        n_samples: int,
        samples: list[str],
        compression_flag: int,
        metadata: Optional[str],
    ) -> None:
        metadata_bytes = b"" if metadata is None else str(metadata).encode("utf-8")
        header_len = 20 + len(metadata_bytes)
        flags = compression_flag | (2 << 2) | (1 << 31)

        sample_parts = []
        for sample in samples:
            sample_bytes = sample.encode("utf-8")
            if len(sample_bytes) > np.iinfo(np.uint16).max:
                raise ValueError("BGEN sample IDs cannot exceed uint16 length.")
            sample_parts.append(_U16.pack(len(sample_bytes)) + sample_bytes)
        sample_payload = b"".join(sample_parts)
        sample_block = _U32.pack(8 + len(sample_payload)) + _U32.pack(n_samples) + sample_payload

        after_offset = (
            _U32.pack(header_len)
            + _U32.pack(n_variants)
            + _U32.pack(n_samples)
            + b"bgen"
            + metadata_bytes
            + _U32.pack(flags)
            + sample_block
        )
        handle.write(_U32.pack(len(after_offset)))
        handle.write(after_offset)

    @classmethod
    def _write_variant(
        cls,
        *,
        handle,
        varid: str,
        rsid: str,
        chrom: str,
        pos: int,
        alleles: list[str],
        probabilities: np.ndarray,
        ploidy: Union[int, np.ndarray],
        phased: bool,
        bit_depth: int,
        compression_flag: int,
        zstd_compressor: Optional[zstd.ZstdCompressor],
    ) -> None:
        if _native_bgen is None:
            raise ImportError("Native BGEN support requires the compiled snputils.snp.io._bgen extension.")
        probabilities = np.ascontiguousarray(probabilities, dtype=np.float64)
        n_samples, width = probabilities.shape
        if len(alleles) > np.iinfo(np.uint16).max:
            raise ValueError("BGEN allele count cannot exceed uint16 range.")
        min_ploidy, max_ploidy, ploidy_array = cls._normalise_ploidy(ploidy, n_samples)
        encoded = _native_bgen.encode_layout2(
            probabilities,
            n_samples,
            width,
            len(alleles),
            min_ploidy,
            max_ploidy,
            bool(phased),
            int(bit_depth),
            ploidy_array if ploidy_array is not None else None,
        )

        if compression_flag == 0:
            genotype_block = _U32.pack(len(encoded)) + encoded
        elif compression_flag == 1:
            compressed = zlib.compress(encoded, level=6)
            genotype_block = _U32.pack(len(compressed) + 4) + _U32.pack(len(encoded)) + compressed
        else:
            if zstd_compressor is None:  # pragma: no cover - guarded by caller
                raise ValueError("Missing zstd compressor.")
            compressed = zstd_compressor.compress(encoded)
            genotype_block = _U32.pack(len(compressed) + 4) + _U32.pack(len(encoded)) + compressed

        handle.write(cls._variant_header(varid, rsid, chrom, pos, alleles))
        handle.write(genotype_block)

    @staticmethod
    def _normalise_ploidy(ploidy: Union[int, np.ndarray], n_samples: int) -> tuple[int, int, Optional[np.ndarray]]:
        if isinstance(ploidy, (int, np.integer)):
            value = int(ploidy)
            if value < 0 or value > 63:
                raise ValueError("BGEN ploidy must be in the 0-63 range.")
            return value, value, None
        ploidy_array = np.ascontiguousarray(ploidy, dtype=np.uint8)
        if ploidy_array.shape != (n_samples,):
            raise ValueError("BGEN ploidy array length must match sample count.")
        if np.any(ploidy_array > 63):
            raise ValueError("BGEN ploidy values must be in the 0-63 range.")
        return int(ploidy_array.min()), int(ploidy_array.max()), ploidy_array

    @staticmethod
    def _variant_header(varid: str, rsid: str, chrom: str, pos: int, alleles: list[str]) -> bytes:
        if pos < 0 or pos > np.iinfo(np.uint32).max:
            raise ValueError("BGEN variant positions must be in the uint32 range.")
        parts = [
            BGENWriter._u16_text(varid, "variant ID"),
            BGENWriter._u16_text(rsid, "RSID"),
            BGENWriter._u16_text(chrom, "chromosome"),
            _U32.pack(int(pos)),
            _U16.pack(len(alleles)),
        ]
        for allele in alleles:
            allele_bytes = str(allele).encode("utf-8")
            if len(allele_bytes) > np.iinfo(np.uint32).max:
                raise ValueError("BGEN allele strings cannot exceed uint32 length.")
            parts.append(_U32.pack(len(allele_bytes)) + allele_bytes)
        return b"".join(parts)

    @staticmethod
    def _u16_text(value: str, field: str) -> bytes:
        value_bytes = str(value).encode("utf-8")
        if len(value_bytes) > np.iinfo(np.uint16).max:
            raise ValueError(f"BGEN {field} cannot exceed uint16 length.")
        return _U16.pack(len(value_bytes)) + value_bytes

    def _probabilities(self, phased: Optional[bool]) -> np.ndarray:
        if self.__snpobj.calldata_gp is not None:
            return np.asarray(self.__snpobj.calldata_gp, dtype=np.float64)
        if self.__snpobj.genotypes is None:
            raise ValueError("BGENWriter requires either `calldata_gp` or `genotypes`.")
        return self._hardcalls_to_probabilities(np.asarray(self.__snpobj.genotypes), phased=phased)

    @staticmethod
    def _hardcalls_to_probabilities(genotypes: np.ndarray, phased: Optional[bool]) -> np.ndarray:
        validate_hardcall_values(
            genotypes,
            allowed_values=(0, 1) if genotypes.ndim == 3 else (0, 1, 2),
        )
        if genotypes.ndim == 3 and phased:
            n_variants, n_samples, n_alleles = genotypes.shape
            if n_alleles != 2:
                raise ValueError("Phased BGEN export expects genotype shape (n_snps, n_samples, 2).")
            probabilities = np.zeros((n_variants, n_samples, 4), dtype=np.float64)
            missing = np.any(genotypes < 0, axis=2)
            probabilities[:, :, 0] = genotypes[:, :, 0] == 0
            probabilities[:, :, 1] = genotypes[:, :, 0] == 1
            probabilities[:, :, 2] = genotypes[:, :, 1] == 0
            probabilities[:, :, 3] = genotypes[:, :, 1] == 1
            probabilities[missing, :] = np.nan
            return probabilities

        if genotypes.ndim == 3:
            dosage = genotypes.sum(axis=2, dtype=np.int16)
            missing = np.any(genotypes < 0, axis=2)
        elif genotypes.ndim == 2:
            dosage = genotypes
            missing = genotypes < 0
        else:
            raise ValueError("`genotypes` must be a 2D hard-call or 3D allele array.")

        n_variants, n_samples = dosage.shape
        probabilities = np.zeros((n_variants, n_samples, 3), dtype=np.float64)
        for genotype_value in (0, 1, 2):
            probabilities[:, :, genotype_value] = dosage == genotype_value
        probabilities[missing, :] = np.nan
        return probabilities

    def _samples(self, n_samples: int) -> np.ndarray:
        if self.__snpobj.samples is None:
            return np.asarray([str(i) for i in range(n_samples)], dtype=object)
        samples = np.asarray(self.__snpobj.samples, dtype=object)
        if samples.shape[0] != n_samples:
            raise ValueError(f"samples length ({samples.shape[0]}) must match genotype sample count ({n_samples}).")
        return samples

    def _required_variant_column(self, attr: str, n_variants: int) -> np.ndarray:
        values = getattr(self.__snpobj, attr)
        if values is None:
            raise ValueError(f"BGENWriter requires `{attr}`.")
        arr = np.asarray(values)
        if arr.shape[0] != n_variants:
            raise ValueError(f"{attr} length ({arr.shape[0]}) must match number of variants ({n_variants}).")
        return arr

    def _variant_ids(self, n_variants: int) -> np.ndarray:
        if self.__snpobj.variants_id is None:
            return np.asarray([f"variant_{idx}" for idx in range(n_variants)], dtype=object)
        arr = np.asarray(self.__snpobj.variants_id, dtype=object)
        if arr.shape[0] != n_variants:
            raise ValueError(f"variants_id length ({arr.shape[0]}) must match number of variants ({n_variants}).")
        return arr

    @staticmethod
    def _variant_phased(probabilities: np.ndarray, alleles: list[str], phased: Optional[bool]) -> bool:
        if phased is not None:
            return bool(phased)

        n_alleles = len(alleles)
        if n_alleles == 2:
            if probabilities.shape[1] == 3:
                return False
            if probabilities.shape[1] == 4 and not np.isnan(probabilities[:, 3]).all():
                return True

        counts = BGENWriter._nonmissing_probability_counts(probabilities)
        counts = counts[counts > 0]
        if counts.size == 0:
            return probabilities.shape[1] == 4 and n_alleles == 2

        phased_ploidy = [BGENWriter._phased_ploidy_from_width(int(count), n_alleles) for count in counts]
        unphased_ploidy = [BGENWriter._unphased_ploidy_from_width(int(count), n_alleles) for count in counts]
        phased_possible = all(ploidy is not None for ploidy in phased_ploidy)
        unphased_possible = all(ploidy is not None for ploidy in unphased_ploidy)

        if phased_possible and not unphased_possible:
            return True
        if unphased_possible and not phased_possible:
            return False

        if phased_possible and unphased_possible:
            # For phased data, each haplotype contributes one probability per allele
            # and each haplotype's allele probabilities sum to one. Unphased rows
            # instead sum to one across the full genotype distribution.
            if BGENWriter._looks_phased(probabilities, n_alleles, phased_ploidy):
                return True

        if probabilities.shape[1] == 4 and n_alleles == 2 and not np.isnan(probabilities[:, 3]).all():
            return True
        return False

    @staticmethod
    def _common_case_ploidy(
        probabilities: np.ndarray,
        alleles: list[str],
        phased: bool,
    ) -> Optional[int]:
        if len(alleles) != 2:
            return None

        expected_width = 4 if phased else 3
        if probabilities.shape[1] != expected_width:
            return None

        finite = np.isfinite(probabilities)
        if finite.all():
            return 2

        row_missing = ~finite.any(axis=1)
        if np.all(row_missing | finite.all(axis=1)):
            return 2
        return None

    @staticmethod
    def _trim_trailing_nan_probability_columns(probabilities: np.ndarray) -> np.ndarray:
        keep = probabilities.shape[1]
        while keep > 1 and np.isnan(probabilities[:, keep - 1]).all():
            keep -= 1
        return probabilities[:, :keep]

    @staticmethod
    def _nonmissing_probability_counts(probabilities: np.ndarray) -> np.ndarray:
        finite = np.isfinite(probabilities)
        counts = finite.sum(axis=1)
        for sample_idx, count in enumerate(counts):
            if count == 0:
                continue
            if finite[sample_idx, :count].all() and not finite[sample_idx, count:].any():
                continue
            raise ValueError(
                "BGEN probability rows may only contain NaN values as all-missing rows "
                "or as trailing padding for lower-ploidy samples."
            )
        return counts

    @staticmethod
    def _phased_ploidy_from_width(width: int, n_alleles: int) -> Optional[int]:
        if n_alleles <= 0 or width <= 0 or width % n_alleles != 0:
            return None
        return width // n_alleles

    @staticmethod
    def _unphased_ploidy_from_width(width: int, n_alleles: int) -> Optional[int]:
        if width <= 0 or n_alleles <= 0:
            return None
        for ploidy in range(0, 64):
            if math.comb(ploidy + n_alleles - 1, n_alleles - 1) == width:
                return ploidy
        return None

    @staticmethod
    def _looks_phased(
        probabilities: np.ndarray,
        n_alleles: int,
        ploidies: list[Optional[int]],
    ) -> bool:
        for sample_probabilities, ploidy in zip(probabilities, ploidies):
            if ploidy is None:
                return False
            if np.isnan(sample_probabilities).all():
                continue
            width = ploidy * n_alleles
            haplotypes = sample_probabilities[:width].reshape(ploidy, n_alleles)
            if not np.allclose(haplotypes.sum(axis=1), 1.0, atol=1e-4, rtol=0):
                return False
        return True

    @staticmethod
    def _infer_ploidy(probabilities: np.ndarray, alleles: list[str], phased: bool) -> Union[int, np.ndarray]:
        n_alleles = len(alleles)
        counts = BGENWriter._nonmissing_probability_counts(probabilities)
        ploidies = np.empty(probabilities.shape[0], dtype=np.uint8)
        inferred = []

        for count in counts:
            if count == 0:
                inferred.append(None)
                continue
            if phased:
                ploidy = BGENWriter._phased_ploidy_from_width(int(count), n_alleles)
            else:
                ploidy = BGENWriter._unphased_ploidy_from_width(int(count), n_alleles)
            if ploidy is None:
                mode = "phased" if phased else "unphased"
                if phased and n_alleles == 2:
                    raise ValueError(
                        f"Biallelic diploid phased BGEN probabilities require 4 columns; "
                        f"got {int(count)}."
                    )
                raise ValueError(
                    f"Cannot infer {mode} BGEN ploidy from {count} probability columns "
                    f"and {n_alleles} alleles."
                )
            inferred.append(ploidy)

        fallback_ploidy = max((ploidy for ploidy in inferred if ploidy is not None), default=2)
        for idx, ploidy in enumerate(inferred):
            ploidies[idx] = fallback_ploidy if ploidy is None else ploidy

        if np.all(ploidies == ploidies[0]):
            return int(ploidies[0])
        return ploidies

    @staticmethod
    def _alleles(ref: Union[str, bytes], alt: Union[str, bytes]) -> list[str]:
        ref_text = str(ref)
        alt_text = str(alt)
        if not ref_text or ref_text == "." or not alt_text or alt_text == ".":
            raise ValueError("BGENWriter requires non-missing REF and ALT alleles.")
        return [ref_text] + alt_text.split(",")
