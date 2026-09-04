from __future__ import annotations

import logging
import operator
import struct
import warnings
import zlib
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import zstandard as zstd

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader

log = logging.getLogger(__name__)

try:
    _native_bgen = import_module("snputils.snp.io._bgen")
except ImportError:  # pragma: no cover - exercised only when the extension is unavailable
    _native_bgen = None

_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()


def _native_zstd_unavailable(exc: BaseException) -> bool:
    return isinstance(exc, NotImplementedError) and "shared libzstd library" in str(exc)


def _warn_zstd_thread_fallback() -> None:
    warnings.warn(
        "Native threaded BGEN zstd decompression is unavailable because libzstd "
        "could not be loaded; falling back to single-threaded Python zstandard decompression.",
        RuntimeWarning,
        stacklevel=3,
    )


def _resolve_compressed_path(sample_path: Union[str, bytes, Path]) -> str:
    import os
    sample_path_str = str(Path(sample_path))
    if os.path.exists(sample_path_str):
        return sample_path_str
    for comp_ext in (".zst", ".gz"):
        candidate = sample_path_str + comp_ext
        if os.path.exists(candidate):
            return candidate
    return sample_path_str


def _open_textfile(filename: str, mode: str = "rt"):
    if filename.endswith(".zst"):
        import zstandard as zstd
        return zstd.open(filename, mode, encoding="utf-8") if "t" in mode else zstd.open(filename, mode)
    elif filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode, encoding="utf-8") if "t" in mode else gzip.open(filename, mode)
    return open(filename, mode, encoding="utf-8") if "t" in mode else open(filename, mode)


@dataclass(frozen=True)
class _BGENHeader:
    first_variant_offset: int
    n_variants: int
    n_samples: int
    compression: int
    layout: int
    has_sample_ids: bool
    metadata: bytes


@dataclass(frozen=True)
class _BGENRecord:
    index: int
    varid: str
    rsid: str
    chrom: str
    pos: int
    alleles: tuple[str, ...]
    probabilities: Optional[np.ndarray]
    phased: Optional[bool]


def _as_field_list(fields: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [fields]
    return list(fields)


def _variant_identifier(varid: str, rsid: str) -> str:
    return varid if varid and varid != "." else rsid


def _read_exact(handle, size: int, context: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Malformed BGEN file: {context} is truncated.")
    return data


def _read_u16(handle, context: str) -> int:
    return _U16.unpack(_read_exact(handle, 2, context))[0]


def _read_u32(handle, context: str) -> int:
    return _U32.unpack(_read_exact(handle, 4, context))[0]


def _read_len_prefixed_text(handle, len_size: int, context: str) -> str:
    if len_size == 2:
        size = _read_u16(handle, f"{context} length")
    elif len_size == 4:
        size = _read_u32(handle, f"{context} length")
    else:  # pragma: no cover - internal misuse guard
        raise ValueError("BGEN string length size must be 2 or 4 bytes.")
    if size == 0:
        return ""
    return _read_exact(handle, size, context).decode("utf-8")


def _read_sample_file(sample_path: Union[str, bytes, Path], n_samples: int) -> np.ndarray:
    samples: list[str] = []
    with _open_textfile(str(sample_path), "rt") as handle:
        next(handle, None)
        next(handle, None)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(line.split()[0])
    if len(samples) != n_samples:
        raise ValueError("BGEN sample file contains an inconsistent number of samples.")
    return np.asarray(samples, dtype=object)


class _DirectBGENFile:
    def __init__(self, filename: Union[str, Path], sample_path: Optional[Union[str, bytes]] = None):
        self.filename = Path(filename)
        self.sample_path = sample_path
        self.handle = None
        self.header: Optional[_BGENHeader] = None
        self.samples: Optional[np.ndarray] = None

    def __enter__(self) -> "_DirectBGENFile":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def open(self) -> None:
        self.handle = open(self.filename, "rb")
        self.header = self._read_header()
        if self.header.layout != 2:
            raise NotImplementedError("Native BGENReader currently supports BGEN layout 2 files.")
        if self.header.compression not in (0, 1, 2):
            raise ValueError("Unsupported BGEN compression flag.")
        if self.header.has_sample_ids:
            self.samples = self._read_embedded_samples(self.header.n_samples)
        elif self.sample_path:
            resolved_sample_path = _resolve_compressed_path(self.sample_path)
            self.samples = _read_sample_file(resolved_sample_path, self.header.n_samples)
        else:
            self.samples = np.asarray([str(idx) for idx in range(self.header.n_samples)], dtype=object)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.handle = None

    def _read_header(self) -> _BGENHeader:
        assert self.handle is not None
        data = _read_exact(self.handle, 20, "header")
        offset, header_length, n_variants, n_samples = struct.unpack("<IIII", data[:16])
        magic = data[16:20]
        if magic not in (b"bgen", b"\x00\x00\x00\x00"):
            raise ValueError("File does not appear to be a BGEN file.")
        if header_length < 20:
            raise ValueError("Malformed BGEN file: header length is too small.")
        metadata = _read_exact(self.handle, header_length - 20, "free data") if header_length > 20 else b""
        flags = _read_u32(self.handle, "flags")
        compression = flags & 0b11
        layout = (flags >> 2) & 0b1111
        has_sample_ids = bool(flags & (1 << 31))
        return _BGENHeader(
            first_variant_offset=offset + 4,
            n_variants=n_variants,
            n_samples=n_samples,
            compression=compression,
            layout=layout,
            has_sample_ids=has_sample_ids,
            metadata=metadata,
        )

    def _read_embedded_samples(self, n_samples: int) -> np.ndarray:
        assert self.handle is not None
        sample_block_len = _read_u32(self.handle, "sample block length")
        sample_count = _read_u32(self.handle, "sample count")
        if sample_count != n_samples:
            raise ValueError("BGEN sample block contains an inconsistent number of samples.")
        samples = []
        bytes_read = 8
        for _ in range(n_samples):
            sample_len = _read_u16(self.handle, "sample ID length")
            sample = _read_exact(self.handle, sample_len, "sample ID").decode("utf-8")
            samples.append(sample)
            bytes_read += 2 + sample_len
        if bytes_read != sample_block_len:
            raise ValueError("Malformed BGEN file: sample block length does not match its contents.")
        return np.asarray(samples, dtype=object)

    def records(self, read_probabilities: bool = True):
        assert self.handle is not None
        assert self.header is not None
        self.handle.seek(self.header.first_variant_offset)
        for idx in range(self.header.n_variants):
            yield self._read_record(idx, read_probabilities)

    def _read_record(self, index: int, read_probabilities: bool) -> _BGENRecord:
        assert self.handle is not None
        assert self.header is not None

        varid = _read_len_prefixed_text(self.handle, 2, "variant ID")
        rsid = _read_len_prefixed_text(self.handle, 2, "RSID")
        chrom = _read_len_prefixed_text(self.handle, 2, "chromosome")
        pos = _read_u32(self.handle, "position")
        n_alleles = _read_u16(self.handle, "allele count")
        alleles = tuple(_read_len_prefixed_text(self.handle, 4, "allele") for _ in range(n_alleles))
        block_len = _read_u32(self.handle, "genotype block length")
        block = _read_exact(self.handle, block_len, "genotype block")

        probabilities = None
        phased = None
        if read_probabilities:
            probabilities, phased = self._decode_probabilities(block, n_alleles)
        return _BGENRecord(
            index=index,
            varid=varid,
            rsid=rsid,
            chrom=chrom,
            pos=pos,
            alleles=alleles,
            probabilities=probabilities,
            phased=phased,
        )

    def _decode_probabilities(self, block: bytes, n_alleles: int) -> tuple[np.ndarray, bool]:
        assert self.header is not None
        if _native_bgen is None:
            raise ImportError("Native BGEN support requires the compiled snputils.snp.io._bgen extension.")
        if self.header.compression == 0:
            payload = block
        else:
            if len(block) < 4:
                raise ValueError("Malformed BGEN genotype block: compressed length field is truncated.")
            expected_len = _U32.unpack(block[:4])[0]
            compressed = block[4:]
            if self.header.compression == 1:
                payload = zlib.decompress(compressed)
            else:
                payload = _ZSTD_DECOMPRESSOR.decompress(compressed, max_output_size=expected_len)
            if len(payload) != expected_len:
                raise ValueError("BGEN genotype block decompressed to the wrong size.")

        buffer, n_samples, width, phased, _bit_depth = _native_bgen.decode_layout2(
            payload,
            self.header.n_samples,
            n_alleles,
        )
        probabilities = np.frombuffer(buffer, dtype=np.float32).reshape(n_samples, width)
        return probabilities, bool(phased)


@SNPBaseReader.register
class BGENReader(SNPBaseReader):
    def read(
        self,
        fields: Optional[Union[str, Sequence[str]]] = None,
        exclude_fields: Optional[Union[str, Sequence[str]]] = None,
        sample_path: Optional[Union[str, bytes]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        sample_idxs: Optional[Sequence[int]] = None,
        variant_ids: Optional[Sequence[str]] = None,
        variant_idxs: Optional[Sequence[int]] = None,
        genotype_mode: Optional[str] = None,
        threads: int = 1,
    ) -> SNPObject:
        """
        Read a BGEN file into a SNPObject.

        Args:
            fields: Fields to include. Available fields are ``GP``, ``GT``, ``IID``,
                ``REF``, ``ALT``, ``#CHROM``, ``ID``, and ``POS``. ``GT`` requires
                ``genotype_mode="phased"``.
            exclude_fields: Fields to exclude from the returned SNPObject.
            genotype_mode: By default, preserve BGEN genotype probabilities in
                ``calldata_gp``. ``"dosage"`` returns expected biallelic alternate-
                allele counts with shape ``(n_variants, n_samples)``. ``"phased"``
                hard-calls each haplotype from phased genotype probabilities and
                returns a diploid allele array with shape ``(n_variants, n_samples, 2)``.
                Unphased or non-diploid records are rejected in phased mode.
            sample_path: Optional Oxford ``.sample`` file for BGEN files without
                embedded sample identifiers.
            sample_ids: Sample IDs to read. If None and sample_idxs is None, all samples are read.
            sample_idxs: Sample indices to read. If None and sample_ids is None, all samples are read.
            variant_ids: Variant IDs to read. Matches BGEN varid, rsid, or ``chrom:pos``.
            variant_idxs: Variant indices to read. If None and variant_ids is None, all variants are read.
            threads: Native decoder threads for unfiltered dosage reads or
                genotype-probability reads requesting only ``fields=["GP"]``.
                If native zstd support is unavailable, zstd reads warn and fall
                back to single-threaded Python decompression.
                The default is 1.

        Returns:
            SNPObject: A SNPObject with genotype probabilities in ``calldata_gp`` by
            default, or dosages/phased hard calls in ``genotypes`` when requested.
            Mixed probability widths are padded with NaN columns in probability mode.
        """
        if genotype_mode is not None:
            genotype_mode = str(genotype_mode).strip().lower()
            if genotype_mode not in {"dosage", "phased"}:
                raise ValueError("BGENReader genotype_mode must be 'dosage' or 'phased' when provided.")
        try:
            threads = operator.index(threads)
        except TypeError as exc:
            raise TypeError("BGENReader threads must be an integer.") from exc
        if threads < 1:
            raise ValueError("BGENReader threads must be at least 1.")
        if threads != 1 and genotype_mode == "phased":
            raise ValueError(
                "BGENReader threads are currently supported for dosage and "
                "full-probability output, not phased hard calls."
            )

        if sample_idxs is not None and sample_ids is not None:
            raise ValueError("Only one of sample_idxs and sample_ids can be specified.")
        if variant_idxs is not None and variant_ids is not None:
            raise ValueError("Only one of variant_idxs and variant_ids can be specified.")

        fields_list = _as_field_list(fields) or ["GP", "IID", "REF", "ALT", "#CHROM", "ID", "POS"]
        exclude = set(_as_field_list(exclude_fields) or [])
        fields_set = {field for field in fields_list if field not in exclude}

        if "GT" in fields_set and genotype_mode != "phased":
            raise NotImplementedError(
                "BGENReader does not hard-call GT by default; request "
                "genotype_mode='phased' to hard-call phased probabilities."
            )
        if genotype_mode == "phased":
            fields_set.discard("GT")
            fields_set.add("GP")
        elif genotype_mode == "dosage":
            fields_set.discard("GP")

        native_bulk_gp = genotype_mode is None and self._can_use_native_bulk_gp(
            fields_set=fields_set,
            sample_path=sample_path,
            sample_ids=sample_ids,
            sample_idxs=sample_idxs,
            variant_ids=variant_ids,
            variant_idxs=variant_idxs,
        )
        if native_bulk_gp:
            try:
                calldata_gp = self._read_native_bulk_gp(threads=threads)
                return SNPObject(genotypes=None, calldata_gp=calldata_gp)
            except (NotImplementedError, ValueError) as exc:
                zstd_fallback = _native_zstd_unavailable(exc)
                if threads != 1 and not zstd_fallback:
                    raise
                if threads != 1:
                    _warn_zstd_thread_fallback()
                fallback_messages = (
                    "uniform probability width",
                    "larger than the allocated output width",
                )
                if (
                    isinstance(exc, ValueError)
                    and not any(message in str(exc) for message in fallback_messages)
                ):
                    raise
                log.debug("Falling back to the general BGEN reader.", exc_info=True)
        elif threads != 1 and genotype_mode is None:
            raise ValueError(
                "Multithreaded BGEN probability reading currently requires "
                "fields=['GP'] with all samples and variants."
            )

        log.info("Reading %s", self.filename)
        with _DirectBGENFile(self.filename, sample_path=sample_path) as bfile:
            assert bfile.header is not None
            assert bfile.samples is not None
            sample_indices = self._resolve_sample_indices(bfile.samples, sample_ids, sample_idxs)
            requested_variant_idxs = self._normalize_variant_indices(
                bfile.header.n_variants,
                variant_idxs,
            )
            records = self._load_records(
                bfile=bfile,
                fields_set=fields_set,
                sample_indices=sample_indices,
                variant_ids=variant_ids,
                variant_idxs=requested_variant_idxs,
            )
            samples = bfile.samples[sample_indices] if "IID" in fields_set else None

        metadata, calldata_gp = self._records_to_arrays(records, sample_indices, fields_set)
        genotypes = None
        if genotype_mode == "phased":
            genotypes = self._records_to_phased_hardcalls(records, len(sample_indices))
            calldata_gp = None
        elif genotype_mode == "dosage":
            genotypes = self.read_dosage(
                sample_path=sample_path,
                sample_ids=sample_ids,
                sample_idxs=sample_idxs,
                variant_ids=variant_ids,
                variant_idxs=variant_idxs,
                threads=threads,
            )
        return SNPObject(
            genotypes=genotypes,
            calldata_gp=calldata_gp,
            samples=samples,
            variants_ref=metadata["variants_ref"],
            variants_alt=metadata["variants_alt"],
            variants_chrom=metadata["variants_chrom"],
            variants_id=metadata["variants_id"],
            variants_pos=metadata["variants_pos"],
        )

    def read_dosage(
        self,
        sample_path: Optional[Union[str, bytes]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        sample_idxs: Optional[Sequence[int]] = None,
        variant_ids: Optional[Sequence[str]] = None,
        variant_idxs: Optional[Sequence[int]] = None,
        threads: int = 1,
    ) -> np.ndarray:
        """
        Read biallelic BGEN alternate-allele dosages as a ``float32`` array.

        The all-samples/all-variants case uses a native streaming decoder that avoids
        materializing genotype probabilities. Filtered reads fall back to the general
        probability reader and convert from ``calldata_gp``. Set ``threads`` above
        1 to decode independent variant blocks in parallel for an unfiltered read.
        If native zstd support is unavailable, zstd reads warn and fall back to
        single-threaded Python decompression.
        """
        try:
            threads = operator.index(threads)
        except TypeError as exc:
            raise TypeError("BGENReader threads must be an integer.") from exc
        if threads < 1:
            raise ValueError("BGENReader threads must be at least 1.")
        filtered = any(
            value is not None
            for value in (sample_ids, sample_idxs, variant_ids, variant_idxs)
        )
        if threads != 1 and filtered:
            raise ValueError("Multithreaded BGEN dosage reading currently requires all samples and variants.")
        python_fallback = False
        if (
            _native_bgen is not None
            and sample_ids is None
            and sample_idxs is None
            and variant_ids is None
            and variant_idxs is None
        ):
            try:
                buffer, n_variants, n_samples = _native_bgen.read_file_dosage(
                    str(self.filename),
                    threads,
                )
                if isinstance(buffer, np.ndarray):
                    return buffer.astype(np.float32, copy=False).reshape(n_variants, n_samples)
                return np.frombuffer(buffer, dtype=np.float32).reshape(n_variants, n_samples)
            except NotImplementedError as exc:
                zstd_fallback = _native_zstd_unavailable(exc)
                if threads != 1 and not zstd_fallback:
                    raise
                python_fallback = True
                if threads != 1:
                    _warn_zstd_thread_fallback()
                log.debug("Falling back to probability-based BGEN dosage reading.", exc_info=True)

        if threads != 1 and not python_fallback:
            raise ImportError("Multithreaded BGEN dosage reading requires the native BGEN extension.")

        snpobj = self.read(
            fields=["GP"],
            sample_path=sample_path,
            sample_ids=sample_ids,
            sample_idxs=sample_idxs,
            variant_ids=variant_ids,
            variant_idxs=variant_idxs,
        )
        return snpobj.dosage().astype(np.float32, copy=False)

    @staticmethod
    def _can_use_native_bulk_gp(
        *,
        fields_set: set[str],
        sample_path: Optional[Union[str, bytes]],
        sample_ids: Optional[Sequence[str]],
        sample_idxs: Optional[Sequence[int]],
        variant_ids: Optional[Sequence[str]],
        variant_idxs: Optional[Sequence[int]],
    ) -> bool:
        return (
            _native_bgen is not None
            and fields_set == {"GP"}
            and sample_path is None
            and sample_ids is None
            and sample_idxs is None
            and variant_ids is None
            and variant_idxs is None
        )

    def _read_native_bulk_gp(self, *, threads: int = 1) -> np.ndarray:
        assert _native_bgen is not None
        buffer, n_variants, n_samples, width = _native_bgen.read_file_probabilities(
            str(self.filename),
            threads,
        )
        if isinstance(buffer, np.ndarray):
            return buffer.astype(np.float32, copy=False).reshape(n_variants, n_samples, width)
        return np.frombuffer(buffer, dtype=np.float32).reshape(n_variants, n_samples, width)

    @staticmethod
    def _variant_metadata_template() -> dict[str, Optional[np.ndarray]]:
        return {
            "variants_ref": None,
            "variants_alt": None,
            "variants_chrom": None,
            "variants_id": None,
            "variants_pos": None,
        }

    @staticmethod
    def _normalize_variant_indices(
        n_variants: int,
        variant_idxs: Optional[Sequence[int]],
    ) -> Optional[np.ndarray]:
        if variant_idxs is None:
            return None
        idx = np.asarray(variant_idxs, dtype=int).ravel()
        if np.any((idx < -n_variants) | (idx >= n_variants)):
            raise ValueError("One or more variant indexes are out of bounds.")
        return np.mod(idx, n_variants)

    @classmethod
    def _load_records(
        cls,
        *,
        bfile: _DirectBGENFile,
        fields_set: set[str],
        sample_indices: np.ndarray,
        variant_ids: Optional[Sequence[str]],
        variant_idxs: Optional[np.ndarray],
    ) -> list[_BGENRecord]:
        read_gp = "GP" in fields_set
        by_index: dict[int, _BGENRecord] = {}
        selected: list[_BGENRecord] = []
        requested_ids = {str(value) for value in np.asarray(variant_ids, dtype=object).ravel()} if variant_ids is not None else None
        found_ids: set[str] = set()
        requested_idx_set = set(int(idx) for idx in variant_idxs) if variant_idxs is not None else None
        scan_limit = int(np.max(variant_idxs)) if variant_idxs is not None and variant_idxs.size else None

        for record in bfile.records(read_probabilities=read_gp):
            include = False
            aliases = {
                _variant_identifier(record.varid, record.rsid),
                record.rsid,
                f"{record.chrom}:{record.pos}",
            }
            if requested_idx_set is not None:
                include = record.index in requested_idx_set
            elif requested_ids is not None:
                include = bool(requested_ids.intersection(aliases))
                if include:
                    found_ids.update(aliases)
            else:
                include = True

            if include:
                if record.probabilities is not None:
                    record = _BGENRecord(
                        index=record.index,
                        varid=record.varid,
                        rsid=record.rsid,
                        chrom=record.chrom,
                        pos=record.pos,
                        alleles=record.alleles,
                        probabilities=record.probabilities[sample_indices, :],
                        phased=record.phased,
                    )
                if requested_idx_set is not None:
                    by_index[record.index] = record
                else:
                    selected.append(record)

            if scan_limit is not None and record.index >= scan_limit:
                break

        if requested_idx_set is not None:
            missing = [int(idx) for idx in variant_idxs if int(idx) not in by_index]
            if missing:
                raise ValueError(f"The following specified variant indexes were not found: {missing}")
            return [by_index[int(idx)] for idx in variant_idxs]

        if requested_ids is not None:
            missing = sorted(requested_ids - found_ids)
            if missing:
                raise ValueError(f"The following specified variants were not found: {missing}")
        return selected

    @staticmethod
    def _records_to_phased_hardcalls(
        records: Sequence[_BGENRecord],
        n_samples: int,
    ) -> np.ndarray:
        genotypes = np.empty((len(records), n_samples, 2), dtype=np.int8)

        for variant_index, record in enumerate(records):
            variant_id = _variant_identifier(record.varid, record.rsid) or f"index {record.index}"
            if record.phased is not True:
                raise ValueError(
                    "BGEN genotype_mode='phased' requires phased probabilities; "
                    f"variant {variant_id!r} is unphased."
                )
            if record.probabilities is None:
                raise ValueError(f"BGEN probabilities were not loaded for variant {variant_id!r}.")

            n_alleles = len(record.alleles)
            if n_alleles < 2 or n_alleles > np.iinfo(np.int8).max:
                raise ValueError(
                    f"Variant {variant_id!r} has unsupported allele count {n_alleles}."
                )

            probabilities = np.asarray(record.probabilities, dtype=np.float32)
            expected_width = 2 * n_alleles
            if probabilities.shape != (n_samples, expected_width):
                raise ValueError(
                    "BGEN genotype_mode='phased' requires diploid phased probabilities; "
                    f"variant {variant_id!r} has probability width {probabilities.shape[1]}, "
                    f"expected {expected_width}."
                )

            haplotype_probabilities = probabilities.reshape(n_samples, 2, n_alleles)
            called = np.all(np.isfinite(haplotype_probabilities), axis=2)
            partial_ploidy = np.sum(called, axis=1) == 1
            if np.any(partial_ploidy):
                sample_index = int(np.flatnonzero(partial_ploidy)[0])
                raise ValueError(
                    "BGEN genotype_mode='phased' requires diploid phased probabilities; "
                    f"variant {variant_id!r} has a non-diploid call for sample index {sample_index}."
                )
            safe_probabilities = np.where(
                np.isfinite(haplotype_probabilities),
                haplotype_probabilities,
                -np.inf,
            )
            variant_genotypes = np.argmax(safe_probabilities, axis=2).astype(np.int8)
            variant_genotypes[~called] = -1
            genotypes[variant_index] = variant_genotypes

        return genotypes

    @classmethod
    def _records_to_arrays(
        cls,
        records: Sequence[_BGENRecord],
        sample_indices: np.ndarray,
        fields_set: set[str],
    ) -> tuple[dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        metadata = cls._variant_metadata_template()
        if "REF" in fields_set:
            metadata["variants_ref"] = np.asarray(
                [record.alleles[0] if record.alleles else "" for record in records],
                dtype=object,
            )
        if "ALT" in fields_set:
            metadata["variants_alt"] = np.asarray(
                [",".join(record.alleles[1:]) if len(record.alleles) > 1 else "" for record in records],
                dtype=object,
            )
        if "#CHROM" in fields_set:
            metadata["variants_chrom"] = np.asarray([record.chrom for record in records], dtype=object)
        if "ID" in fields_set:
            metadata["variants_id"] = np.asarray(
                [_variant_identifier(record.varid, record.rsid) for record in records],
                dtype=object,
            )
        if "POS" in fields_set:
            metadata["variants_pos"] = np.asarray([record.pos for record in records], dtype=np.int64)

        calldata_gp = None
        if "GP" in fields_set:
            if not records:
                calldata_gp = np.empty((0, len(sample_indices), 0), dtype=np.float32)
            else:
                widths = [record.probabilities.shape[1] for record in records if record.probabilities is not None]
                max_width = max(widths, default=0)
                calldata_gp = np.full((len(records), len(sample_indices), max_width), np.nan, dtype=np.float32)
                for idx, record in enumerate(records):
                    if record.probabilities is None:
                        continue
                    width = record.probabilities.shape[1]
                    calldata_gp[idx, :, :width] = record.probabilities
        return metadata, calldata_gp

    @staticmethod
    def _resolve_sample_indices(
        file_samples: np.ndarray,
        sample_ids: Optional[Sequence[str]],
        sample_idxs: Optional[Sequence[int]],
    ) -> np.ndarray:
        if sample_idxs is not None:
            idx = np.asarray(sample_idxs, dtype=int).ravel()
            n_samples = len(file_samples)
            if np.any((idx < -n_samples) | (idx >= n_samples)):
                raise ValueError("One or more sample indexes are out of bounds.")
            return np.mod(idx, n_samples)

        if sample_ids is None:
            return np.arange(len(file_samples), dtype=int)

        requested = np.asarray(sample_ids, dtype=object).ravel()
        sample_lookup = {str(sample): i for i, sample in enumerate(file_samples)}
        missing = [str(sample) for sample in requested if str(sample) not in sample_lookup]
        if missing:
            raise ValueError(f"The following specified samples were not found: {missing}")
        return np.asarray([sample_lookup[str(sample)] for sample in requested], dtype=int)
