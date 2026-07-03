from __future__ import annotations

import logging
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from snputils.snp.genobj import SNPObject

try:
    from snputils.snp.io.write import _bcf as _native_bcf
except ImportError:  # pragma: no cover - exercised only when optional extension is unavailable
    _native_bcf = None

log = logging.getLogger(__name__)

_FLOAT_MISSING_BITS = np.uint32(0x7F800001)
_INFO_ITEM_RE = re.compile(r"^([^=;]+)(?:=(.*))?$")
_INT_RE = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class _InfoMeta:
    idx: int
    id: str
    number: str
    type: str


def _encode_typed_descriptor(n: int, type_code: int) -> bytes:
    if n < 0:
        raise ValueError("BCF typed value length cannot be negative.")
    if n < 15:
        return bytes([(n << 4) | type_code])
    if n <= 127:
        len_enc = b"\x11" + bytes([n])
    elif n <= 32767:
        len_enc = b"\x12" + struct.pack("<h", n)
    elif n <= 2147483647:
        len_enc = b"\x13" + struct.pack("<i", n)
    else:
        raise ValueError("BCF typed value length exceeds int32 range.")
    return bytes([(15 << 4) | type_code]) + len_enc


def _encode_typed_string(s: str) -> bytes:
    if not s:
        return b"\x07"
    s_bytes = str(s).encode("utf-8")
    return _encode_typed_descriptor(len(s_bytes), 7) + s_bytes


def _int_type_for_values(values: Sequence[Optional[int]]) -> tuple[int, int]:
    present = [int(value) for value in values if value is not None]
    if not present:
        return 1, 1
    min_val = min(present)
    max_val = max(present)
    if min_val >= -120 and max_val <= 127:
        return 1, 1
    if min_val >= -32760 and max_val <= 32767:
        return 2, 2
    return 3, 4


def _encode_typed_int_list(ints: Sequence[Optional[int]]) -> bytes:
    n = len(ints)
    if n == 0:
        return b"\x01"

    type_code, type_size = _int_type_for_values(ints)
    if type_size == 1:
        missing = 0x80
        val_bytes = bytes(missing if value is None else (int(value) & 0xFF) for value in ints)
    elif type_size == 2:
        missing = 0x8000
        val_bytes = b"".join(
            struct.pack("<H", missing) if value is None else struct.pack("<h", int(value))
            for value in ints
        )
    else:
        missing = 0x80000000
        val_bytes = b"".join(
            struct.pack("<I", missing) if value is None else struct.pack("<i", int(value))
            for value in ints
        )
    return _encode_typed_descriptor(n, type_code) + val_bytes


def _encode_typed_float_list(values: Sequence[Optional[float]]) -> bytes:
    encoded = bytearray(_encode_typed_descriptor(len(values), 5))
    for value in values:
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            encoded.extend(struct.pack("<I", int(_FLOAT_MISSING_BITS)))
        else:
            encoded.extend(struct.pack("<f", float(value)))
    return bytes(encoded)


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text == "." or text.lower() in {"nan", "none"}


def _normalise_text_array(
    values: Optional[Sequence[object]],
    n_variants: int,
    *,
    attr: str,
    default: str,
    required: bool = False,
) -> np.ndarray:
    if values is None:
        if required:
            raise ValueError(f"BCFWriter requires `{attr}`.")
        arr = np.full(n_variants, default, dtype=object)
    else:
        arr = np.asarray(values, dtype=object)
        if arr.shape[0] != n_variants:
            raise ValueError(f"{attr} length ({arr.shape[0]}) must match number of variants ({n_variants}).")

    out = []
    for value in arr:
        if _is_missing_value(value):
            out.append(default)
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def _normalise_positions(values: Optional[Sequence[object]], n_variants: int) -> np.ndarray:
    if values is None:
        raise ValueError("BCFWriter requires `variants_pos`.")
    arr = np.asarray(values)
    if arr.shape[0] != n_variants:
        raise ValueError(f"variants_pos length ({arr.shape[0]}) must match number of variants ({n_variants}).")
    positions = arr.astype(np.int64, copy=False)
    if np.any(positions < 1) or np.any(positions > np.iinfo(np.int32).max):
        raise ValueError("BCF variant positions must be in the 1-based int32 range.")
    return np.ascontiguousarray(positions, dtype=np.int64)


def _normalise_qual_bits(values: Optional[Sequence[object]], n_variants: int) -> np.ndarray:
    if values is None:
        return np.full(n_variants, _FLOAT_MISSING_BITS, dtype=np.uint32)
    arr = np.asarray(values, dtype=object)
    if arr.shape[0] != n_variants:
        raise ValueError(f"variants_qual length ({arr.shape[0]}) must match number of variants ({n_variants}).")
    qual = np.empty(n_variants, dtype=np.float32)
    missing = np.zeros(n_variants, dtype=bool)
    for idx, value in enumerate(arr):
        if _is_missing_value(value):
            missing[idx] = True
            qual[idx] = np.nan
        else:
            qual[idx] = np.float32(value)
    bits = qual.view(np.uint32).copy()
    bits[missing] = _FLOAT_MISSING_BITS
    return np.ascontiguousarray(bits, dtype=np.uint32)


def _missing_mask(values: np.ndarray, before: Union[int, float, str]) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        if isinstance(before, (float, np.floating)) and np.isnan(before):
            return np.isnan(values)
        if isinstance(before, (int, np.integer, float, np.floating)) and float(before) < 0:
            return values < 0
    return values == before


def _normalise_genotypes(
    genotypes: Optional[np.ndarray],
    *,
    rename_missing_values: bool,
    before: Union[int, float, str],
) -> tuple[Optional[np.ndarray], int, int, int]:
    if genotypes is None:
        return None, 0, 0, 1

    arr = np.asarray(genotypes)
    if arr.ndim not in (2, 3):
        raise ValueError("BCFWriter requires `genotypes` to be a 2D hard-call or 3D allele array.")

    if arr.dtype.kind in "iu":
        gt = arr.astype(np.int16, copy=False)
        missing = _missing_mask(gt, before) if rename_missing_values else gt < 0
    elif arr.dtype.kind == "f":
        gt_float = arr.astype(np.float64, copy=False)
        missing = np.isnan(gt_float)
        if rename_missing_values:
            missing |= _missing_mask(gt_float, before)
        nonmissing = ~missing
        if np.any(nonmissing & (gt_float != np.floor(gt_float))):
            raise ValueError("BCF genotypes must contain integer allele or dosage values.")
        gt = np.zeros(arr.shape, dtype=np.int16)
        gt[nonmissing] = gt_float[nonmissing].astype(np.int16)
    else:
        text = arr.astype(str)
        missing = np.isin(np.char.lower(np.char.strip(text)), ["", ".", "nan", "none"])
        if rename_missing_values:
            missing |= text == str(before)
        gt = np.zeros(arr.shape, dtype=np.int16)
        if np.any(~missing):
            gt[~missing] = text[~missing].astype(np.int16)

    if np.any(missing):
        gt = gt.copy()
        gt[missing] = -1

    if arr.ndim == 2:
        if np.any((gt >= 0) & ((gt < 0) | (gt > 2))):
            raise ValueError("2D BCF genotypes must contain hard-call dosages 0, 1, 2, or missing values.")
        return np.ascontiguousarray(gt, dtype=np.int16), 2, 2, 1

    gt_width = gt.shape[2]
    if gt_width not in (1, 2):
        raise ValueError("BCFWriter supports haploid or diploid GT fields.")
    if np.any(gt < -1):
        raise ValueError("BCF genotype allele values must be >= 0 or missing.")

    max_allele = int(gt[gt >= 0].max(initial=0))
    encoded_max = ((max_allele + 1) << 1) | 1
    if encoded_max <= 0x7F:
        gt_type_size = 1
    elif encoded_max <= 0x7FFF:
        gt_type_size = 2
    else:
        gt_type_size = 4
    return np.ascontiguousarray(gt, dtype=np.int16), 3, gt_width, gt_type_size


def _normalise_gp(calldata_gp: Optional[np.ndarray], n_variants: int) -> Optional[np.ndarray]:
    if calldata_gp is None:
        return None
    gp = np.asarray(calldata_gp, dtype=np.float32)
    if gp.ndim != 3:
        raise ValueError("BCF FORMAT/GP must have shape (n_variants, n_samples, n_probabilities).")
    if gp.shape[0] != n_variants:
        raise ValueError(f"calldata_gp length ({gp.shape[0]}) must match number of variants ({n_variants}).")
    if gp.shape[2] < 1:
        raise ValueError("BCF FORMAT/GP must contain at least one probability per sample.")
    return np.ascontiguousarray(gp, dtype=np.float32)


def _infer_sample_count(genotypes: Optional[np.ndarray], gp: Optional[np.ndarray], samples: Optional[np.ndarray]) -> int:
    n_samples: Optional[int] = None
    if genotypes is not None:
        n_samples = int(genotypes.shape[1])
    if gp is not None:
        if n_samples is not None and int(gp.shape[1]) != n_samples:
            raise ValueError("GT and GP sample counts do not match.")
        n_samples = int(gp.shape[1])
    if n_samples is None:
        n_samples = 0 if samples is None else len(samples)
    return n_samples


def _normalise_samples(samples: Optional[Sequence[object]], n_samples: int) -> np.ndarray:
    if n_samples == 0:
        return np.asarray([], dtype=object)
    if samples is None:
        return np.asarray([str(idx) for idx in range(n_samples)], dtype=object)
    arr = np.asarray(samples, dtype=object)
    if arr.shape[0] != n_samples:
        raise ValueError(f"samples length ({arr.shape[0]}) must match genotype sample count ({n_samples}).")
    return np.asarray([str(sample) for sample in arr], dtype=object)


def _normalise_filter_values(values: Optional[Sequence[object]], n_variants: int) -> np.ndarray:
    if values is None:
        return np.full(n_variants, "PASS", dtype=object)
    arr = np.asarray(values, dtype=object)
    if arr.shape[0] != n_variants:
        raise ValueError(f"variants_filter_pass length ({arr.shape[0]}) must match number of variants ({n_variants}).")
    if arr.dtype == np.bool_:
        return np.where(arr.astype(bool), "PASS", ".").astype(object)

    out = []
    for value in arr:
        if isinstance(value, (bool, np.bool_)):
            out.append("PASS" if bool(value) else ".")
        elif _is_missing_value(value):
            out.append(".")
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def _collect_filter_ids(filter_values: np.ndarray, start_idx: int) -> tuple[OrderedDict[str, int], int]:
    filter_ids: OrderedDict[str, int] = OrderedDict()
    next_idx = start_idx
    for value in filter_values:
        text = str(value)
        if text in {"", ".", "PASS"}:
            continue
        for label in text.split(";"):
            if label and label not in filter_ids and label != "PASS":
                filter_ids[label] = next_idx
                next_idx += 1
    return filter_ids, next_idx


def _build_filter_blobs(filter_values: np.ndarray, filter_ids: OrderedDict[str, int]) -> list[bytes]:
    blobs = []
    for value in filter_values:
        text = str(value)
        if text == "PASS":
            blobs.append(_encode_typed_int_list([0]))
        elif text in {"", "."}:
            blobs.append(_encode_typed_int_list([None]))
        else:
            ids = [0 if label == "PASS" else filter_ids[label] for label in text.split(";") if label]
            blobs.append(_encode_typed_int_list(ids if ids else [None]))
    return blobs


def _normalise_info_values(
    data: SNPObject,
    variants_info: Optional[Sequence[str]],
    n_variants: int,
) -> np.ndarray:
    if variants_info is not None:
        values = np.asarray(variants_info, dtype=object)
    elif data.variants_info is not None and len(np.asarray(data.variants_info)) == n_variants:
        values = np.asarray(data.variants_info, dtype=object)
    else:
        values = np.full(n_variants, ".", dtype=object)

    if values.shape[0] != n_variants:
        raise ValueError("variants_info length must match the number of variants.")

    return np.asarray(["." if _is_missing_value(value) else str(value) for value in values], dtype=object)


def _iter_info_items(info: object) -> list[tuple[str, Optional[str]]]:
    if _is_missing_value(info):
        return []
    items: list[tuple[str, Optional[str]]] = []
    for raw_item in str(info).split(";"):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        match = _INFO_ITEM_RE.match(raw_item)
        if match is None:
            raise ValueError(f"Malformed INFO field item: {raw_item!r}")
        key, value = match.groups()
        if not key:
            raise ValueError(f"Malformed INFO field item: {raw_item!r}")
        items.append((key, value))
    return items


def _token_type(token: str) -> Optional[str]:
    if token == "." or token == "":
        return None
    if _INT_RE.match(token):
        return "Integer"
    try:
        float(token)
    except ValueError:
        return "String"
    return "Float"


def _infer_info_meta(info_values: np.ndarray, start_idx: int) -> tuple[OrderedDict[str, _InfoMeta], int]:
    order: OrderedDict[str, None] = OrderedDict()
    key_types: dict[str, set[str]] = {}
    key_counts: dict[str, list[int]] = {}
    key_has_flag: dict[str, bool] = {}

    for info in info_values:
        for key, value in _iter_info_items(info):
            order.setdefault(key, None)
            key_types.setdefault(key, set())
            key_counts.setdefault(key, [])
            key_has_flag.setdefault(key, False)
            if value is None:
                key_has_flag[key] = True
                key_counts[key].append(0)
                continue
            tokens = value.split(",")
            key_counts[key].append(len(tokens))
            for token in tokens:
                inferred = _token_type(token)
                if inferred is not None:
                    key_types[key].add(inferred)

    meta: OrderedDict[str, _InfoMeta] = OrderedDict()
    next_idx = start_idx
    for key in order:
        observed_types = key_types.get(key, set())
        if key_has_flag.get(key, False) and not observed_types:
            info_type = "Flag"
            number = "0"
        elif "String" in observed_types:
            info_type = "String"
            positive_counts = [count for count in key_counts[key] if count > 0]
            number = str(positive_counts[0]) if positive_counts and len(set(positive_counts)) == 1 else "."
        elif "Float" in observed_types:
            info_type = "Float"
            positive_counts = [count for count in key_counts[key] if count > 0]
            number = str(positive_counts[0]) if positive_counts and len(set(positive_counts)) == 1 else "."
        elif "Integer" in observed_types:
            info_type = "Integer"
            positive_counts = [count for count in key_counts[key] if count > 0]
            number = str(positive_counts[0]) if positive_counts and len(set(positive_counts)) == 1 else "."
        else:
            info_type = "String"
            number = "."
        meta[key] = _InfoMeta(idx=next_idx, id=key, number=number, type=info_type)
        next_idx += 1
    return meta, next_idx


def _parse_int_tokens(value: str) -> list[Optional[int]]:
    return [None if token in {"", "."} else int(token) for token in value.split(",")]


def _parse_float_tokens(value: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for token in value.split(","):
        if token in {"", "."}:
            out.append(None)
        else:
            out.append(float(token))
    return out


def _build_info_blobs(info_values: np.ndarray, info_meta: OrderedDict[str, _InfoMeta]) -> tuple[list[bytes], np.ndarray]:
    blobs: list[bytes] = []
    counts = np.empty(len(info_values), dtype=np.uint16)
    for idx, info in enumerate(info_values):
        encoded = bytearray()
        n_info = 0
        for key, value in _iter_info_items(info):
            meta = info_meta[key]
            encoded.extend(_encode_typed_int_list([meta.idx]))
            if value is None:
                encoded.extend(b"\x00" if meta.type == "Flag" else _encode_typed_string(""))
            elif meta.type == "Integer":
                encoded.extend(_encode_typed_int_list(_parse_int_tokens(value)))
            elif meta.type == "Float":
                encoded.extend(_encode_typed_float_list(_parse_float_tokens(value)))
            elif meta.type == "Flag":
                encoded.extend(b"\x00")
            else:
                encoded.extend(_encode_typed_string(value))
            n_info += 1
        if n_info > np.iinfo(np.uint16).max:
            raise ValueError("BCF records cannot contain more than 65535 INFO entries.")
        blobs.append(bytes(encoded))
        counts[idx] = n_info
    return blobs, np.ascontiguousarray(counts, dtype=np.uint16)


def _escape_header_value(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _build_header(
    *,
    contigs: Sequence[str],
    samples: np.ndarray,
    filter_ids: OrderedDict[str, int],
    info_meta: OrderedDict[str, _InfoMeta],
    has_gt: bool,
    has_gp: bool,
    gt_idx: int,
    gp_idx: int,
) -> str:
    lines = [
        "##fileformat=VCFv4.3",
        '##FILTER=<ID=PASS,Description="All filters passed",IDX=0>',
    ]
    for label, idx in filter_ids.items():
        label_esc = _escape_header_value(label)
        lines.append(f'##FILTER=<ID={label_esc},Description="Filter {label_esc}",IDX={idx}>')
    if has_gt:
        lines.append(f'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype",IDX={gt_idx}>')
    if has_gp:
        lines.append(f'##FORMAT=<ID=GP,Number=G,Type=Float,Description="Genotype probabilities",IDX={gp_idx}>')
    for meta in info_meta.values():
        info_id = _escape_header_value(meta.id)
        lines.append(
            f'##INFO=<ID={info_id},Number={meta.number},Type={meta.type},'
            f'Description="Written by snputils",IDX={meta.idx}>'
        )
    for contig in contigs:
        lines.append(f"##contig=<ID={_escape_header_value(contig)}>")

    cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    if len(samples) > 0 and (has_gt or has_gp):
        cols += ["FORMAT", *[str(sample) for sample in samples]]
    lines.append("\t".join(cols))
    return "\n".join(lines) + "\n"


def _unique_preserving_order(values: Sequence[object]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _write_native(
    *,
    filename: Path,
    header_text: str,
    contig_to_idx: dict[str, int],
    variants_chrom: np.ndarray,
    variants_pos: np.ndarray,
    variants_id: np.ndarray,
    variants_ref: np.ndarray,
    variants_alt: np.ndarray,
    qual_bits: np.ndarray,
    filter_blobs: list[bytes],
    info_blobs: list[bytes],
    n_info: np.ndarray,
    genotypes: Optional[np.ndarray],
    gt_mode: int,
    gt_width: int,
    gt_type_size: int,
    gp: Optional[np.ndarray],
    phased: bool,
    gt_idx: int,
    gp_idx: int,
    compression_level: int,
) -> None:
    if _native_bcf is None:
        raise ImportError(
            "The native BCF writer extension is not available. Reinstall snputils "
            "so snputils.snp.io.write._bcf can be built."
        )

    chrom_ids = np.ascontiguousarray([contig_to_idx[str(chrom)] for chrom in variants_chrom], dtype=np.int32)
    _native_bcf.write_bcf(
        filename=str(filename),
        header_text=header_text,
        chrom_ids=chrom_ids,
        positions=variants_pos,
        ids=[str(value) for value in variants_id],
        refs=[str(value) for value in variants_ref],
        alts=[str(value) for value in variants_alt],
        qual_bits=qual_bits,
        filter_blobs=filter_blobs,
        info_blobs=info_blobs,
        n_info=n_info,
        genotypes=np.empty((0, 0), dtype=np.int16) if genotypes is None else genotypes,
        gt_mode=gt_mode,
        gt_width=gt_width,
        gt_type_size=gt_type_size,
        gp=None if gp is None else gp,
        phased=bool(phased),
        gt_idx=int(gt_idx),
        gp_idx=int(gp_idx),
        compression_level=int(compression_level),
    )


class BCFWriter:
    """
    A writer class for exporting SNP data from a `snputils.snp.genobj.SNPObject`
    into a `.bcf` file.
    """

    def __init__(self, snpobj: SNPObject, filename: Union[str, Path], n_jobs: int = -1, phased: bool = False):
        """
        Args:
            snpobj:
                A SNPObject instance.
            filename:
                Path to the file where the data will be saved. It should end
                with `.bcf`. If the provided path does not have this extension,
                the `.bcf` extension will be appended.
            n_jobs:
                Number of jobs to run in parallel. Unused, included for API consistency.
            phased:
                If True, diploid GT values are written with phased separators.
        """
        del n_jobs
        self.__snpobj = snpobj
        self.__filename = Path(filename)
        self.__phased = phased

    def write(
        self,
        chrom_partition: bool = False,
        rename_missing_values: bool = True,
        before: Union[int, float, str] = -1,
        after: Union[int, float, str] = ".",
        variants_info: Optional[Sequence[str]] = None,
        compression_level: int = 1,
    ) -> None:
        """
        Writes the SNP data to BCF file(s).

        Args:
            chrom_partition:
                If True, individual BCF files are generated for each chromosome.
                If False, a single BCF file containing data for all chromosomes is created.
            rename_missing_values:
                If True, common missing representations in `genotypes` are encoded
                as native BCF missing GT values. The SNPObject is not mutated.
            before:
                Current representation of missing values in `genotypes`.
            after:
                Unused for BCF because the binary GT representation has a native
                missing value. Kept for API compatibility.
            variants_info:
                Optional per-variant INFO column values. When omitted,
                `snpobj.variants_info` is used when present.
            compression_level:
                BGZF compression level from 0 to 9. The default favors write speed
                while producing standard BGZF-compressed BCF.
        """
        del after
        filename = self.__filename
        if filename.suffix != ".bcf":
            filename = filename.with_suffix(".bcf")
        self.__filename = filename

        data = self.__snpobj
        if chrom_partition:
            chroms = data.unique_chrom
            if chroms is None:
                raise ValueError("BCF chromosome partitioning requires `variants_chrom`.")
            for chrom in chroms:
                data_chrom = data.filter_variants(chrom=chrom, inplace=False)
                info_chrom = None
                if variants_info is not None:
                    mask = np.asarray(data.variants_chrom) == chrom
                    info_chrom = [variants_info[i] for i in np.where(mask)[0]]
                log.debug("Storing chromosome %s", chrom)
                file = filename.parent / f"{filename.stem}_{chrom}.bcf"
                self._write_chromosome_data(
                    file,
                    data_chrom,
                    variants_info=info_chrom,
                    rename_missing_values=rename_missing_values,
                    before=before,
                    compression_level=compression_level,
                )
            return

        self._write_chromosome_data(
            filename,
            data,
            variants_info=variants_info,
            rename_missing_values=rename_missing_values,
            before=before,
            compression_level=compression_level,
        )

    def _write_chromosome_data(
        self,
        file: Path,
        data: SNPObject,
        *,
        variants_info: Optional[Sequence[str]],
        rename_missing_values: bool,
        before: Union[int, float, str],
        compression_level: int,
    ) -> None:
        n_variants = data.n_snps
        variants_chrom = _normalise_text_array(
            data.variants_chrom, n_variants, attr="variants_chrom", default=".", required=True
        )
        variants_ref = _normalise_text_array(
            data.variants_ref, n_variants, attr="variants_ref", default="N", required=True
        )
        variants_alt = _normalise_text_array(data.variants_alt, n_variants, attr="variants_alt", default=".")
        variants_id = _normalise_text_array(data.variants_id, n_variants, attr="variants_id", default=".")
        variants_pos = _normalise_positions(data.variants_pos, n_variants)
        qual_bits = _normalise_qual_bits(data.variants_qual, n_variants)
        genotypes, gt_mode, gt_width, gt_type_size = _normalise_genotypes(
            data.genotypes,
            rename_missing_values=rename_missing_values,
            before=before,
        )
        gp = _normalise_gp(data.calldata_gp, n_variants)

        n_samples = _infer_sample_count(genotypes, gp, data.samples)
        if n_samples > 0 and genotypes is None and gp is None:
            raise ValueError("BCFWriter requires `genotypes` or `calldata_gp` when samples are present.")
        samples = _normalise_samples(data.samples, n_samples)

        has_gt = genotypes is not None and n_samples > 0
        has_gp = gp is not None and n_samples > 0
        filter_values = _normalise_filter_values(data.variants_filter_pass, n_variants)
        info_values = _normalise_info_values(data, variants_info, n_variants)

        next_idx = 1
        filter_ids, next_idx = _collect_filter_ids(filter_values, next_idx)
        gt_idx = next_idx if has_gt else -1
        if has_gt:
            next_idx += 1
        gp_idx = next_idx if has_gp else -1
        if has_gp:
            next_idx += 1
        info_meta, next_idx = _infer_info_meta(info_values, next_idx)
        del next_idx

        filter_blobs = _build_filter_blobs(filter_values, filter_ids)
        info_blobs, n_info = _build_info_blobs(info_values, info_meta)
        contigs = _unique_preserving_order(variants_chrom)
        contig_to_idx = {contig: idx for idx, contig in enumerate(contigs)}
        header_text = _build_header(
            contigs=contigs,
            samples=samples,
            filter_ids=filter_ids,
            info_meta=info_meta,
            has_gt=has_gt,
            has_gp=has_gp,
            gt_idx=gt_idx,
            gp_idx=gp_idx,
        )

        log.info("Writing BCF to '%s'...", file)
        _write_native(
            filename=file,
            header_text=header_text,
            contig_to_idx=contig_to_idx,
            variants_chrom=variants_chrom,
            variants_pos=variants_pos,
            variants_id=variants_id,
            variants_ref=variants_ref,
            variants_alt=variants_alt,
            qual_bits=qual_bits,
            filter_blobs=filter_blobs,
            info_blobs=info_blobs,
            n_info=n_info,
            genotypes=genotypes,
            gt_mode=gt_mode if has_gt else 0,
            gt_width=gt_width,
            gt_type_size=gt_type_size,
            gp=gp if has_gp else None,
            phased=self.__phased,
            gt_idx=gt_idx,
            gp_idx=gp_idx,
            compression_level=compression_level,
        )
        log.info("Finished writing BCF to '%s'.", file)
