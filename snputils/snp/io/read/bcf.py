from __future__ import annotations

import gzip
import logging
import re
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from snputils._utils.genotypes import (
    GenotypeMode,
    normalize_genotype_mode,
    sum_diploid_genotypes,
)
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from snputils.snp.io.read.vcf import (
    _non_diploid_chromosome_mask_or_none,
    _normalized_non_diploid_chromosome,
    _parse_vcf_region,
    _vcf_region_matches,
)

log = logging.getLogger(__name__)

_DEFAULT_FIELDS = ["GT", "IID", "REF", "ALT", "#CHROM", "ID", "POS", "QUAL", "FILTER"]
_ALL_FIELDS = ["GT", "GP", "IID", "REF", "ALT", "#CHROM", "ID", "POS", "QUAL", "FILTER", "INFO"]
_CORE_FIELDS = frozenset(_DEFAULT_FIELDS)

_BCF_MAGIC = b"BCF\x02\x02"
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_F32 = struct.Struct("<f")
_TYPE_SIZES = {0: 0, 1: 1, 2: 2, 3: 4, 5: 4, 7: 1}
_INT_UNSIGNED_DTYPES = {1: np.uint8, 2: np.dtype("<u2"), 4: np.dtype("<u4")}
_FLOAT_MISSING = 0x7F800001
_FLOAT_VECTOR_END = 0x7F800002
_HEADER_META_RE = re.compile(r"^##(contig|INFO|FORMAT|FILTER)=<(.*)>$")
_HEADER_KV_RE = re.compile(r'([^=,]+)=(".*?"|[^,<>]+)')

# Struct for reading the 6 fixed u32 fields from a BCF record's shared section.
# Layout at base (= record_offset + 8):
#   [0] chrom_id (i32), [1] pos (i32), [2] rlen (i32), [3] qual (f32),
#   [4] n_alleles_info (u32: top 16 = n_alleles, bottom 16 = n_info),
#   [5] n_fmt_n_samples (u32: top 8 = n_fmt, bottom 24 = n_samples)
_FIXED_FIELDS = struct.Struct("<iiIfII")


@dataclass(frozen=True)
class _BCFHeader:
    samples: np.ndarray
    contigs: Dict[int, str]
    filters: Dict[int, str]
    info: Dict[int, Dict[str, str]]
    formats: Dict[int, Dict[str, str]]


def _as_field_list(fields: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [fields]
    return list(fields)


def _normalize_fields(
    fields: Optional[Union[str, Sequence[str]]],
    exclude_fields: Optional[Union[str, Sequence[str]]],
) -> list[str]:
    exclude = {"#CHROM" if field == "CHROM" else field for field in (_as_field_list(exclude_fields) or [])}
    requested = _as_field_list(fields)
    if requested is None:
        resolved = list(_DEFAULT_FIELDS)
    elif requested == ["*"]:
        resolved = list(_ALL_FIELDS)
    else:
        resolved = requested

    normalized = []
    for field in resolved:
        canonical = "#CHROM" if field == "CHROM" else field
        if canonical not in _ALL_FIELDS:
            raise ValueError(
                f"Unsupported BCF field: {field}. "
                f"Supported fields are {', '.join(_ALL_FIELDS)} and '*'."
            )
        if canonical not in exclude:
            normalized.append(canonical)
    return normalized


def _resolve_sample_indices(
    file_samples: np.ndarray,
    sample_ids: Optional[Sequence[str]],
    sample_idxs: Optional[Sequence[int]],
) -> np.ndarray:
    if sample_idxs is not None and sample_ids is not None:
        raise ValueError("Only one of sample_idxs and sample_ids can be specified.")

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


def _parse_header_fields(text: str) -> Dict[str, str]:
    return {
        key: value.strip('"')
        for key, value in _HEADER_KV_RE.findall(text)
    }


def _parse_bcf_header(text: str) -> _BCFHeader:
    samples: list[str] = []
    contigs: Dict[int, str] = {}
    filters: Dict[int, str] = {0: "PASS"}
    info: Dict[int, Dict[str, str]] = {}
    formats: Dict[int, Dict[str, str]] = {}
    contig_idx = 0

    for line in text.rstrip("\0").splitlines():
        if not line:
            continue
        if line.startswith("#CHROM"):
            parts = line.split("\t")
            samples = parts[9:] if len(parts) > 9 else []
            continue

        match = _HEADER_META_RE.match(line)
        if match is None:
            continue

        kind, payload = match.groups()
        fields = _parse_header_fields(payload)
        if kind == "contig":
            contigs[contig_idx] = fields["ID"]
            contig_idx += 1
            continue

        idx = int(fields.get("IDX", "-1"))
        if idx < 0:
            continue
        if kind == "FILTER":
            filters[idx] = fields["ID"]
        elif kind == "INFO":
            info[idx] = fields
        elif kind == "FORMAT":
            formats[idx] = fields

    return _BCFHeader(
        samples=np.asarray(samples, dtype=object),
        contigs=contigs,
        filters=filters,
        info=info,
        formats=formats,
    )


def _header_has_non_diploid_contigs(header: _BCFHeader) -> bool:
    return any(
        _normalized_non_diploid_chromosome(chrom) is not None
        for chrom in header.contigs.values()
    )


def _non_diploid_contig_mask_or_none(
    contig_ids: np.ndarray,
    header: _BCFHeader,
) -> Optional[np.ndarray]:
    unique_ids = np.unique(contig_ids)
    non_diploid_ids = {
        int(contig_id)
        for contig_id in unique_ids
        if _normalized_non_diploid_chromosome(header.contigs[int(contig_id)]) is not None
    }
    if not non_diploid_ids:
        return None
    return np.isin(contig_ids, list(non_diploid_ids))


def _normalize_chromosome_ploidy(value: Optional[str]) -> str:
    if value is None:
        return "auto"

    value = str(value).strip().lower()
    if value not in {"auto", "autosomal", "mixed"}:
        raise ValueError(
            "chromosome_ploidy must be one of None, 'auto', 'autosomal', or 'mixed'."
        )
    return value


def _read_bgzf_or_gzip(filename: Union[str, bytes]) -> bytes:
    """Read a BGZF-compressed file, falling back to generic gzip.

    BCF files are normally BGZF. Parsing BGZF blocks directly avoids per-member
    overhead in ``gzip.GzipFile`` while keeping the dependency footprint at the
    Python standard library.
    """
    chunks = []
    with open(filename, "rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return b"".join(chunks)
            if len(header) < 12 or header[:3] != b"\x1f\x8b\x08" or not (header[3] & 4):
                break

            xlen = int.from_bytes(header[10:12], "little")
            extra = handle.read(xlen)
            if len(extra) != xlen:
                raise EOFError("Unexpected end of BGZF extra header.")

            extra_offset = 0
            block_size = None
            while extra_offset + 4 <= xlen:
                subfield_len = int.from_bytes(extra[extra_offset + 2:extra_offset + 4], "little")
                if extra_offset + 4 + subfield_len > xlen:
                    raise ValueError("Malformed BGZF extra header: subfield length extends beyond XLEN.")
                if extra[extra_offset:extra_offset + 2] == b"BC" and subfield_len == 2:
                    block_size = int.from_bytes(extra[extra_offset + 4:extra_offset + 6], "little") + 1
                    break
                extra_offset += 4 + subfield_len

            if block_size is None:
                break

            remaining = block_size - 12 - xlen
            if remaining < 8:
                raise ValueError("Malformed BGZF block: block size is too small.")

            block_tail = handle.read(remaining)
            if len(block_tail) != remaining:
                raise EOFError("Unexpected end of BGZF block.")

            compressed = block_tail[:-8]
            if compressed:
                chunk = zlib.decompress(compressed, -15)
                if chunk:
                    chunks.append(chunk)

    with gzip.open(filename, "rb") as handle:
        return handle.read()


def _load_bcf_data(filename: Union[str, bytes]) -> Tuple[bytes, int, _BCFHeader]:
    data = _read_bgzf_or_gzip(filename)
    if data[:5] != _BCF_MAGIC:
        raise ValueError(f"{filename!r} does not look like a BCF2.2 file.")
    header_len = _U32.unpack_from(data, 5)[0]
    header_start = 9
    header_end = header_start + header_len
    header_text = data[header_start:header_end].decode("utf-8", "replace")
    return data, header_end, _parse_bcf_header(header_text)


def _read_typed_descriptor(data: bytes, offset: int) -> Tuple[int, int, int, int]:
    byte = data[offset]
    offset += 1
    type_code = byte & 0x0F
    n_vals = byte >> 4
    if n_vals == 0:
        return 0, 0, 0, offset
    if n_vals == 15:
        length_descriptor = data[offset]
        offset += 1
        length_type = length_descriptor & 0x0F
        length_size = _TYPE_SIZES.get(length_type, 0)
        if length_type not in (1, 2, 3) or length_size == 0:
            raise ValueError("Cannot identify the BCF typed-value length encoding.")
        n_vals = int.from_bytes(data[offset:offset + length_size], "little", signed=False)
        offset += length_size
    type_size = _TYPE_SIZES.get(type_code, 0)
    if type_code not in _TYPE_SIZES:
        raise ValueError(f"Unsupported BCF atomic type code: {type_code}")
    return n_vals, type_code, type_size, offset


def _skip_typed_value_fast(data: bytes, offset: int) -> int:
    b = data[offset]
    type_size = _TYPE_SIZES[b & 0x0F]
    n_vals = b >> 4
    if n_vals < 15:
        return offset + 1 + n_vals * type_size
    length_descriptor = data[offset + 1]
    length_type = length_descriptor & 0x0F
    length_size = _TYPE_SIZES[length_type]
    n_vals = int.from_bytes(data[offset + 2 : offset + 2 + length_size], "little")
    return offset + 2 + length_size + n_vals * type_size


def _read_scalar_typed_int(data: bytes, offset: int) -> Tuple[int, int]:
    n_vals, type_code, type_size, offset = _read_typed_descriptor(data, offset)
    if n_vals != 1 or type_code not in (1, 2, 3):
        raise ValueError("Expected a scalar integer typed value in the BCF record.")
    value = int.from_bytes(data[offset:offset + type_size], "little", signed=False)
    return value, offset + type_size


def _read_typed_string(data: bytes, offset: int) -> Tuple[str, int]:
    n_vals, type_code, type_size, offset = _read_typed_descriptor(data, offset)
    if type_code != 7:
        raise ValueError("Expected a typed string in the BCF record.")
    end = offset + n_vals * type_size
    value = data[offset:end].split(b"\0", 1)[0].decode("utf-8")
    return value, end


def _skip_typed_value(data: bytes, offset: int) -> int:
    n_vals, _type_code, type_size, offset = _read_typed_descriptor(data, offset)
    return offset + n_vals * type_size


def _read_int_list(data: bytes, offset: int) -> Tuple[List[Optional[int]], int]:
    n_vals, type_code, type_size, offset = _read_typed_descriptor(data, offset)
    if type_code == 0:
        return [], offset
    if type_code not in (1, 2, 3):
        raise ValueError(f"Expected an integer typed value, found atomic type {type_code}.")

    missing = 1 << ((type_size * 8) - 1)
    vector_end = missing | 0x1
    values: List[Optional[int]] = []
    for _ in range(n_vals):
        raw = int.from_bytes(data[offset:offset + type_size], "little", signed=False)
        offset += type_size
        if raw == vector_end:
            break
        if raw == missing:
            values.append(None)
            continue
        values.append(int.from_bytes(raw.to_bytes(type_size, "little"), "little", signed=True))
    return values, offset


def _read_float_list(data: bytes, offset: int) -> Tuple[List[float], int]:
    n_vals, type_code, type_size, offset = _read_typed_descriptor(data, offset)
    if type_code == 0:
        return [], offset
    if type_code != 5 or type_size != 4:
        raise ValueError(f"Expected a float typed value, found atomic type {type_code}.")

    values: List[float] = []
    for _ in range(n_vals):
        raw = _U32.unpack_from(data, offset)[0]
        offset += 4
        if raw == _FLOAT_VECTOR_END:
            break
        if raw == _FLOAT_MISSING:
            values.append(np.nan)
            continue
        values.append(_F32.unpack_from(data, offset - 4)[0])
    return values, offset


def _render_info_value(value: Any) -> str:
    if value is None:
        return "."
    if isinstance(value, list):
        rendered = []
        for item in value:
            if item is None:
                rendered.append(".")
            elif isinstance(item, float) and np.isnan(item):
                rendered.append(".")
            else:
                rendered.append(str(item))
        return ",".join(rendered)
    if isinstance(value, float) and np.isnan(value):
        return "."
    return str(value)


def _variant_qual(data: bytes, base_offset: int) -> float:
    raw = _U32.unpack_from(data, base_offset + 12)[0]
    if raw == _FLOAT_MISSING:
        return np.nan
    return _F32.unpack_from(data, base_offset + 12)[0]


def _decode_record_identifiers(
    data: bytes,
    record_offset: int,
    header: _BCFHeader,
) -> Tuple[str, int, str, str, Tuple[str, ...]]:
    base = record_offset + 8
    chrom = header.contigs[_I32.unpack_from(data, base)[0]]
    pos = _I32.unpack_from(data, base + 4)[0] + 1
    n_alleles = _U32.unpack_from(data, base + 16)[0] >> 16
    offset = base + 24
    variant_id, offset = _read_typed_string(data, offset)
    ref, offset = _read_typed_string(data, offset)
    alts = []
    for _ in range(max(0, n_alleles - 1)):
        alt, offset = _read_typed_string(data, offset)
        alts.append(alt)
    return chrom, pos, variant_id, ref, tuple(alts)


def _record_identifiers(chrom: str, pos: int, variant_id: str, ref: str, alts: Sequence[str]) -> set[str]:
    identifiers = {f"{chrom}:{pos}", f"{chrom}:{pos}:{ref}:{','.join(alts)}"}
    if variant_id not in ("", "."):
        identifiers.add(variant_id)
    return identifiers


def _parse_chrom_pos_identifier(identifier: str) -> Optional[Tuple[str, int]]:
    parts = identifier.split(":")
    if len(parts) != 2 or not parts[0]:
        return None
    try:
        pos = int(parts[1])
    except ValueError:
        return None
    if pos < 1:
        return None
    return parts[0], pos


def _count_records(data: bytes, body_offset: int) -> int:
    offset = body_offset
    end = len(data)
    count = 0
    while offset < end:
        l_shared = _U32.unpack_from(data, offset)[0]
        l_indiv = _U32.unpack_from(data, offset + 4)[0]
        offset += 8 + l_shared + l_indiv
        count += 1
    if offset != end:
        raise ValueError("Malformed BCF: record boundaries do not consume the full file.")
    return count


_U32_PAIR = struct.Struct("<II")


def _build_record_offsets(data: bytes, body_offset: int) -> np.ndarray:
    """Build an array of byte offsets for every record in one pass."""
    offsets = []
    offset = body_offset
    end = len(data)
    unpack = _U32_PAIR.unpack_from
    while offset < end:
        offsets.append(offset)
        l_shared, l_indiv = unpack(data, offset)
        offset += 8 + l_shared + l_indiv
    if offset != end:
        raise ValueError("Malformed BCF: record boundaries do not consume the full file.")
    return np.asarray(offsets, dtype=np.int64)


def _build_indiv_offsets(data: bytes, body_offset: int) -> Tuple[np.ndarray, bool]:
    """Build individual-section offsets for all records.

    This is the cheapest scan needed by the GT-only fast path. It intentionally
    avoids reading fixed fields such as POS/QUAL/INFO counts when callers only
    need FORMAT/GT.
    """
    offset = body_offset
    end = len(data)
    if offset >= end:
        return np.empty(0, dtype=np.int64), True
    if offset + 8 > end:
        raise ValueError("Malformed BCF: record header is truncated.")

    indiv_offsets = []
    append_offset = indiv_offsets.append
    unpack = _U32_PAIR.unpack_from
    uniform_indiv = True

    l_shared, first_l_indiv = unpack(data, offset)
    append_offset(offset + 8 + l_shared)
    offset += 8 + l_shared + first_l_indiv

    while offset < end:
        if offset + 8 > end:
            raise ValueError("Malformed BCF: record header is truncated.")
        l_shared, l_indiv = unpack(data, offset)
        append_offset(offset + 8 + l_shared)
        if l_indiv != first_l_indiv:
            uniform_indiv = False
        offset += 8 + l_shared + l_indiv

    if offset != end:
        raise ValueError("Malformed BCF: record boundaries do not consume the full file.")
    return np.asarray(indiv_offsets, dtype=np.int64), uniform_indiv


def _gather_u32(raw: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Read little-endian uint32 values at given byte offsets using numpy gather.

    Reads 4 consecutive bytes at each offset and assembles them into uint32 values
    using vectorized shift-and-add instead of per-element struct.unpack_from.
    """
    b0 = raw[offsets].astype(np.uint32)
    b1 = raw[offsets + 1].astype(np.uint32)
    b2 = raw[offsets + 2].astype(np.uint32)
    b3 = raw[offsets + 3].astype(np.uint32)
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


def _extract_fixed_fields(
    data: bytes,
    record_offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized extraction of fixed-layout fields from all records.

    Uses numpy byte-level gather to read all fixed fields across all records
    in bulk, avoiding per-record Python loops and struct.unpack_from calls.

    Returns:
        l_shared, l_indiv, contig_ids, positions, qual_raw, n_alleles, n_info, n_fmt
        All as 1-D numpy arrays with one element per record.
    """
    raw = np.frombuffer(data, dtype=np.uint8)

    # Record header: l_shared (u32 at +0), l_indiv (u32 at +4)
    l_shared = _gather_u32(raw, record_offsets)
    l_indiv = _gather_u32(raw, record_offsets + 4)

    # Fixed section at base = offset + 8:
    # chrom_id (i32 at +0), pos (i32 at +4), rlen (i32 at +8), qual (u32 at +12),
    # n_alleles_info (u32 at +16), n_fmt_n_samples (u32 at +20)
    base_offsets = record_offsets + 8
    contig_ids = _gather_u32(raw, base_offsets).view(np.int32)
    positions = _gather_u32(raw, base_offsets + 4).view(np.int32).astype(np.int64) + 1
    qual_raw = _gather_u32(raw, base_offsets + 12)
    n_alleles_info = _gather_u32(raw, base_offsets + 16)
    n_fmt_n_samples = _gather_u32(raw, base_offsets + 20)

    n_alleles = (n_alleles_info >> 16).astype(np.uint16)
    n_info = (n_alleles_info & 0xFFFF).astype(np.uint16)
    n_fmt = (n_fmt_n_samples >> 24).astype(np.uint8)

    return l_shared, l_indiv, contig_ids, positions, qual_raw, n_alleles, n_info, n_fmt


def _resolve_variant_request(
    data: bytes,
    body_offset: int,
    header: _BCFHeader,
    region_filter: Optional[Tuple[str, Optional[int], Optional[int]]],
    variant_ids: Optional[Sequence[str]],
    variant_idxs: Optional[Sequence[int]],
) -> Tuple[int, Optional[List[int]], Optional[List[int]]]:
    requested_variant_idxs = None
    n_records: Optional[int] = None
    if variant_idxs is not None:
        raw_variant_idxs = np.asarray(variant_idxs, dtype=int).ravel()
        if np.any(raw_variant_idxs < 0):
            n_records = _count_records(data, body_offset)
            if np.any((raw_variant_idxs < -n_records) | (raw_variant_idxs >= n_records)):
                raise ValueError("One or more variant indexes are out of bounds.")
            requested_variant_idxs = np.mod(raw_variant_idxs, n_records).tolist()
        else:
            requested_variant_idxs = raw_variant_idxs.tolist()

    if variant_ids is None and requested_variant_idxs is None and region_filter is None:
        n_records = _count_records(data, body_offset)
        return n_records, None, None

    if region_filter is not None and requested_variant_idxs is None and variant_ids is None:
        contig_lookup = {name: idx for idx, name in header.contigs.items()}
        region_chrom, start, end_pos = region_filter
        contig_id = contig_lookup.get(region_chrom)
        if contig_id is not None:
            try:
                from snputils.snp.io.read import _bcf
            except ImportError:
                _bcf = None
            if _bcf is not None:
                offsets_buffer, _n_selected, n_records = _bcf.select_region_offsets(
                    data,
                    body_offset,
                    int(contig_id),
                    -1 if start is None else int(start),
                    -1 if end_pos is None else int(end_pos),
                )
                selected_offsets = np.frombuffer(offsets_buffer, dtype=np.dtype("<i8")).astype(np.int64, copy=False)
                return int(n_records), selected_offsets.tolist(), None
        else:
            return 0, [], None

    requested_idx_set = None if requested_variant_idxs is None else set(requested_variant_idxs)
    requested_id_values = None if variant_ids is None else [
        str(value) for value in np.asarray(variant_ids, dtype=object).ravel()
    ]
    requested_ids = None if requested_id_values is None else set(requested_id_values)

    if requested_id_values is not None and requested_variant_idxs is None and region_filter is None:
        unique_requested_ids = list(requested_ids)
        parsed_ids = [_parse_chrom_pos_identifier(identifier) for identifier in unique_requested_ids]
        if all(parsed is not None for parsed in parsed_ids):
            try:
                from snputils.snp.io.read import _bcf
            except ImportError:
                _bcf = None
            if _bcf is not None:
                contig_lookup = {name: idx for idx, name in header.contigs.items()}
                found_ids = set()
                selected_offset_set = set()
                n_records = 0
                for identifier, (chrom, pos) in zip(unique_requested_ids, parsed_ids):
                    contig_id = contig_lookup.get(chrom)
                    if contig_id is None:
                        continue
                    offsets_buffer, n_selected, n_records = _bcf.select_region_offsets(
                        data,
                        body_offset,
                        int(contig_id),
                        int(pos),
                        int(pos),
                    )
                    if n_selected:
                        offsets = np.frombuffer(offsets_buffer, dtype=np.dtype("<i8")).astype(np.int64, copy=False)
                        selected_offset_set.update(int(offset) for offset in offsets)
                        found_ids.add(identifier)
                missing = sorted(requested_ids - found_ids)
                if missing:
                    raise ValueError(f"The following specified variants were not found: {missing}")
                return int(n_records), sorted(selected_offset_set), None

    found_ids = set()
    selected_offsets: List[int] = []
    selected_by_row: Dict[int, int] = {}

    offset = body_offset
    row_idx = 0
    end = len(data)
    while offset < end:
        l_shared = _U32.unpack_from(data, offset)[0]
        l_indiv = _U32.unpack_from(data, offset + 4)[0]

        want_row = requested_idx_set is None or row_idx in requested_idx_set
        passes = want_row
        chrom = None
        pos = None
        if passes and region_filter is not None:
            base = offset + 8
            chrom = header.contigs[_I32.unpack_from(data, base)[0]]
            pos = _I32.unpack_from(data, base + 4)[0] + 1
            if not _vcf_region_matches(chrom, pos, region_filter):
                passes = False
        if passes and requested_ids is not None:
            chrom, pos, variant_id, ref, alts = _decode_record_identifiers(data, offset, header)
            record_ids = _record_identifiers(chrom, pos, variant_id, ref, alts)
            matched = requested_ids.intersection(record_ids)
            if matched:
                found_ids.update(matched)
            else:
                passes = False

        if passes:
            if requested_variant_idxs is None:
                selected_offsets.append(offset)
            else:
                selected_by_row[row_idx] = offset

        offset += 8 + l_shared + l_indiv
        row_idx += 1

    if offset != end:
        raise ValueError("Malformed BCF: record boundaries do not consume the full file.")
    if n_records is None:
        n_records = row_idx
    if requested_variant_idxs is not None and requested_variant_idxs:
        if min(requested_variant_idxs) < 0 or max(requested_variant_idxs) >= n_records:
            raise ValueError("One or more variant indexes are out of bounds.")

    if requested_ids is not None:
        missing = sorted(requested_ids - found_ids)
        if missing:
            raise ValueError(f"The following specified variants were not found: {missing}")

    if requested_variant_idxs is None:
        return n_records, selected_offsets, None
    return n_records, None, [selected_by_row[row] for row in requested_variant_idxs if row in selected_by_row]


def _raise_if_unphased_bcf_gt(raw: np.ndarray, decoded: np.ndarray) -> None:
    if raw.shape[1] < 2:
        return
    second_allele_called = decoded[:, 1] >= 0
    second_allele_phased = (raw[:, 1] & 1) != 0
    if np.any(second_allele_called & ~second_allele_phased):
        raise ValueError(
            "Cannot read unphased BCF genotypes with genotype_mode='phased'; "
            "use genotype_mode='dosage' to load 0/1/2 genotype dosages."
        )


def _decode_gt_array(
    data: bytes,
    offset: int,
    n_samples: int,
    n_vals: int,
    type_size: int,
    *,
    require_phase: bool = False,
    phase_sample_idxs: Optional[np.ndarray] = None,
) -> np.ndarray:
    if type_size not in _INT_UNSIGNED_DTYPES:
        raise ValueError(f"Unsupported GT integer width in BCF FORMAT/GT: {type_size}")
    if n_vals < 1:
        return np.empty((n_samples, 0), dtype=np.int8)
    raw = np.frombuffer(
        data,
        dtype=_INT_UNSIGNED_DTYPES[type_size],
        count=n_samples * n_vals,
        offset=offset,
    ).reshape(n_samples, n_vals)
    decoded = (raw.astype(np.int32, copy=False) >> 1) - 1
    if n_vals == 1:
        padded = np.full((n_samples, 2), -1, dtype=np.int8)
        padded[:, 0] = decoded[:, 0].astype(np.int8, copy=False)
        return padded
    if n_vals != 2:
        raise ValueError("BCFReader currently supports haploid or diploid GT fields only.")
    if require_phase:
        phase_raw = raw if phase_sample_idxs is None else raw[phase_sample_idxs]
        phase_decoded = decoded if phase_sample_idxs is None else decoded[phase_sample_idxs]
        _raise_if_unphased_bcf_gt(phase_raw, phase_decoded)
    return decoded.astype(np.int8, copy=False)


def _decode_gp_array(data: bytes, offset: int, n_samples: int, n_vals: int) -> np.ndarray:
    if n_vals < 1:
        return np.empty((n_samples, 0), dtype=np.float32)
    raw = np.frombuffer(
        data,
        dtype=np.dtype("<u4"),
        count=n_samples * n_vals,
        offset=offset,
    ).reshape(n_samples, n_vals)
    values = np.frombuffer(raw.tobytes(), dtype=np.dtype("<f4")).reshape(n_samples, n_vals).copy()
    values[raw == _FLOAT_MISSING] = np.nan
    return values


def _decode_gp_raw(raw: np.ndarray) -> np.ndarray:
    values = np.frombuffer(raw.tobytes(), dtype=np.dtype("<f4")).reshape(raw.shape).copy()
    values[raw == _FLOAT_MISSING] = np.nan
    return values

def _parse_filter_pass(
    data: bytes,
    offset: int,
    header: _BCFHeader,
) -> Tuple[bool, int]:
    filter_ids, offset = _read_int_list(data, offset)
    if any(idx is None for idx in filter_ids):
        return False, offset
    filter_names = [header.filters.get(int(idx), str(idx)) for idx in filter_ids if idx is not None]
    if not filter_names:
        return True, offset
    return len(filter_names) == 1 and filter_names[0] == "PASS", offset


def _parse_info_string(
    data: bytes,
    offset: int,
    n_info: int,
    header: _BCFHeader,
) -> str:
    items = []
    for _ in range(n_info):
        info_idx, offset = _read_scalar_typed_int(data, offset)
        meta = header.info.get(info_idx, {"ID": f"INFO_{info_idx}", "Type": "", "Number": ""})
        key = meta["ID"]
        n_vals, type_code, type_size, value_offset = _read_typed_descriptor(data, offset)
        offset = value_offset

        if type_code == 0:
            items.append(key)
            continue
        if type_code == 7:
            value = data[offset:offset + n_vals * type_size].split(b"\0", 1)[0].decode("utf-8")
            offset += n_vals * type_size
            items.append(f"{key}={value}")
            continue
        if type_code == 5:
            values = []
            for _ in range(n_vals):
                raw = _U32.unpack_from(data, offset)[0]
                offset += 4
                if raw == _FLOAT_VECTOR_END:
                    break
                if raw == _FLOAT_MISSING:
                    values.append(np.nan)
                    continue
                values.append(_F32.unpack_from(data, offset - 4)[0])
            items.append(f"{key}={_render_info_value(values[0] if len(values) == 1 else values)}")
            continue
        if type_code in (1, 2, 3):
            values = []
            missing = 1 << ((type_size * 8) - 1)
            vector_end = missing | 0x1
            for _ in range(n_vals):
                raw = int.from_bytes(data[offset:offset + type_size], "little", signed=False)
                offset += type_size
                if raw == vector_end:
                    break
                if raw == missing:
                    values.append(None)
                    continue
                values.append(int.from_bytes(raw.to_bytes(type_size, "little"), "little", signed=True))
            items.append(f"{key}={_render_info_value(values[0] if len(values) == 1 else values)}")
            continue

        raise ValueError(f"Unsupported BCF INFO atomic type code: {type_code}")

    return ";".join(items) if items else "."


def _skip_info_block(data: bytes, offset: int, n_info: int) -> int:
    for _ in range(n_info):
        offset = _skip_typed_value(data, offset)
        offset = _skip_typed_value(data, offset)
    return offset


def _skip_shared_to_filter(data: bytes, base: int, n_alleles: int) -> int:
    """Skip from base+24 past ID, REF, and ALT strings to reach the FILTER field."""
    offset = base + 24
    # Skip ID string
    offset = _skip_typed_value(data, offset)
    # Skip REF string
    offset = _skip_typed_value(data, offset)
    # Skip ALT strings
    for _ in range(max(0, n_alleles - 1)):
        offset = _skip_typed_value(data, offset)
    return offset


def _probe_gt_layout(
    data: bytes,
    indiv_offset: int,
    n_fmt: int,
    n_samples: int,
    header: _BCFHeader,
) -> Optional[Tuple[int, int, int, int]]:
    """Probe the FORMAT section of one record to find GT layout.

    Returns (gt_data_offset_from_indiv, n_vals, type_size, total_indiv_bytes)
    or None if GT is not found.

    gt_data_offset_from_indiv is the byte offset from indiv_offset to the start
    of the GT sample data for this record.
    """
    format_offset = indiv_offset
    for _ in range(n_fmt):
        fmt_idx, format_offset = _read_scalar_typed_int(data, format_offset)
        n_vals, type_code, type_size, values_offset = _read_typed_descriptor(data, format_offset)
        key = header.formats.get(fmt_idx, {"ID": f"FORMAT_{fmt_idx}"})["ID"]
        values_nbytes = n_samples * n_vals * type_size

        if key == "GT":
            gt_data_offset = values_offset - indiv_offset
            return gt_data_offset, n_vals, type_size, values_offset + values_nbytes - indiv_offset
        format_offset = values_offset + values_nbytes
    return None


def _probe_gp_layout(
    data: bytes,
    indiv_offset: int,
    n_fmt: int,
    n_samples: int,
    header: _BCFHeader,
) -> Optional[Tuple[int, int, int]]:
    """Probe the FORMAT section of one record to find GP layout.

    Returns (gp_data_offset_from_indiv, n_vals, type_size) or None if GP is not
    found.
    """
    format_offset = indiv_offset
    for _ in range(n_fmt):
        fmt_idx, format_offset = _read_scalar_typed_int(data, format_offset)
        n_vals, type_code, type_size, values_offset = _read_typed_descriptor(data, format_offset)
        key = header.formats.get(fmt_idx, {"ID": f"FORMAT_{fmt_idx}"})["ID"]
        values_nbytes = n_samples * n_vals * type_size

        if key == "GP":
            if type_code != 5 or type_size != 4:
                raise ValueError("BCF FORMAT/GP is expected to be stored as float32 values.")
            gp_data_offset = values_offset - indiv_offset
            return gp_data_offset, n_vals, type_size
        format_offset = values_offset + values_nbytes
    return None


def _batch_decode_gt(
    data: bytes,
    indiv_offsets: np.ndarray,
    gt_data_rel_offset: int,
    n_vals: int,
    type_size: int,
    n_samples: int,
    n_records: int,
    sample_index_array: np.ndarray,
    return_dosage: bool,
    missing_as_haploid: Optional[Union[bool, np.ndarray]] = None,
) -> np.ndarray:
    """Batch-decode GT data for all records using vectorized numpy operations.

    Instead of calling np.frombuffer per record, gather GT bytes in bounded
    chunks and decode each chunk with NumPy. Chunking keeps the temporary
    byte-offset matrix small for cohorts with thousands of samples.
    """
    if type_size not in _INT_UNSIGNED_DTYPES:
        raise ValueError(f"Unsupported GT integer width in BCF FORMAT/GT: {type_size}")
    n_sel = len(sample_index_array)
    if n_vals < 1:
        if return_dosage:
            return np.empty((n_records, n_sel), dtype=np.int8)
        return np.empty((n_records, n_sel, 0), dtype=np.int8)
    if n_vals not in (1, 2):
        raise ValueError("BCFReader currently supports haploid or diploid GT fields only.")

    dtype = _INT_UNSIGNED_DTYPES[type_size]
    gt_starts = indiv_offsets + gt_data_rel_offset
    raw_bytes = np.frombuffer(data, dtype=np.uint8)

    all_samples = (
        n_sel == n_samples
        and sample_index_array.dtype.kind in "iu"
        and np.array_equal(sample_index_array, np.arange(n_samples, dtype=sample_index_array.dtype))
    )
    sample_stride = n_vals * type_size
    if all_samples:
        rel_byte_offsets = np.arange(n_samples * sample_stride, dtype=np.int64)
        decode_samples = n_samples
    else:
        within_sample = np.arange(sample_stride, dtype=np.int64)
        rel_byte_offsets = (
            sample_index_array.astype(np.int64, copy=False)[:, None] * sample_stride
            + within_sample[None, :]
        ).ravel()
        decode_samples = n_sel

    if return_dosage:
        out = np.empty((n_records, n_sel), dtype=np.int8)
    else:
        out = np.empty((n_records, n_sel, 2), dtype=np.int8)

    # Keep the int64 offset matrix under roughly 64 MiB per chunk. The gathered
    # byte buffer is smaller, so this cap controls peak temporary memory.
    max_offset_bytes = 64 * 1024 * 1024
    records_per_chunk = max(1, max_offset_bytes // max(1, rel_byte_offsets.size * np.dtype(np.int64).itemsize))

    for start in range(0, n_records, records_per_chunk):
        stop = min(start + records_per_chunk, n_records)
        byte_offsets = gt_starts[start:stop, None] + rel_byte_offsets[None, :]
        gathered = raw_bytes[byte_offsets.ravel()]

        if type_size == 1:
            raw = gathered.reshape(stop - start, decode_samples, n_vals)
        else:
            raw = np.frombuffer(gathered.tobytes(), dtype=dtype).reshape(stop - start, decode_samples, n_vals)

        # Decode: BCF GT encoding is (allele_index + 1) << 1 | phase.
        decoded = (raw.astype(np.int16, copy=False) >> 1) - 1

        if n_vals == 1:
            if return_dosage:
                out[start:stop] = decoded[:, :, 0].astype(np.int8, copy=False)
            else:
                chunk = out[start:stop]
                chunk[:, :, 0] = decoded[:, :, 0].astype(np.int8, copy=False)
                chunk[:, :, 1] = -1
            continue

        if return_dosage:
            chunk_missing_as_haploid = None
            if missing_as_haploid is not None:
                chunk_missing_as_haploid = missing_as_haploid
                if not np.isscalar(missing_as_haploid):
                    chunk_missing_as_haploid = np.asarray(missing_as_haploid)[start:stop]
            out[start:stop] = sum_diploid_genotypes(
                decoded,
                missing_as_haploid=chunk_missing_as_haploid,
            )
        else:
            _raise_if_unphased_bcf_gt(raw.reshape(-1, n_vals), decoded.reshape(-1, n_vals))
            out[start:stop] = decoded.astype(np.int8, copy=False)

    return out


def _batch_decode_gp(
    data: bytes,
    indiv_offsets: np.ndarray,
    gp_data_rel_offset: int,
    n_vals: int,
    n_samples: int,
    n_records: int,
    sample_index_array: np.ndarray,
) -> np.ndarray:
    """Batch-decode GP data for all records using vectorized numpy operations."""
    if n_vals < 1:
        return np.empty((n_records, len(sample_index_array), 0), dtype=np.float32)

    gp_bytes_per_record = n_samples * n_vals * 4  # float32
    gp_starts = indiv_offsets + gp_data_rel_offset

    byte_offsets_per_sample = np.arange(gp_bytes_per_record, dtype=np.int64)
    all_byte_offsets = gp_starts[:, None] + byte_offsets_per_sample[None, :]

    raw_bytes = np.frombuffer(data, dtype=np.uint8)
    gathered = raw_bytes[all_byte_offsets.ravel()]

    raw = np.frombuffer(gathered.tobytes(), dtype=np.dtype("<u4")).reshape(n_records, n_samples, n_vals)
    values = _decode_gp_raw(raw)
    return values[:, sample_index_array, :]


def _vectorized_qual(qual_raw: np.ndarray) -> np.ndarray:
    """Convert raw uint32 qual values to float32, handling BCF missing sentinel."""
    result = np.empty(len(qual_raw), dtype=np.float32)
    missing_mask = qual_raw == _FLOAT_MISSING
    # Reinterpret the uint32 bits as float32
    result[:] = np.frombuffer(qual_raw.tobytes(), dtype=np.float32)
    result[missing_mask] = np.nan
    return result


def _all_samples_selected(sample_index_array: np.ndarray, n_samples: int) -> bool:
    return (
        len(sample_index_array) == n_samples
        and sample_index_array.dtype.kind in "iu"
        and np.array_equal(sample_index_array, np.arange(n_samples, dtype=sample_index_array.dtype))
    )


@SNPBaseReader.register
class BCFReader(SNPBaseReader):
    def read(
        self,
        fields: Optional[Union[str, Sequence[str]]] = None,
        exclude_fields: Optional[Union[str, Sequence[str]]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        sample_idxs: Optional[Sequence[int]] = None,
        variant_ids: Optional[Sequence[str]] = None,
        variant_idxs: Optional[Sequence[int]] = None,
        region: Optional[str] = None,
        genotype_mode: GenotypeMode = "auto",
        chromosome_ploidy: Optional[str] = None,
    ) -> SNPObject:
        """
        Read a BCF file into a SNPObject.

        Args:
            fields: Fields to include. Supported fields are ``GT``, ``GP``, ``IID``,
                ``REF``, ``ALT``, ``#CHROM``, ``ID``, ``POS``, ``QUAL``,
                ``FILTER``, and ``INFO``. Use ``"*"`` to request the full set.
                If None, the default core fields are loaded.
            exclude_fields: Fields to exclude from the returned SNPObject.
            sample_ids: Sample IDs to read. If None and sample_idxs is None, all
                samples are read.
            sample_idxs: Sample indices to read. Negative indexes follow NumPy
                conventions.
            variant_ids: Variant identifiers to read. Matches BCF ``ID``,
                ``chrom:pos``, or ``chrom:pos:ref:alt``.
            variant_idxs: Variant indices to read. Negative indexes follow NumPy
                conventions.
            region: Optional genomic region, such as ``"22"`` or
                ``"22:100000-200000"``.
            genotype_mode: ``"dosage"`` sums the two allele indexes per sample
                (yielding ``0``, ``1``, or ``2`` for biallelic data).
                ``"phased"`` keeps phased allele columns separate and rejects
                unphased calls. ``"auto"`` (default) preserves phased calls and
                falls back to dosage for unphased calls.
            chromosome_ploidy:
                Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
                all selected variants should be treated as ordinary diploid/autosomal; this
                skips non-diploid chromosome checks and can be faster. The default None/"auto"
                preserves existing behavior.

        Returns:
            SNPObject: Object containing selected genotype, sample, and variant
            fields. ``GP`` is stored on ``SNPObject.calldata_gp`` when present.
        """
        if sample_idxs is not None and sample_ids is not None:
            raise ValueError("Only one of sample_idxs and sample_ids can be specified.")
        if variant_idxs is not None and variant_ids is not None:
            raise ValueError("Only one of variant_idxs and variant_ids can be specified.")

        genotype_mode = normalize_genotype_mode(genotype_mode)
        chromosome_ploidy_mode = _normalize_chromosome_ploidy(chromosome_ploidy)
        if genotype_mode == "auto":
            try:
                return self.read(
                    fields=fields,
                    exclude_fields=exclude_fields,
                    sample_ids=sample_ids,
                    sample_idxs=sample_idxs,
                    variant_ids=variant_ids,
                    variant_idxs=variant_idxs,
                    region=region,
                    genotype_mode="phased",
                    chromosome_ploidy=chromosome_ploidy,
                )
            except ValueError as exc:
                if "Cannot read unphased BCF genotypes" not in str(exc):
                    raise
                return self.read(
                    fields=fields,
                    exclude_fields=exclude_fields,
                    sample_ids=sample_ids,
                    sample_idxs=sample_idxs,
                    variant_ids=variant_ids,
                    variant_idxs=variant_idxs,
                    region=region,
                    genotype_mode="dosage",
                    chromosome_ploidy=chromosome_ploidy,
                )
        return_dosage = genotype_mode == "dosage"
        detect_non_diploid = return_dosage and chromosome_ploidy_mode != "autosomal"

        selected_fields = _normalize_fields(fields, exclude_fields)
        region_filter = _parse_vcf_region(region)
        data, body_offset, header = _load_bcf_data(str(self.filename))
        file_samples = np.asarray(header.samples, dtype=object)
        sample_index_array = _resolve_sample_indices(file_samples, sample_ids, sample_idxs)

        has_filtering = (variant_ids is not None or variant_idxs is not None or region_filter is not None)

        if has_filtering:
            return self._read_filtered(
                data, body_offset, header, file_samples, sample_index_array,
                selected_fields, region_filter, variant_ids, variant_idxs, return_dosage,
                detect_non_diploid,
            )

        return self._read_all(
            data, body_offset, header, file_samples, sample_index_array,
            selected_fields, return_dosage, detect_non_diploid,
        )

    def _read_all(
        self,
        data: bytes,
        body_offset: int,
        header: _BCFHeader,
        file_samples: np.ndarray,
        sample_index_array: np.ndarray,
        selected_fields: list[str],
        return_dosage: bool,
        detect_non_diploid: bool,
    ) -> SNPObject:
        """Optimized bulk read of all records with no variant filtering."""
        if selected_fields == ["GT"]:
            gt_only = self._try_read_gt_only_all(
                data, body_offset, header, file_samples, sample_index_array, return_dosage,
                detect_non_diploid,
            )
            if gt_only is not None:
                return gt_only
        elif "GT" in selected_fields and set(selected_fields).issubset(_CORE_FIELDS):
            core = self._try_read_core_all(
                data, body_offset, header, file_samples, sample_index_array, selected_fields, return_dosage,
                detect_non_diploid,
            )
            if core is not None:
                return core

        # Pass 1: build record offset table and extract fixed fields
        record_offsets = _build_record_offsets(data, body_offset)
        n_records = len(record_offsets)

        if n_records == 0:
            return self._empty_snpobject(selected_fields, file_samples, sample_index_array, return_dosage)

        l_shared, l_indiv, contig_ids, positions, qual_raw, n_alleles, n_info_arr, n_fmt_arr = \
            _extract_fixed_fields(data, record_offsets)

        n_file_samples = len(file_samples)
        n_selected_samples = len(sample_index_array)

        # Vectorized chrom
        need_chrom = "#CHROM" in selected_fields
        variants_chrom = None
        if need_chrom:
            variants_chrom = np.empty(n_records, dtype=object)
            unique_contig_ids = np.unique(contig_ids)
            for cid in unique_contig_ids:
                mask = contig_ids == cid
                variants_chrom[mask] = header.contigs[int(cid)]

        # Vectorized pos
        variants_pos = positions if "POS" in selected_fields else None

        # Vectorized qual
        variants_qual = _vectorized_qual(qual_raw) if "QUAL" in selected_fields else None

        # Samples
        samples = file_samples[sample_index_array] if "IID" in selected_fields else None

        # Compute indiv offsets for GT/GP decode
        indiv_offsets = record_offsets + 8 + l_shared.astype(np.int64)

        need_gt = "GT" in selected_fields
        need_gp = "GP" in selected_fields
        genotypes = None
        calldata_gp = None
        non_diploid_chromosomes = None
        if need_gt and detect_non_diploid:
            non_diploid_chromosomes = _non_diploid_contig_mask_or_none(contig_ids, header)

        # Batch GT decode
        if need_gt and n_records > 0:
            if n_file_samples == 0 and np.all(n_fmt_arr == 0):
                if return_dosage:
                    genotypes = np.empty((n_records, 0), dtype=np.int8)
                else:
                    genotypes = np.empty((n_records, 0, 2), dtype=np.int8)
            else:
                # Probe the first record to determine GT layout
                first_n_fmt = int(n_fmt_arr[0])
                first_indiv_offset = int(indiv_offsets[0])
                gt_layout = _probe_gt_layout(data, first_indiv_offset, first_n_fmt, n_file_samples, header)
                if gt_layout is None:
                    raise ValueError("BCF FORMAT field does not contain GT for all selected records.")

                gt_data_rel_offset, gt_n_vals, gt_type_size, _ = gt_layout

                # Check if all records have uniform l_indiv (same FORMAT layout)
                uniform_indiv = np.all(l_indiv == l_indiv[0])

                if uniform_indiv:
                    genotypes = _batch_decode_gt(
                        data, indiv_offsets, gt_data_rel_offset, gt_n_vals, gt_type_size,
                        n_file_samples, n_records, sample_index_array, return_dosage,
                        missing_as_haploid=non_diploid_chromosomes,
                    )
                else:
                    # Fallback: per-record GT decode
                    if return_dosage:
                        genotypes = np.empty((n_records, n_selected_samples), dtype=np.int8)
                    else:
                        genotypes = np.empty((n_records, n_selected_samples, 2), dtype=np.int8)
                    for i in range(n_records):
                        cur_indiv = int(indiv_offsets[i])
                        cur_n_fmt = int(n_fmt_arr[i])
                        cur_gt_layout = _probe_gt_layout(data, cur_indiv, cur_n_fmt, n_file_samples, header)
                        if cur_gt_layout is None:
                            raise ValueError("BCF FORMAT field does not contain GT for all selected records.")
                        rel_off, nv, ts, _ = cur_gt_layout
                        gt = _decode_gt_array(
                            data,
                            cur_indiv + rel_off,
                            n_file_samples,
                            nv,
                            ts,
                            require_phase=not return_dosage,
                            phase_sample_idxs=sample_index_array,
                        )
                        gt = gt[sample_index_array]
                        if return_dosage:
                            missing_as_haploid = None
                            if non_diploid_chromosomes is not None:
                                missing_as_haploid = bool(non_diploid_chromosomes[i])
                            genotypes[i] = sum_diploid_genotypes(
                                gt,
                                missing_as_haploid=missing_as_haploid,
                            )
                        else:
                            genotypes[i] = gt

        # Batch GP decode
        if need_gp and n_records > 0:
            first_indiv_offset = int(indiv_offsets[0])
            first_n_fmt = int(n_fmt_arr[0])
            gp_layout = _probe_gp_layout(data, first_indiv_offset, first_n_fmt, n_file_samples, header)

            if gp_layout is not None:
                gp_data_rel_offset, gp_n_vals, _ = gp_layout
                uniform_indiv = np.all(l_indiv == l_indiv[0])

                if uniform_indiv:
                    calldata_gp = _batch_decode_gp(
                        data, indiv_offsets, gp_data_rel_offset, gp_n_vals,
                        n_file_samples, n_records, sample_index_array,
                    )
                else:
                    gp_rows: list[Optional[np.ndarray]] = [None] * n_records
                    gp_width = 0
                    for i in range(n_records):
                        cur_indiv = int(indiv_offsets[i])
                        cur_n_fmt = int(n_fmt_arr[i])
                        cur_gp = _probe_gp_layout(data, cur_indiv, cur_n_fmt, n_file_samples, header)
                        if cur_gp is not None:
                            rel_off, nv, _ = cur_gp
                            gp = _decode_gp_array(data, cur_indiv + rel_off, n_file_samples, nv)[sample_index_array]
                            gp_rows[i] = gp
                            gp_width = max(gp_width, gp.shape[1])
                    if gp_width > 0:
                        calldata_gp = self._pad_gp_rows(gp_rows, n_selected_samples, gp_width)

        # String fields: ID, REF, ALT, FILTER, INFO - must iterate per-record
        need_id = "ID" in selected_fields
        need_ref = "REF" in selected_fields
        need_alt = "ALT" in selected_fields
        need_filter = "FILTER" in selected_fields
        need_info = "INFO" in selected_fields
        need_strings = need_id or need_ref or need_alt or need_filter or need_info

        variants_id = np.empty(n_records, dtype=object) if need_id else None
        variants_ref = np.empty(n_records, dtype=object) if need_ref else None
        variants_alt = np.empty(n_records, dtype=object) if need_alt else None
        variants_filter_pass = np.empty(n_records, dtype=bool) if need_filter else None
        variants_info = np.empty(n_records, dtype=object) if need_info else None

        if need_strings:
            ref_cache = {}
            alt_cache = {}
            filters_dict = header.filters

            for i in range(n_records):
                base = int(record_offsets[i]) + 8
                cur_n_alleles = int(n_alleles[i])
                cur_n_info = int(n_info_arr[i])
                offset = base + 24

                # 1. ID field
                if need_id:
                    b = data[offset]
                    offset += 1
                    n_vals = b >> 4
                    if n_vals < 15:
                        end = offset + n_vals
                        if n_vals == 0:
                            variant_id = "."
                        elif n_vals == 1 and data[offset] == 46:
                            variant_id = "."
                        else:
                            val = data[offset:end]
                            idx = val.find(b"\0")
                            variant_id = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                        offset = end
                    else:
                        variant_id, offset = _read_typed_string(data, offset - 1)
                    variants_id[i] = variant_id if variant_id else "."
                else:
                    offset = _skip_typed_value_fast(data, offset)

                # 2. REF field
                if need_ref:
                    b = data[offset]
                    offset += 1
                    n_vals = b >> 4
                    if n_vals < 15:
                        end = offset + n_vals
                        val = data[offset:end]
                        ref = ref_cache.get(val)
                        if ref is None:
                            idx = val.find(b"\0")
                            ref = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                            ref_cache[val] = ref
                        offset = end
                    else:
                        ref, offset = _read_typed_string(data, offset - 1)
                    variants_ref[i] = ref
                else:
                    offset = _skip_typed_value_fast(data, offset)

                # 3. ALT field
                if need_alt:
                    if cur_n_alleles <= 1:
                        alt_str = ""
                    elif cur_n_alleles == 2:
                        b = data[offset]
                        offset += 1
                        n_vals = b >> 4
                        if n_vals < 15:
                            end = offset + n_vals
                            val = data[offset:end]
                            alt_str = alt_cache.get(val)
                            if alt_str is None:
                                idx = val.find(b"\0")
                                alt_str = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                                alt_cache[val] = alt_str
                            offset = end
                        else:
                            alt_str, offset = _read_typed_string(data, offset - 1)
                    else:
                        alts = []
                        for _ in range(cur_n_alleles - 1):
                            b = data[offset]
                            offset += 1
                            n_vals = b >> 4
                            if n_vals < 15:
                                end = offset + n_vals
                                val = data[offset:end]
                                alt = alt_cache.get(val)
                                if alt is None:
                                    idx = val.find(b"\0")
                                    alt = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                                    alt_cache[val] = alt
                                offset = end
                            else:
                                alt, offset = _read_typed_string(data, offset - 1)
                            alts.append(alt)
                        alt_str = ",".join(alts)
                    variants_alt[i] = alt_str
                else:
                    for _ in range(max(0, cur_n_alleles - 1)):
                        offset = _skip_typed_value_fast(data, offset)

                # 4. FILTER field
                if need_filter:
                    b = data[offset]
                    offset += 1
                    type_code = b & 0x0F
                    n_vals = b >> 4
                    if n_vals == 0:
                        filter_pass = True
                    elif n_vals == 1 and type_code == 1:
                        val = data[offset]
                        offset += 1
                        if val == 128:
                            filter_pass = False
                        elif val == 129:
                            filter_pass = True
                        else:
                            filter_name = filters_dict.get(val, str(val))
                            filter_pass = (filter_name == "PASS")
                    else:
                        filter_pass, offset = _parse_filter_pass(data, offset - 1, header)
                    variants_filter_pass[i] = filter_pass
                elif need_info:
                    offset = _skip_typed_value_fast(data, offset)

                # 5. INFO field
                if need_info:
                    variants_info[i] = _parse_info_string(data, offset, cur_n_info, header)

        return SNPObject(
            genotypes=genotypes,
            calldata_gp=calldata_gp,
            samples=samples,
            variants_ref=variants_ref,
            variants_alt=variants_alt,
            variants_chrom=variants_chrom,
            variants_id=variants_id,
            variants_pos=variants_pos,
            variants_qual=variants_qual,
            variants_filter_pass=variants_filter_pass,
            variants_info=variants_info,
        )

    def _try_read_gt_only_all(
        self,
        data: bytes,
        body_offset: int,
        header: _BCFHeader,
        file_samples: np.ndarray,
        sample_index_array: np.ndarray,
        return_dosage: bool,
        detect_non_diploid: bool,
    ) -> Optional[SNPObject]:
        """Fast path for full-file genotype-only reads.

        The benchmarked BCF path requests only FORMAT/GT. In that case we can
        avoid vectorized extraction of POS/QUAL/INFO metadata and decode the
        individual sections directly.
        """
        if body_offset >= len(data):
            return self._empty_snpobject(["GT"], file_samples, sample_index_array, return_dosage)
        if detect_non_diploid and _header_has_non_diploid_contigs(header):
            return None

        first_l_shared, first_l_indiv = _U32_PAIR.unpack_from(data, body_offset)
        n_fmt_n_samples = _U32.unpack_from(data, body_offset + 8 + 20)[0]
        n_fmt = n_fmt_n_samples >> 24
        n_samples = n_fmt_n_samples & 0xFFFFFF
        if n_samples != len(file_samples):
            raise ValueError(
                f"BCF record sample count ({n_samples}) does not match header sample count "
                f"({len(file_samples)})."
            )
        if n_samples == 0 and n_fmt == 0:
            n_records = _count_records(data, body_offset)
            if return_dosage:
                genotypes = np.empty((n_records, 0), dtype=np.int8)
            else:
                genotypes = np.empty((n_records, 0, 2), dtype=np.int8)
            return SNPObject(genotypes=genotypes)

        first_indiv_offset = body_offset + 8 + first_l_shared
        gt_layout = _probe_gt_layout(data, first_indiv_offset, n_fmt, n_samples, header)
        if gt_layout is None:
            raise ValueError("BCF FORMAT field does not contain GT for all selected records.")

        gt_data_rel_offset, gt_n_vals, gt_type_size, _total_indiv_bytes = gt_layout

        try:
            from snputils.snp.io.read import _bcf
        except ImportError:
            _bcf = None

        if _bcf is not None:
            sample_arg = None if _all_samples_selected(sample_index_array, n_samples) else sample_index_array.tolist()
            decoded = _bcf.decode_gt(
                data,
                body_offset,
                gt_data_rel_offset,
                n_samples,
                gt_n_vals,
                gt_type_size,
                first_l_indiv,
                sample_arg,
                return_dosage,
            )
            if decoded is not None:
                gt_buffer, n_records = decoded
                genotypes = np.frombuffer(gt_buffer, dtype=np.int8)
                if return_dosage:
                    genotypes = genotypes.reshape(n_records, len(sample_index_array))
                else:
                    genotypes = genotypes.reshape(n_records, len(sample_index_array), 2)
                return SNPObject(genotypes=genotypes)

        indiv_offsets, uniform_indiv = _build_indiv_offsets(data, body_offset)
        n_records = len(indiv_offsets)
        if not uniform_indiv:
            return None

        genotypes = _batch_decode_gt(
            data, indiv_offsets, gt_data_rel_offset, gt_n_vals, gt_type_size,
            n_samples, n_records, sample_index_array, return_dosage,
        )
        return SNPObject(genotypes=genotypes)

    def _try_read_core_all(
        self,
        data: bytes,
        body_offset: int,
        header: _BCFHeader,
        file_samples: np.ndarray,
        sample_index_array: np.ndarray,
        selected_fields: list[str],
        return_dosage: bool,
        detect_non_diploid: bool,
    ) -> Optional[SNPObject]:
        """Fast path for full-file GT plus core variant metadata reads."""
        if body_offset >= len(data):
            return self._empty_snpobject(selected_fields, file_samples, sample_index_array, return_dosage)
        if detect_non_diploid and _header_has_non_diploid_contigs(header):
            return None

        first_l_shared, first_l_indiv = _U32_PAIR.unpack_from(data, body_offset)
        n_fmt_n_samples = _U32.unpack_from(data, body_offset + 8 + 20)[0]
        n_fmt = n_fmt_n_samples >> 24
        n_samples = n_fmt_n_samples & 0xFFFFFF
        if n_samples != len(file_samples):
            raise ValueError(
                f"BCF record sample count ({n_samples}) does not match header sample count "
                f"({len(file_samples)})."
            )
        if n_samples == 0 and n_fmt == 0:
            return None

        first_indiv_offset = body_offset + 8 + first_l_shared
        gt_layout = _probe_gt_layout(data, first_indiv_offset, n_fmt, n_samples, header)
        if gt_layout is None:
            raise ValueError("BCF FORMAT field does not contain GT for all selected records.")

        try:
            from snputils.snp.io.read import _bcf
        except ImportError:
            return None

        gt_data_rel_offset, gt_n_vals, gt_type_size, _total_indiv_bytes = gt_layout
        sample_arg = None if _all_samples_selected(sample_index_array, n_samples) else sample_index_array.tolist()
        pass_filter_id = next((idx for idx, name in header.filters.items() if name == "PASS"), -1)
        decoded = _bcf.decode_core(
            data,
            body_offset,
            gt_data_rel_offset,
            n_samples,
            gt_n_vals,
            gt_type_size,
            first_l_indiv,
            sample_arg,
            return_dosage,
            pass_filter_id,
        )
        if decoded is None:
            return None

        (
            gt_buffer,
            chrom_buffer,
            pos_buffer,
            qual_buffer,
            filter_buffer,
            ids,
            refs,
            alts,
            n_records,
        ) = decoded

        genotypes = np.frombuffer(gt_buffer, dtype=np.int8)
        if return_dosage:
            genotypes = genotypes.reshape(n_records, len(sample_index_array))
        else:
            genotypes = genotypes.reshape(n_records, len(sample_index_array), 2)

        variants_chrom = None
        if "#CHROM" in selected_fields:
            contig_ids = np.frombuffer(chrom_buffer, dtype=np.dtype("<i4"))
            variants_chrom = np.empty(n_records, dtype=object)
            for cid in np.unique(contig_ids):
                variants_chrom[contig_ids == cid] = header.contigs[int(cid)]

        variants_pos = (
            np.frombuffer(pos_buffer, dtype=np.dtype("<i8"))
            if "POS" in selected_fields
            else None
        )
        variants_qual = (
            _vectorized_qual(np.frombuffer(qual_buffer, dtype=np.dtype("<u4")))
            if "QUAL" in selected_fields
            else None
        )
        variants_filter_pass = (
            np.frombuffer(filter_buffer, dtype=np.bool_)
            if "FILTER" in selected_fields
            else None
        )

        return SNPObject(
            genotypes=genotypes,
            samples=file_samples[sample_index_array] if "IID" in selected_fields else None,
            variants_ref=np.asarray(refs, dtype=object) if "REF" in selected_fields else None,
            variants_alt=np.asarray(alts, dtype=object) if "ALT" in selected_fields else None,
            variants_chrom=variants_chrom,
            variants_id=np.asarray(ids, dtype=object) if "ID" in selected_fields else None,
            variants_pos=variants_pos,
            variants_qual=variants_qual,
            variants_filter_pass=variants_filter_pass,
        )

    def _read_filtered(
        self,
        data: bytes,
        body_offset: int,
        header: _BCFHeader,
        file_samples: np.ndarray,
        sample_index_array: np.ndarray,
        selected_fields: list[str],
        region_filter: Optional[Tuple[str, Optional[int], Optional[int]]],
        variant_ids: Optional[Sequence[str]],
        variant_idxs: Optional[Sequence[int]],
        return_dosage: bool,
        detect_non_diploid: bool,
    ) -> SNPObject:
        """Read with variant filtering - uses the original per-record approach."""
        n_records, selected_offsets, requested_offsets = _resolve_variant_request(
            data, body_offset, header, region_filter, variant_ids, variant_idxs,
        )
        if requested_offsets is not None:
            record_offsets_list = requested_offsets
        elif selected_offsets is not None:
            record_offsets_list = selected_offsets
        else:
            record_offsets_list = None

        if record_offsets_list is None:
            # No filtering was actually applied - redirect to fast path
            return self._read_all(data, body_offset, header, file_samples,
                                  sample_index_array, selected_fields, return_dosage,
                                  detect_non_diploid)

        n_selected_records = len(record_offsets_list)
        n_selected_samples = len(sample_index_array)
        n_file_samples = len(file_samples)

        samples = file_samples[sample_index_array] if "IID" in selected_fields else None

        if "GT" in selected_fields:
            if return_dosage:
                genotypes = np.empty((n_selected_records, n_selected_samples), dtype=np.int8)
            else:
                genotypes = np.empty((n_selected_records, n_selected_samples, 2), dtype=np.int8)
        else:
            genotypes = None

        gp_rows: Optional[List[Optional[np.ndarray]]] = [None] * n_selected_records if "GP" in selected_fields else None
        gp_width = 0

        variants_ref = np.empty(n_selected_records, dtype=object) if "REF" in selected_fields else None
        variants_alt = np.empty(n_selected_records, dtype=object) if "ALT" in selected_fields else None
        variants_chrom = np.empty(n_selected_records, dtype=object) if "#CHROM" in selected_fields else None
        variants_id = np.empty(n_selected_records, dtype=object) if "ID" in selected_fields else None
        variants_pos = np.empty(n_selected_records, dtype=np.int64) if "POS" in selected_fields else None
        variants_qual = np.empty(n_selected_records, dtype=np.float32) if "QUAL" in selected_fields else None
        variants_filter_pass = np.empty(n_selected_records, dtype=bool) if "FILTER" in selected_fields else None
        variants_info = np.empty(n_selected_records, dtype=object) if "INFO" in selected_fields else None

        need_strings = any(field in selected_fields for field in ("ID", "REF", "ALT", "FILTER", "INFO"))
        need_info = "INFO" in selected_fields
        need_filter = "FILTER" in selected_fields
        need_gt = "GT" in selected_fields
        need_gp = "GP" in selected_fields
        need_id = variants_id is not None
        need_ref = variants_ref is not None
        need_alt = variants_alt is not None
        ref_cache = {}
        alt_cache = {}
        filters_dict = header.filters

        for out_idx, record_offset in enumerate(record_offsets_list):
            l_shared = _U32.unpack_from(data, record_offset)[0]
            l_indiv = _U32.unpack_from(data, record_offset + 4)[0]
            base = record_offset + 8
            indiv_offset = base + l_shared

            contig_id = _I32.unpack_from(data, base)[0]
            pos = _I32.unpack_from(data, base + 4)[0] + 1
            n_alleles = _U32.unpack_from(data, base + 16)[0] >> 16
            n_info = _U32.unpack_from(data, base + 16)[0] & 0xFFFF
            n_fmt = _U32.unpack_from(data, base + 20)[0] >> 24
            n_samples = _U32.unpack_from(data, base + 20)[0] & 0xFFFFFF
            missing_as_haploid = None
            if detect_non_diploid:
                missing_as_haploid = (
                    _normalized_non_diploid_chromosome(header.contigs[contig_id]) is not None
                )

            if n_samples != n_file_samples:
                raise ValueError(
                    f"BCF record sample count ({n_samples}) does not match header sample count "
                    f"({n_file_samples})."
                )

            if variants_chrom is not None:
                variants_chrom[out_idx] = header.contigs[contig_id]
            if variants_pos is not None:
                variants_pos[out_idx] = pos
            if variants_qual is not None:
                variants_qual[out_idx] = _variant_qual(data, base)

            if need_strings:
                offset = base + 24

                # 1. ID field
                if need_id:
                    b = data[offset]
                    offset += 1
                    n_vals = b >> 4
                    if n_vals < 15:
                        end = offset + n_vals
                        if n_vals == 0:
                            variant_id = "."
                        elif n_vals == 1 and data[offset] == 46:
                            variant_id = "."
                        else:
                            val = data[offset:end]
                            idx = val.find(b"\0")
                            variant_id = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                        offset = end
                    else:
                        variant_id, offset = _read_typed_string(data, offset - 1)
                    variants_id[out_idx] = variant_id if variant_id else "."
                else:
                    offset = _skip_typed_value_fast(data, offset)

                # 2. REF field
                if need_ref:
                    b = data[offset]
                    offset += 1
                    n_vals = b >> 4
                    if n_vals < 15:
                        end = offset + n_vals
                        val = data[offset:end]
                        ref = ref_cache.get(val)
                        if ref is None:
                            idx = val.find(b"\0")
                            ref = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                            ref_cache[val] = ref
                        offset = end
                    else:
                        ref, offset = _read_typed_string(data, offset - 1)
                    variants_ref[out_idx] = ref
                else:
                    offset = _skip_typed_value_fast(data, offset)

                # 3. ALT field
                if need_alt:
                    if n_alleles <= 1:
                        alt_str = ""
                    elif n_alleles == 2:
                        b = data[offset]
                        offset += 1
                        n_vals = b >> 4
                        if n_vals < 15:
                            end = offset + n_vals
                            val = data[offset:end]
                            alt_str = alt_cache.get(val)
                            if alt_str is None:
                                idx = val.find(b"\0")
                                alt_str = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                                alt_cache[val] = alt_str
                            offset = end
                        else:
                            alt_str, offset = _read_typed_string(data, offset - 1)
                    else:
                        alts = []
                        for _ in range(n_alleles - 1):
                            b = data[offset]
                            offset += 1
                            n_vals = b >> 4
                            if n_vals < 15:
                                end = offset + n_vals
                                val = data[offset:end]
                                alt = alt_cache.get(val)
                                if alt is None:
                                    idx = val.find(b"\0")
                                    alt = val[:idx].decode("utf-8") if idx != -1 else val.decode("utf-8")
                                    alt_cache[val] = alt
                                offset = end
                            else:
                                alt, offset = _read_typed_string(data, offset - 1)
                            alts.append(alt)
                        alt_str = ",".join(alts)
                    variants_alt[out_idx] = alt_str
                else:
                    for _ in range(max(0, n_alleles - 1)):
                        offset = _skip_typed_value_fast(data, offset)

                # 4. FILTER field
                if need_filter:
                    b = data[offset]
                    offset += 1
                    type_code = b & 0x0F
                    n_vals = b >> 4
                    if n_vals == 0:
                        filter_pass = True
                    elif n_vals == 1 and type_code == 1:
                        val = data[offset]
                        offset += 1
                        if val == 128:
                            filter_pass = False
                        elif val == 129:
                            filter_pass = True
                        else:
                            filter_name = filters_dict.get(val, str(val))
                            filter_pass = (filter_name == "PASS")
                    else:
                        filter_pass, offset = _parse_filter_pass(data, offset - 1, header)
                    variants_filter_pass[out_idx] = filter_pass
                elif need_info:
                    offset = _skip_typed_value_fast(data, offset)

                # 5. INFO field
                if need_info:
                    variants_info[out_idx] = _parse_info_string(data, offset, n_info, header)

            if not (need_gt or need_gp):
                continue
            if l_indiv == 0:
                if need_gt and n_samples > 0:
                    raise ValueError("BCF FORMAT field does not contain GT for all selected records.")
                continue

            format_offset = indiv_offset
            gt_seen = False
            for _ in range(n_fmt):
                fmt_idx, format_offset = _read_scalar_typed_int(data, format_offset)
                n_vals, type_code, type_size, values_offset = _read_typed_descriptor(data, format_offset)
                key = header.formats.get(fmt_idx, {"ID": f"FORMAT_{fmt_idx}"})["ID"]
                values_nbytes = n_samples * n_vals * type_size

                if key == "GT" and need_gt:
                    gt = _decode_gt_array(
                        data,
                        values_offset,
                        n_samples,
                        n_vals,
                        type_size,
                        require_phase=not return_dosage,
                        phase_sample_idxs=sample_index_array,
                    )
                    gt = gt[sample_index_array]
                    if return_dosage:
                        genotypes[out_idx] = sum_diploid_genotypes(
                            gt,
                            missing_as_haploid=missing_as_haploid,
                        )
                    else:
                        genotypes[out_idx] = gt
                    gt_seen = True
                elif key == "GP" and need_gp:
                    if type_code != 5 or type_size != 4:
                        raise ValueError("BCF FORMAT/GP is expected to be stored as float32 values.")
                    gp = _decode_gp_array(data, values_offset, n_samples, n_vals)[sample_index_array]
                    gp_rows[out_idx] = gp
                    gp_width = max(gp_width, gp.shape[1])

                format_offset = values_offset + values_nbytes

            if need_gt and not gt_seen:
                raise ValueError("BCF FORMAT field does not contain GT for all selected records.")

        calldata_gp = None
        if gp_rows is not None and gp_width > 0:
            calldata_gp = self._pad_gp_rows(gp_rows, n_selected_samples, gp_width)

        return SNPObject(
            genotypes=genotypes,
            calldata_gp=calldata_gp,
            samples=samples,
            variants_ref=variants_ref,
            variants_alt=variants_alt,
            variants_chrom=variants_chrom,
            variants_id=variants_id,
            variants_pos=variants_pos,
            variants_qual=variants_qual,
            variants_filter_pass=variants_filter_pass,
            variants_info=variants_info,
        )

    @staticmethod
    def _pad_gp_rows(
        gp_rows: List[Optional[np.ndarray]],
        n_selected_samples: int,
        gp_width: int,
    ) -> np.ndarray:
        padded_rows = []
        for row in gp_rows:
            if row is None:
                padded_rows.append(np.full((n_selected_samples, gp_width), np.nan, dtype=np.float32))
                continue
            if row.shape[1] == gp_width:
                padded_rows.append(row)
                continue
            padded = np.full((n_selected_samples, gp_width), np.nan, dtype=np.float32)
            padded[:, : row.shape[1]] = row
            padded_rows.append(padded)
        return np.stack(padded_rows, axis=0) if padded_rows else np.empty((0, n_selected_samples, gp_width), dtype=np.float32)

    @staticmethod
    def _empty_snpobject(
        selected_fields: list[str],
        file_samples: np.ndarray,
        sample_index_array: np.ndarray,
        return_dosage: bool,
    ) -> SNPObject:
        n_sel = len(sample_index_array)
        return SNPObject(
            genotypes=np.empty((0, n_sel) if return_dosage else (0, n_sel, 2), dtype=np.int8) if "GT" in selected_fields else None,
            calldata_gp=None,
            samples=file_samples[sample_index_array] if "IID" in selected_fields else None,
            variants_ref=np.empty(0, dtype=object) if "REF" in selected_fields else None,
            variants_alt=np.empty(0, dtype=object) if "ALT" in selected_fields else None,
            variants_chrom=np.empty(0, dtype=object) if "#CHROM" in selected_fields else None,
            variants_id=np.empty(0, dtype=object) if "ID" in selected_fields else None,
            variants_pos=np.empty(0, dtype=np.int64) if "POS" in selected_fields else None,
            variants_qual=np.empty(0, dtype=np.float32) if "QUAL" in selected_fields else None,
            variants_filter_pass=np.empty(0, dtype=bool) if "FILTER" in selected_fields else None,
            variants_info=np.empty(0, dtype=object) if "INFO" in selected_fields else None,
        )
