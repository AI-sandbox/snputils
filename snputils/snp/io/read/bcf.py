from __future__ import annotations

import gzip
import logging
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from snputils.snp.io.read.vcf import _parse_vcf_region, _vcf_region_matches

log = logging.getLogger(__name__)

_DEFAULT_FIELDS = ["GT", "IID", "REF", "ALT", "#CHROM", "ID", "POS", "QUAL", "FILTER"]
_ALL_FIELDS = ["GT", "GP", "IID", "REF", "ALT", "#CHROM", "ID", "POS", "QUAL", "FILTER", "INFO"]

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


def _load_bcf_data(filename: Union[str, bytes]) -> Tuple[bytes, int, _BCFHeader]:
    with gzip.open(filename, "rb") as handle:
        data = handle.read()
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
    n_records = _count_records(data, body_offset)

    requested_variant_idxs = None
    if variant_idxs is not None:
        requested_variant_idxs = np.asarray(variant_idxs, dtype=int).ravel()
        if np.any((requested_variant_idxs < -n_records) | (requested_variant_idxs >= n_records)):
            raise ValueError("One or more variant indexes are out of bounds.")
        requested_variant_idxs = np.mod(requested_variant_idxs, n_records).tolist()

    if variant_ids is None and requested_variant_idxs is None and region_filter is None:
        return n_records, None, None

    requested_idx_set = None if requested_variant_idxs is None else set(requested_variant_idxs)
    requested_ids = None if variant_ids is None else {
        str(value) for value in np.asarray(variant_ids, dtype=object).ravel()
    }
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
        if passes and (region_filter is not None or requested_ids is not None):
            chrom, pos, variant_id, ref, alts = _decode_record_identifiers(data, offset, header)
            if region_filter is not None and not _vcf_region_matches(chrom, pos, region_filter):
                passes = False
            if passes and requested_ids is not None:
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
            "Cannot read unphased BCF genotypes with `sum_strands=False`; "
            "use `sum_strands=True` to load 0/1/2 genotype dosages."
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
    return np.frombuffer(
        data,
        dtype=np.dtype("<f4"),
        count=n_samples * n_vals,
        offset=offset,
    ).reshape(n_samples, n_vals).copy()


def _parse_filter_pass(
    data: bytes,
    offset: int,
    header: _BCFHeader,
) -> Tuple[bool, int]:
    filter_ids, offset = _read_int_list(data, offset)
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
    sum_strands: bool,
) -> np.ndarray:
    """Batch-decode GT data for all records using vectorized numpy operations.

    Instead of calling np.frombuffer per record, we gather all GT bytes into a
    single contiguous buffer and decode them all at once.
    """
    if type_size not in _INT_UNSIGNED_DTYPES:
        raise ValueError(f"Unsupported GT integer width in BCF FORMAT/GT: {type_size}")
    if n_vals < 1:
        n_sel = len(sample_index_array)
        if sum_strands:
            return np.empty((n_records, n_sel), dtype=np.int8)
        return np.empty((n_records, n_sel, 0), dtype=np.int8)

    dtype = _INT_UNSIGNED_DTYPES[type_size]
    gt_bytes_per_record = n_samples * n_vals * type_size

    # Build index array to gather all GT bytes from the raw buffer
    gt_starts = indiv_offsets + gt_data_rel_offset
    # Create a (n_records, gt_bytes_per_record) array of byte offsets
    byte_offsets_per_sample = np.arange(gt_bytes_per_record, dtype=np.int64)
    all_byte_offsets = gt_starts[:, None] + byte_offsets_per_sample[None, :]

    # Gather all bytes into a contiguous buffer
    raw_bytes = np.frombuffer(data, dtype=np.uint8)
    gathered = raw_bytes[all_byte_offsets.ravel()]

    # Reinterpret as the correct integer type
    raw = np.frombuffer(gathered.tobytes(), dtype=dtype).reshape(n_records, n_samples, n_vals)

    # Decode: BCF GT encoding is (allele_index + 1) << 1 | phase
    decoded = (raw.astype(np.int32, copy=False) >> 1) - 1

    if n_vals == 1:
        # Haploid: pad to diploid with -1
        padded = np.full((n_records, n_samples, 2), -1, dtype=np.int8)
        padded[:, :, 0] = decoded[:, :, 0].astype(np.int8, copy=False)
        selected = padded[:, sample_index_array, :]
    elif n_vals == 2:
        if not sum_strands:
            selected_raw = raw[:, sample_index_array, :]
            selected_decoded = decoded[:, sample_index_array, :]
            second_allele_called = selected_decoded[:, :, 1] >= 0
            second_allele_phased = (selected_raw[:, :, 1] & 1) != 0
            if np.any(second_allele_called & ~second_allele_phased):
                raise ValueError(
                    "Cannot read unphased BCF genotypes with `sum_strands=False`; "
                    "use `sum_strands=True` to load 0/1/2 genotype dosages."
                )
        selected = decoded[:, sample_index_array, :].astype(np.int8, copy=False)
    else:
        raise ValueError("BCFReader currently supports haploid or diploid GT fields only.")

    if sum_strands:
        return selected.sum(axis=2, dtype=np.int8)
    return selected


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

    raw = np.frombuffer(gathered.tobytes(), dtype=np.dtype("<f4")).reshape(n_records, n_samples, n_vals)
    return raw[:, sample_index_array, :].copy()


def _vectorized_qual(qual_raw: np.ndarray) -> np.ndarray:
    """Convert raw uint32 qual values to float32, handling BCF missing sentinel."""
    result = np.empty(len(qual_raw), dtype=np.float32)
    missing_mask = qual_raw == _FLOAT_MISSING
    # Reinterpret the uint32 bits as float32
    result[:] = np.frombuffer(qual_raw.tobytes(), dtype=np.float32)
    result[missing_mask] = np.nan
    return result


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
        sum_strands: bool = True,
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
            sum_strands: If True, sum the two diploid alleles per sample and
                return dosages in ``genotypes``. If False, keep the two allele
                columns separate; unphased GT calls are rejected because their
                allele order is not meaningful.

        Returns:
            SNPObject: Object containing selected genotype, sample, and variant
            fields. ``GP`` is stored on ``SNPObject.calldata_gp`` when present.
        """
        if sample_idxs is not None and sample_ids is not None:
            raise ValueError("Only one of sample_idxs and sample_ids can be specified.")
        if variant_idxs is not None and variant_ids is not None:
            raise ValueError("Only one of variant_idxs and variant_ids can be specified.")

        selected_fields = _normalize_fields(fields, exclude_fields)
        region_filter = _parse_vcf_region(region)
        data, body_offset, header = _load_bcf_data(str(self.filename))
        file_samples = np.asarray(header.samples, dtype=object)
        sample_index_array = _resolve_sample_indices(file_samples, sample_ids, sample_idxs)

        has_filtering = (variant_ids is not None or variant_idxs is not None or region_filter is not None)

        if has_filtering:
            return self._read_filtered(
                data, body_offset, header, file_samples, sample_index_array,
                selected_fields, region_filter, variant_ids, variant_idxs, sum_strands,
            )

        return self._read_all(
            data, body_offset, header, file_samples, sample_index_array,
            selected_fields, sum_strands,
        )

    def _read_all(
        self,
        data: bytes,
        body_offset: int,
        header: _BCFHeader,
        file_samples: np.ndarray,
        sample_index_array: np.ndarray,
        selected_fields: list[str],
        sum_strands: bool,
    ) -> SNPObject:
        """Optimized bulk read of all records with no variant filtering."""
        # Pass 1: build record offset table and extract fixed fields
        record_offsets = _build_record_offsets(data, body_offset)
        n_records = len(record_offsets)

        if n_records == 0:
            return self._empty_snpobject(selected_fields, file_samples, sample_index_array, sum_strands)

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

        # Batch GT decode
        if need_gt and n_records > 0:
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
                    n_file_samples, n_records, sample_index_array, sum_strands,
                )
            else:
                # Fallback: per-record GT decode
                if sum_strands:
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
                        require_phase=not sum_strands,
                        phase_sample_idxs=sample_index_array,
                    )
                    gt = gt[sample_index_array]
                    if sum_strands:
                        genotypes[i] = gt.sum(axis=1, dtype=np.int8)
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
                        if val == 128 or val == 129:
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
        sum_strands: bool,
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
                                  sample_index_array, selected_fields, sum_strands)

        n_selected_records = len(record_offsets_list)
        n_selected_samples = len(sample_index_array)
        n_file_samples = len(file_samples)

        samples = file_samples[sample_index_array] if "IID" in selected_fields else None

        if "GT" in selected_fields:
            if sum_strands:
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
                        if val == 128 or val == 129:
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

            if not (need_gt or need_gp) or l_indiv == 0:
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
                        require_phase=not sum_strands,
                        phase_sample_idxs=sample_index_array,
                    )
                    gt = gt[sample_index_array]
                    if sum_strands:
                        genotypes[out_idx] = gt.sum(axis=1, dtype=np.int8)
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
        sum_strands: bool,
    ) -> SNPObject:
        n_sel = len(sample_index_array)
        return SNPObject(
            genotypes=np.empty((0, n_sel) if sum_strands else (0, n_sel, 2), dtype=np.int8) if "GT" in selected_fields else None,
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
