import inspect
import logging
from typing import Optional, List, Any, Union, Iterable, Iterator, Tuple, Dict, Sequence
from pathlib import Path
import gzip
import csv
import mmap
import operator

import numpy as np
import polars as pl
from snputils._utils.genotypes import (
    ExplicitGenotypeMode,
    GenotypeMode,
    _MULTIALLELIC_DOSAGE_ERROR,
    normalize_genotype_mode,
    sum_diploid_alleles,
)
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
import pathlib 
log = logging.getLogger(__name__)


_POLARS_CSV_HAS_N_THREADS = "n_threads" in inspect.signature(pl.read_csv).parameters


_UNPHASED_VCF_PHASED_ERROR = (
    "Cannot read unphased VCF genotypes with genotype_mode='phased'; "
    "use genotype_mode='dosage' to load 0/1/2 genotype dosages."
)


def _raise_if_unphased_vcf_gt_separator(separators: np.ndarray) -> None:
    if np.any(separators == ord("/")):
        raise ValueError(_UNPHASED_VCF_PHASED_ERROR)


def _normalized_non_diploid_chromosome(chrom: Any) -> Optional[str]:
    value = str(chrom).strip().lower()
    if value.startswith("chrom"):
        value = value[5:]
    elif value.startswith("chr"):
        value = value[3:]
    elif value.startswith("chm"):
        value = value[3:]
    if value in {"x", "y", "m", "mt", "mitochondria", "mitochondrial", "mitochondrion"}:
        return value
    return None


def _non_diploid_chromosome_mask_or_none(chromosomes: np.ndarray) -> Optional[np.ndarray]:
    mask = np.fromiter(
        (_normalized_non_diploid_chromosome(chrom) is not None for chrom in chromosomes),
        dtype=bool,
        count=len(chromosomes),
    )
    return mask if np.any(mask) else None


def _check_assume_biallelic(value: bool) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("assume_biallelic must be a boolean.")


def _normalize_chromosome_ploidy(value: Optional[str]) -> str:
    if value is None:
        return "auto"

    value = str(value).strip().lower()
    if value not in {"auto", "autosomal", "mixed"}:
        raise ValueError(
            "chromosome_ploidy must be one of None, 'auto', 'autosomal', or 'mixed'."
        )
    return value


def _get_vcf_col_names_and_sep(vcf_path: str, separator: Optional[str] = None):
    """
    Get the column names and separator used in the VCF file.

    Args:
        vcf_path: The path to the VCF file.
        separator: Separator character. If None, the separator is automatically detected.

    Returns:
        col_names: List of column names.
        separator: Separator character.
    """
    vcf_path = Path(vcf_path)
    if vcf_path.suffixes[-2:] == ['.vcf', '.gz']:
        open_func = gzip.open
        mode = 'rt'
    elif vcf_path.suffixes[-2:] == ['.vcf', '.zst']:
        import zstandard as zstd
        open_func = lambda p, m: zstd.open(p, m, encoding="utf-8")
        mode = 'rt'
    elif vcf_path.suffix == '.vcf':
        open_func = open
        mode = 'r'
    else:
        raise ValueError(f"Unsupported file extension: {vcf_path.suffixes}")

    col_names = None
    with open_func(vcf_path, mode) as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("##"):
                continue
            if stripped_line.startswith("#CHROM") or stripped_line.startswith("CHROM"):
                if separator is None:
                    if "\t" in stripped_line:
                        separator = "\t"
                    else:
                        try:
                            separator = csv.Sniffer().sniff(stripped_line).delimiter
                        except csv.Error:
                            separator = "\t"
                col_names = [x.strip() for x in stripped_line.split(separator)]
                break

    if col_names is None:
        raise ValueError(
            "Could not find VCF header line. Expected a line starting with 'CHROM' or '#CHROM'."
        )

    return col_names, separator


def _open_vcf_binary(vcf_path: Union[str, pathlib.Path]):
    vcf_path = Path(vcf_path)
    if vcf_path.suffixes[-2:] == ['.vcf', '.gz']:
        return gzip.open(vcf_path, 'rb')
    if vcf_path.suffixes[-2:] == ['.vcf', '.zst']:
        import zstandard as zstd
        import io
        return io.BufferedReader(zstd.open(vcf_path, 'rb'))
    if vcf_path.suffix == '.vcf':
        return open(vcf_path, 'rb')
    raise ValueError(f"Unsupported file extension: {vcf_path.suffixes}")


def _vcf_header_columns(vcf_path: Union[str, pathlib.Path]) -> list[str]:
    with _open_vcf_binary(vcf_path) as file:
        for line in file:
            if line.startswith(b"##"):
                continue
            stripped = line.rstrip(b"\r\n")
            if stripped.startswith(b"#CHROM") or stripped.startswith(b"CHROM"):
                return [value.decode("utf-8") for value in stripped.split(b"\t")]
    raise ValueError("Could not find VCF header line. Expected a line starting with 'CHROM' or '#CHROM'.")


def _vcf_body_offset(data: Union[bytes, bytearray]) -> int:
    offset = 0
    while offset < len(data):
        line_end = data.find(b"\n", offset)
        if line_end < 0:
            line_end = len(data)
        content_end = line_end - 1 if line_end > offset and data[line_end - 1] == ord("\r") else line_end
        if data[offset:content_end].startswith((b"#CHROM", b"CHROM")):
            return min(len(data), line_end + 1)
        offset = line_end + 1
    raise ValueError("Could not find VCF header line. Expected a line starting with 'CHROM' or '#CHROM'.")


def _parse_vcf_region(region: Optional[str]) -> Optional[tuple[str, Optional[int], Optional[int]]]:
    if region is None:
        return None
    region = str(region).strip()
    if not region:
        raise ValueError("region must be non-empty.")
    if ":" not in region:
        return region, None, None

    chrom, interval = region.split(":", 1)
    interval = interval.replace(",", "")
    if not chrom or not interval:
        raise ValueError(f"Invalid VCF region: {region!r}.")
    if "-" in interval:
        start_text, end_text = interval.split("-", 1)
        start = int(start_text) if start_text else None
        end = int(end_text) if end_text else None
    else:
        start = int(interval)
        end = start
    if start is not None and start < 1:
        raise ValueError("VCF region start must be >= 1.")
    if end is not None and end < 1:
        raise ValueError("VCF region end must be >= 1.")
    if start is not None and end is not None and start > end:
        raise ValueError("VCF region start must be <= end.")
    return chrom, start, end


def _vcf_region_matches(
    chrom: str,
    pos: int,
    region_filter: Optional[tuple[str, Optional[int], Optional[int]]],
) -> bool:
    if region_filter is None:
        return True
    region_chrom, start, end = region_filter
    if chrom != region_chrom:
        return False
    if start is not None and pos < start:
        return False
    if end is not None and pos > end:
        return False
    return True


def _region_mask_from_raw(
    raw: np.ndarray,
    chrom_starts: np.ndarray,
    chrom_ends: np.ndarray,
    pos_starts: np.ndarray,
    pos_ends: np.ndarray,
    region_filter: Optional[tuple[str, Optional[int], Optional[int]]],
) -> np.ndarray:
    if region_filter is None:
        return np.ones(len(chrom_starts), dtype=bool)

    region_chrom, start, end = region_filter
    encoded_chrom = region_chrom.encode("utf-8")
    chrom_lengths = chrom_ends - chrom_starts
    mask = chrom_lengths == len(encoded_chrom)
    for offset, byte in enumerate(encoded_chrom):
        mask = mask & (raw[chrom_starts + offset] == byte)

    positions = _parse_ascii_ints(raw, pos_starts, pos_ends)
    if start is not None:
        mask = mask & (positions >= start)
    if end is not None:
        mask = mask & (positions <= end)
    return mask


def _count_vcf_records(
    vcf_path: Union[str, pathlib.Path],
    region_filter: Optional[tuple[str, Optional[int], Optional[int]]] = None,
    separator: Union[str, bytes] = "\t",
) -> int:
    separator_bytes = separator.encode("utf-8") if isinstance(separator, str) else separator
    if region_filter is None:
        n_records = 0
        saw_body_tail = False
        tail = b""
        with _open_vcf_binary(vcf_path) as file:
            for line in file:
                if line.startswith(b"#"):
                    continue
                n_records = 1
                break
            else:
                return 0

            while chunk := file.read(8 * 1024 * 1024):
                n_records += chunk.count(b"\n")
                saw_body_tail = True
                tail = chunk[-1:]

        if saw_body_tail and tail != b"\n":
            n_records += 1
        return n_records

    n_records = 0
    with _open_vcf_binary(vcf_path) as file:
        for line in file:
            if line and not line.startswith(b"#"):
                if region_filter is not None:
                    parts = line.split(separator_bytes, 2)
                    if len(parts) < 2:
                        raise ValueError("Malformed VCF record with fewer than 2 delimited fields.")
                    if not _vcf_region_matches(
                        parts[0].decode("utf-8"),
                        int(parts[1]),
                        region_filter,
                    ):
                        continue
                n_records += 1
    return n_records


def _first_record_is_fixed_width_gt_only(
    vcf_path: Union[str, pathlib.Path],
    n_samples_total: int,
) -> bool:
    with _open_vcf_binary(vcf_path) as file:
        for line in file:
            if line.startswith(b"#"):
                continue
            parts = line.rstrip(b"\r\n").split(b"\t", 9)
            if len(parts) < 10:
                return False
            return parts[8] == b"GT" and len(parts[9]) == n_samples_total * 4 - 1
    return True


def _decode_vcf_value(value: bytes) -> str:
    return value.decode("utf-8")


def _vcf_value_to_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_vcf_qual(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    out = np.empty(arr.shape[0], dtype=np.float32)
    for idx, value in enumerate(arr):
        value_str = _vcf_value_to_str(value)
        out[idx] = np.nan if value_str in ("", ".") else float(value_str)
    return out


def _normalize_vcf_filter_pass(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        return arr
    return np.fromiter((_vcf_value_to_str(value) == "PASS" for value in arr), dtype=bool, count=arr.size)


def _empty_genotype_array(n_variants: int, n_samples: int, return_dosage: bool) -> np.ndarray:
    if return_dosage:
        return np.empty((n_variants, n_samples), dtype=np.int8)
    return np.empty((n_variants, n_samples, 2), dtype=np.int8)


def _concat_axis0_releasing(chunks: list[np.ndarray]) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]

    # Avoid np.concatenate's full source-plus-destination peak: copy each
    # chunk into the final array and release that chunk reference immediately.
    shape = list(chunks[0].shape)
    shape[0] = sum(chunk.shape[0] for chunk in chunks)
    out = np.empty(tuple(shape), dtype=np.result_type(*(chunk.dtype for chunk in chunks)))
    offset = 0
    for idx, chunk in enumerate(chunks):
        height = chunk.shape[0]
        out[offset:offset + height] = chunk
        offset += height
        chunks[idx] = np.empty((0,) + chunk.shape[1:], dtype=chunk.dtype)
    return out


def _initial_stream_capacity(vcf_path: Union[str, pathlib.Path]) -> int:
    path = Path(vcf_path)
    if path.suffixes[-2:] in ([".vcf", ".gz"], [".vcf", ".zst"]):
        try:
            # A gzip stream does not expose row count cheaply. The compressed
            # byte size is a conservative starting point that avoids an extra
            # decompression pass for common 1KGP-style GT-only files.
            return max(65_536, int(path.stat().st_size // 128))
        except OSError:
            pass
    return 65_536


def _ensure_axis0_capacity(
    array: Optional[np.ndarray],
    used: int,
    additional: int,
    tail_shape: tuple[int, ...],
    dtype: np.dtype,
    initial_capacity: int,
) -> np.ndarray:
    required = used + additional
    if array is None:
        capacity = max(required, initial_capacity)
        return np.empty((capacity,) + tail_shape, dtype=dtype)
    if required <= array.shape[0]:
        return array

    capacity = max(required, array.shape[0] * 2)
    grown = np.empty((capacity,) + tail_shape, dtype=array.dtype)
    grown[:used] = array[:used]
    return grown


def _parse_vcf_qual_raw(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    lengths = ends - starts
    if starts.size == 0:
        return np.array([], dtype=np.float32)
    if np.all((lengths == 1) & (raw[starts] == ord("."))):
        return np.full(starts.shape[0], np.nan, dtype=np.float32)
    return np.fromiter(
        (
            np.nan
            if raw[int(start):int(end)].tobytes() in (b"", b".")
            else float(raw[int(start):int(end)].tobytes())
            for start, end in zip(starts, ends)
        ),
        dtype=np.float32,
        count=len(starts),
    )


def _vcf_filter_pass_raw(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    lengths = ends - starts
    out = lengths == 4
    if starts.size == 0:
        return out
    encoded = b"PASS"
    for offset, byte in enumerate(encoded):
        valid = lengths > offset
        compare = np.zeros_like(out)
        compare[valid] = raw[starts[valid] + offset] == byte
        out = out & compare
    return out


_FAST_DEFAULT_FIELDS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]


def _resolve_fast_vcf_columns(
    names: list[str],
    fields: Optional[list[str]],
    exclude_fields: Optional[list[str]],
    samples: Optional[Sequence[Union[str, int]]],
) -> tuple[list[str], list[str], np.ndarray]:
    first_sample_idx = next(
        (i for i, col in enumerate(names) if col not in ['#CHROM', 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']),
        len(names),
    )
    has_sample_columns = first_sample_idx < len(names)

    if fields is None:
        fields = list(_FAST_DEFAULT_FIELDS)
        if not has_sample_columns and "INFO" in names:
            fields.append("INFO")
        if exclude_fields is not None:
            excluded = set(exclude_fields)
            if "CHROM" in excluded or "#CHROM" in excluded:
                excluded.update(("CHROM", "#CHROM"))
            fields = [field for field in fields if field not in excluded]
        exclude_fields = None

    field_columns, sample_columns, _ = _extract_columns(
        names,
        fields,
        exclude_fields,
        None if samples is None else list(samples),
    )
    all_samples = names[first_sample_idx:]
    sample_to_idx = {sample: idx for idx, sample in enumerate(all_samples)}
    sample_idxs = np.asarray([sample_to_idx[sample] for sample in sample_columns], dtype=np.int64)
    return field_columns, sample_columns, sample_idxs


def _decode_vcf_slice(raw: np.ndarray, start: int, end: int) -> str:
    return raw[int(start):int(end)].tobytes().decode("utf-8")


def _line_content_end(raw: np.ndarray, end: int) -> int:
    return int(end) - 1 if end > 0 and raw[int(end) - 1] == ord("\r") else int(end)


def _vcf_body_bounds(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    newlines = np.flatnonzero(raw == ord("\n"))
    if raw.size and (newlines.size == 0 or newlines[-1] != raw.size - 1):
        newlines = np.concatenate((newlines, np.asarray([raw.size], dtype=newlines.dtype)))
    if newlines.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    starts = np.empty(newlines.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = newlines[:-1] + 1
    nonempty = starts < raw.size
    starts = starts[nonempty]
    ends = newlines[nonempty].astype(np.int64, copy=True)
    body = (raw[starts] != ord("#")) & (raw[starts] != ord("\n")) & (raw[starts] != ord("\r"))
    starts = starts[body]
    ends = ends[body]
    if ends.size:
        crlf = raw[ends - 1] == ord("\r")
        ends[crlf] -= 1
    return starts, ends


def _ascii_gt_to_int(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.int16, copy=False) - ord("0")
    out = out.astype(np.int8, copy=False)
    missing = (values == ord(".")) | (values == ord("-"))
    out[missing] = -1
    return out


# Streaming decodes one row at a time. Reuse the byte conversion for small arrays.
_GT_BYTE_TO_ALLELE = _ascii_gt_to_int(np.arange(256, dtype=np.uint8))


def _single_allele_byte(values: np.ndarray) -> np.ndarray:
    return ((values >= ord("0")) & (values <= ord("9"))) | (values == ord(".")) | (values == ord("-"))


def _parse_ascii_ints(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    lengths = (ends - starts).astype(np.int64, copy=False)
    if lengths.size == 0:
        return np.array([], dtype=np.int32)
    width = int(lengths.max())
    columns = np.arange(width, dtype=np.int64)
    indexes = starts[:, None] + columns[None, :]
    mask = columns[None, :] < lengths[:, None]
    digits = np.where(mask, raw[indexes] - ord("0"), 0).astype(np.int64, copy=False)
    powers = 10 ** np.maximum(lengths[:, None] - columns[None, :] - 1, 0)
    values = np.sum(np.where(mask, digits * powers, 0), axis=1, dtype=np.int64)
    return values.astype(np.int32, copy=False)


def _field_matches_value(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray, value: str) -> bool:
    encoded = value.encode("utf-8")
    lengths = ends - starts
    if not np.all(lengths == len(encoded)):
        return False
    for offset, byte in enumerate(encoded):
        if not np.all(raw[starts + offset] == byte):
            return False
    return True


def _field_is_gt_first(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> bool:
    lengths = ends - starts
    if not np.all(lengths >= 2):
        return False
    if not (np.all(raw[starts] == ord("G")) and np.all(raw[starts + 1] == ord("T"))):
        return False
    exact = lengths == 2
    with_subfields = (lengths > 2) & (raw[starts + 2] == ord(":"))
    return bool(np.all(exact | with_subfields))


def _decode_fields(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (_decode_vcf_slice(raw, start, end) for start, end in zip(starts, ends)),
        dtype=object,
        count=len(starts),
    )


def _decode_allele_fields(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if starts.size == 0:
        return np.array([], dtype="<U1")
    if np.all(ends - starts == 1):
        return np.frombuffer(raw[starts].tobytes(), dtype="S1").astype("U1")
    return _decode_fields(raw, starts, ends)


def _raise_if_multiallelic_alt_bytes(
    raw: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> None:
    """Check ALT spans already located by a VCF parser, avoiding another file pass."""
    lengths = ends - starts
    candidates = np.flatnonzero(lengths >= 3)
    for idx in candidates:
        if np.any(raw[int(starts[idx]):int(ends[idx])] == ord(",")):
            raise ValueError(_MULTIALLELIC_DOSAGE_ERROR)


def _first_tabs_by_line(block: bytes, line_starts: np.ndarray, n_tabs: int) -> np.ndarray:
    tabs = np.empty((len(line_starts), n_tabs), dtype=np.int64)
    for row, start in enumerate(line_starts):
        pos = int(start) - 1
        for col in range(n_tabs):
            pos = block.find(b"\t", pos + 1)
            if pos < 0:
                raise ValueError("VCF record has fewer tab-delimited columns than expected.")
            tabs[row, col] = pos
    return tabs


def _assign_constant_or_decode(
    raw: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    arrays: dict[str, Optional[np.ndarray]],
    constants: dict[str, str],
    field: str,
    lo: int,
    hi: int,
    n_records: int,
) -> None:
    value = constants[field]
    if arrays[field] is None and _field_matches_value(raw, starts, ends, value):
        return
    if arrays[field] is None:
        arrays[field] = np.empty(n_records, dtype=object)
        arrays[field][:lo] = value
    arrays[field][lo:hi] = _decode_fields(raw, starts, ends)


def _assign_allele_field(
    raw: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    arrays: dict[str, Optional[np.ndarray]],
    field: str,
    lo: int,
    hi: int,
    n_records: int,
) -> None:
    if arrays[field] is None:
        arrays[field] = np.empty(n_records, dtype="<U1")
    if arrays[field].dtype.kind == "U" and np.all(ends - starts == 1):
        arrays[field][lo:hi] = np.frombuffer(raw[starts].tobytes(), dtype="S1").astype("U1")
        return
    if arrays[field].dtype.kind == "U":
        previous = arrays[field]
        arrays[field] = np.empty(n_records, dtype=object)
        arrays[field][:lo] = previous[:lo].astype(object)
    arrays[field][lo:hi] = _decode_fields(raw, starts, ends)


def _finalize_string_array(
    arrays: dict[str, Optional[np.ndarray]],
    constants: dict[str, str],
    field: str,
    n_records: int,
) -> np.ndarray:
    if arrays[field] is not None:
        return arrays[field]
    value = constants[field]
    return np.full(n_records, value, dtype=f"<U{max(1, len(value))}")


def _fixed_diploid_gt_row(samples: bytes, n_samples: int) -> bool:
    if len(samples) != 4 * n_samples - 1:
        return False
    # Every expected boundary must be a tab, with no extra tabs inside a GT.
    n_tabs = n_samples - 1
    if samples[3::4].count(b"\t") != n_tabs or samples.count(b"\t") != n_tabs:
        return False
    separators = samples[1::4]
    return separators.count(b"|") + separators.count(b"/") == n_samples


def _parse_gt_sample_bytes(
    sample_bytes: bytes,
    *,
    format_value: Optional[bytes] = None,
    n_samples_total: int,
    sample_idxs: np.ndarray,
    all_samples: bool,
    return_dosage: bool,
    non_diploid_chromosome: Optional[bool] = None,
) -> np.ndarray:
    sample_bytes = sample_bytes.rstrip(b"\r\n")
    n_selected = int(sample_idxs.size)
    if n_selected == 0:
        return np.empty((0,), dtype=np.int8)

    if format_value in (None, b"GT") and sample_bytes:
        raw = np.frombuffer(sample_bytes, dtype=np.uint8)
        selection = slice(None) if all_samples else sample_idxs
        if _fixed_diploid_gt_row(sample_bytes, n_samples_total):
            if not return_dosage:
                _raise_if_unphased_vcf_gt_separator(raw[1::4][selection])
            first = _GT_BYTE_TO_ALLELE[raw[0::4][selection]]
            second = _GT_BYTE_TO_ALLELE[raw[2::4][selection]]
        else:
            tabs = np.flatnonzero(raw == ord("\t"))
            if tabs.size != n_samples_total - 1:
                raise ValueError("VCF record does not have the expected number of samples.")
            boundaries = np.empty(n_samples_total + 1, dtype=np.intp)
            boundaries[0] = -1
            boundaries[1:-1] = tabs
            boundaries[-1] = raw.size
            first, second = _decode_gt_fields(
                raw, boundaries[:-1][selection] + 1, boundaries[1:][selection],
                require_phase=not return_dosage,
            )
    else:
        gt_index = 0 if format_value is None else _format_gt_index(format_value)
        sample_fields = sample_bytes.split(b"\t")
        genotype = _parse_gt_row(
            (sample_fields[i] for i in sample_idxs), gt_index, require_phase=not return_dosage,
        )
        first, second = genotype.T
    if return_dosage:
        return sum_diploid_alleles(first, second, missing_as_haploid=non_diploid_chromosome)
    return np.stack((first, second), axis=-1)


def _format_gt_index(format_value: Union[str, bytes]) -> int:
    if isinstance(format_value, bytes):
        fields = format_value.rstrip(b"\r\n").split(b":")
        try:
            return fields.index(b"GT")
        except ValueError as exc:
            raise ValueError(f"VCF FORMAT field does not contain GT: {_decode_vcf_value(format_value)}") from exc
    fields = str(format_value).rstrip("\r\n").split(":")
    try:
        return fields.index("GT")
    except ValueError as exc:
        raise ValueError(f"VCF FORMAT field does not contain GT: {format_value}") from exc


def _format_gt_is_first(format_value: Any) -> bool:
    value = _vcf_value_to_str(format_value)
    return value == "GT" or value.startswith("GT:")


def _sample_gt_token(sample_field: Union[str, bytes], gt_index: int) -> Union[str, bytes]:
    separator = b":" if isinstance(sample_field, bytes) else ":"
    missing = b"." if isinstance(sample_field, bytes) else "."
    fields = sample_field.rstrip(b"\r\n").split(separator) if isinstance(sample_field, bytes) else str(sample_field).rstrip("\r\n").split(separator)
    if gt_index >= len(fields):
        return missing
    return fields[gt_index]


def _parse_gt_token(value: Union[str, bytes], *, require_phase: bool) -> tuple[int, int]:
    gt = _vcf_value_to_str(value)
    if "/" in gt:
        if require_phase:
            raise ValueError(_UNPHASED_VCF_PHASED_ERROR)
        gt = gt.replace("/", "|")
    # VCF 4.4 allows an explicit phase indicator before the first allele.
    first, _, second = gt.removeprefix("|").partition("|")
    missing = ("", ".", "-", "-1")
    first_allele = -1 if first in missing else int(first)
    second_allele = -1 if second in missing else int(second)
    return first_allele, second_allele


def _parse_gt_row(sample_fields: Iterable[Union[str, bytes]], gt_index: int, *, require_phase: bool) -> np.ndarray:
    calls = np.asarray([
        _parse_gt_token(_sample_gt_token(field, gt_index), require_phase=require_phase)
        for field in sample_fields
    ], dtype=np.int64)
    return calls.astype(_allele_dtype(int(calls.max())), copy=False)


def _allele_dtype(max_allele: int) -> np.dtype:
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        if max_allele <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise OverflowError("VCF allele index exceeds signed 64-bit range.")


def _widen_gt_buffer(buffer: np.ndarray, dtype: np.dtype, used: int) -> np.ndarray:
    widened = np.empty(buffer.shape, dtype=dtype)
    widened[:used] = buffer[:used]
    return widened


def _parse_gt_sample_matrix(
    sample_values: np.ndarray,
    format_values: np.ndarray,
    *,
    return_dosage: bool,
    non_diploid_chromosomes: Optional[np.ndarray] = None,
) -> np.ndarray:
    height, n_selected = sample_values.shape
    # A fourth character distinguishes a complete GT from a multi-digit allele.
    # Other FORMAT fields can contain Unicode, even in this four-character prefix.
    text = np.ascontiguousarray(sample_values, dtype="U4")
    prefix = text.view(np.uint32).reshape(height, n_selected, 4)
    sep = prefix[:, :, 1]
    simple = np.fromiter((_format_gt_is_first(value) for value in format_values), bool, count=height)
    diploid = ((sep == ord("|")) | (sep == ord("/"))) & _single_allele_byte(prefix[:, :, 2])
    gt_end = (prefix[:, :, 3] == 0) | (prefix[:, :, 3] == ord(":"))
    haploid = (sep == 0) | (sep == ord(":"))
    simple &= (_single_allele_byte(prefix[:, :, 0]) & ((diploid & gt_end) | haploid)).all(axis=1)
    genotype = np.empty((height, n_selected, 2), dtype=np.int8)
    if np.any(simple):
        fixed = prefix if simple.all() else prefix[simple]
        if not return_dosage:
            _raise_if_unphased_vcf_gt_separator(fixed[:, :, 1])
        genotype[simple, :, 0] = _ascii_gt_to_int(fixed[:, :, 0])
        genotype[simple, :, 1] = np.where(haploid[simple], -1, _ascii_gt_to_int(fixed[:, :, 2]))
    for row in np.flatnonzero(~simple):
        calls = _parse_gt_row(
            sample_values[row], _format_gt_index(format_values[row]), require_phase=not return_dosage,
        )
        if calls.itemsize > genotype.itemsize:
            genotype = genotype.astype(calls.dtype)
        genotype[row] = calls
    if return_dosage:
        return sum_diploid_alleles(
            genotype[:, :, 0], genotype[:, :, 1], missing_as_haploid=non_diploid_chromosomes,
        )
    return genotype


def _decode_gt_fields(
    raw: np.ndarray, starts: np.ndarray, ends: np.ndarray, *, require_phase: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode GT-only fields using their actual, exclusive end offsets."""
    widths = ends - starts
    if np.all(widths == 3):
        separators = raw[starts + 1]
        if np.all((separators == ord("|")) | (separators == ord("/"))):
            if require_phase:
                _raise_if_unphased_vcf_gt_separator(separators)
            return _ascii_gt_to_int(raw[starts]), _ascii_gt_to_int(raw[ends - 1])

    last = ends - 1
    separators = raw[np.minimum(starts + 1, last)]
    if require_phase:
        _raise_if_unphased_vcf_gt_separator(separators)
    diploid = (widths == 3) & ((separators == ord("|")) | (separators == ord("/")))
    first_bytes = raw[np.minimum(starts, last)]
    first = _ascii_gt_to_int(first_bytes)
    second = _ascii_gt_to_int(raw[last])
    second[~diploid] = -1
    # Single-byte haploids are complete. Parse complex GT tokens individually.
    haploid = (widths == 1) & _single_allele_byte(first_bytes)
    for idx in zip(*np.nonzero(~(diploid | haploid))):
        a, b = _parse_gt_token(raw[starts[idx]:ends[idx]].tobytes(), require_phase=require_phase)
        if max(a, b) > np.iinfo(first.dtype).max:
            dtype = _allele_dtype(max(a, b))
            first, second = first.astype(dtype), second.astype(dtype)
        first[idx], second[idx] = a, b
    return first, second


def _sample_field_bounds(
    tabs: np.ndarray, record_ends: np.ndarray, n_samples: int, sample_idxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_idxs.size == n_samples and np.array_equal(sample_idxs, np.arange(n_samples)):
        starts = tabs[:, 8:] + 1
        ends = np.empty_like(starts)
        ends[:, :-1] = tabs[:, 9:]
        ends[:, -1] = record_ends
        return starts, ends
    starts = tabs[:, 8 + sample_idxs] + 1
    last_sample = sample_idxs == n_samples - 1
    ends = np.empty_like(starts)
    ends[:, last_sample] = record_ends[:, None]
    ends[:, ~last_sample] = tabs[:, 9 + sample_idxs[~last_sample]]
    return starts, ends


def _write_gt_block(
    raw: np.ndarray,
    tabs: np.ndarray,
    ends: np.ndarray,
    n_samples: int,
    sample_idxs: np.ndarray,
    out: np.ndarray,
    offset: int,
    return_dosage: bool,
    non_diploid_chromosomes: Optional[np.ndarray],
) -> np.ndarray:
    if not _field_matches_value(raw, tabs[:, 7] + 1, tabs[:, 8], "GT"):
        raise ValueError("The block fast path requires GT-only sample fields.")

    # Bound the temporary arrays even for blocks containing millions of calls.
    rows_per_batch = max(1, 131_072 // sample_idxs.size)
    for lo in range(0, len(ends), rows_per_batch):
        hi = min(lo + rows_per_batch, len(ends))
        starts, field_ends = _sample_field_bounds(tabs[lo:hi], ends[lo:hi], n_samples, sample_idxs)
        first, second = _decode_gt_fields(raw, starts, field_ends, require_phase=not return_dosage)
        target = slice(offset + lo, offset + hi)
        if return_dosage:
            non_diploid = None if non_diploid_chromosomes is None else non_diploid_chromosomes[lo:hi]
            out[target] = sum_diploid_alleles(first, second, missing_as_haploid=non_diploid)
        else:
            if first.itemsize > out.itemsize:
                out = _widen_gt_buffer(out, first.dtype, offset + lo)
            out[target, :, 0] = first
            out[target, :, 1] = second
    return out


def _variant_id_matches(parts: list[bytes], wanted_variant_ids: set[str]) -> bool:
    variant_id = _decode_vcf_value(parts[2])
    if variant_id in wanted_variant_ids:
        return True
    generated = ":".join(
        (
            _decode_vcf_value(parts[0]),
            _decode_vcf_value(parts[1]),
            _decode_vcf_value(parts[3]),
            _decode_vcf_value(parts[4]),
        )
    )
    return generated in wanted_variant_ids


@SNPBaseReader.register
class VCFReader(SNPBaseReader):
    """
    Reads VCF files into an SNPObject with a NumPy parser optimized for GT columns.

    ``.vcf`` and ``.vcf.gz`` files with GT-only sample fields use a block
    parser that avoids materializing genotype strings in a DataFrame. Simple
    diploid FORMAT layouts such as ``GT:DP`` and ``DP:GT`` use a streaming byte
    parser. Other supported VCF layouts fall back to a pandas chunked parser.
    By default it reads the core variant fields ``CHROM``, ``POS``, ``ID``,
    ``REF``, ``ALT``, ``QUAL``, and ``FILTER``; pass ``fields="*"`` or include
    ``"INFO"`` when the INFO column is required.

    Supports reading sampleless (annotation-only) VCF files. In this case, 
    the returned `SNPObject` will have an empty `genotypes` array with a 
    variant axis (shape `(n_snps, 0)` or `(n_snps, 0, 2)`).
    """

    def __init__(self, filename: Union[str, pathlib.Path]):
        super().__init__(filename)

    def _make_snpobject(
        self,
        *,
        genotypes: np.ndarray,
        sample_columns: Sequence[str],
        arrays: dict[str, np.ndarray],
    ) -> SNPObject:
        variants_qual = _normalize_vcf_qual(arrays["QUAL"]) if "QUAL" in arrays else np.array([])
        variants_filter_pass = (
            _normalize_vcf_filter_pass(arrays["FILTER"])
            if "FILTER" in arrays
            else np.array([])
        )
        return SNPObject(
            genotypes=genotypes,
            samples=np.asarray(sample_columns),
            variants_ref=arrays.get("REF", np.array([])),
            variants_alt=arrays.get("ALT", np.array([])),
            variants_chrom=arrays.get("#CHROM", arrays.get("CHROM", np.array([]))),
            variants_filter_pass=variants_filter_pass,
            variants_id=arrays.get("ID", np.array([])),
            variants_pos=arrays.get("POS", np.array([])),
            variants_qual=variants_qual,
            variants_info=arrays.get("INFO", np.array([])),
        )

    def _read_parallel_gt_only(
        self,
        *,
        names: list[str],
        field_columns: list[str],
        sample_columns: list[str],
        sample_idxs: np.ndarray,
        return_dosage: bool,
        assume_biallelic: bool,
        threads: int,
    ) -> SNPObject:
        from snputils.snp.io.read import _vcf
        from snputils.snp.io.read.bcf import _read_bgzf_parallel

        data = _read_bgzf_parallel(self._filename, threads)
        body_offset = _vcf_body_offset(data)
        n_samples_total = max(0, len(names) - 9)
        if n_samples_total == 0:
            raw = np.frombuffer(data, dtype=np.uint8, offset=body_offset)
            body_starts, _ = _vcf_body_bounds(raw)
            n_records = int(body_starts.size)
            tabs_per_record = len(names) - 1
            tabs = np.flatnonzero(raw == ord("\t"))
            if tabs.size != n_records * tabs_per_record:
                raise ValueError("VCF records do not have the expected number of tab-delimited columns.")
            tabs = tabs.reshape(n_records, tabs_per_record)
            if return_dosage and not assume_biallelic:
                _raise_if_multiallelic_alt_bytes(raw, tabs[:, 3] + 1, tabs[:, 4])
            arrays = {}
            if "POS" in field_columns:
                arrays["POS"] = _parse_ascii_ints(raw, tabs[:, 0] + 1, tabs[:, 1])
            return self._make_snpobject(
                genotypes=_empty_genotype_array(n_records, 0, return_dosage),
                sample_columns=sample_columns,
                arrays=arrays,
            )

        all_samples = (
            len(sample_idxs) == n_samples_total
            and np.array_equal(sample_idxs, np.arange(n_samples_total, dtype=sample_idxs.dtype))
        )
        sample_arg = None if all_samples else sample_idxs.tolist()
        gt_buffer, pos_buffer, n_records = _vcf.decode_gt(
            data,
            body_offset,
            n_samples_total,
            sample_arg,
            return_dosage,
            threads,
            assume_biallelic,
        )
        genotypes = np.frombuffer(gt_buffer, dtype=np.int8)
        if return_dosage:
            genotypes = genotypes.reshape(n_records, len(sample_idxs))
        else:
            genotypes = genotypes.reshape(n_records, len(sample_idxs), 2)
        arrays = {}
        if "POS" in field_columns:
            arrays["POS"] = np.frombuffer(pos_buffer, dtype=np.int32)
        return self._make_snpobject(
            genotypes=genotypes,
            sample_columns=sample_columns,
            arrays=arrays,
        )

    def _read_mmap_gt_only(
        self,
        *,
        names: list[str],
        field_columns: list[str],
        sample_columns: list[str],
        sample_idxs: np.ndarray,
        return_dosage: bool,
        detect_non_diploid: bool,
        assume_biallelic: bool = False,
    ) -> SNPObject:
        if Path(self._filename).suffix != ".vcf":
            raise ValueError("The memory-mapped fast path only supports uncompressed .vcf files.")

        n_samples_total = len(names) - 9
        if n_samples_total < 0:
            raise ValueError("Malformed VCF header with fewer than 9 fixed columns.")

        include = set(field_columns)
        with open(self._filename, "rb") as file:
            mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                raw = np.frombuffer(mapped, dtype=np.uint8)
                body_starts, body_ends = _vcf_body_bounds(raw)
                n_records = int(body_starts.size)
                if n_records == 0:
                    return self._make_snpobject(
                        genotypes=_empty_genotype_array(0, len(sample_columns), return_dosage),
                        sample_columns=sample_columns,
                        arrays={},
                    )

                first_tabs = np.flatnonzero(raw[int(body_starts[0]):int(body_ends[0])] == ord("\t")) + int(body_starts[0])
                tabs_per_record = 8 + n_samples_total
                if first_tabs.size != tabs_per_record:
                    raise ValueError("VCF records do not have the expected number of tab-delimited columns.")
                format_value = _decode_vcf_slice(raw, first_tabs[7] + 1, first_tabs[8])
                if sample_columns and format_value != "GT":
                    raise ValueError("The memory-mapped fast path requires GT-only sample fields.")

                n_selected = len(sample_columns)
                genotypes = _empty_genotype_array(n_records, n_selected, return_dosage)

                arrays: dict[str, np.ndarray] = {}
                dynamic_arrays: dict[str, Optional[np.ndarray]] = {}
                constants: dict[str, str] = {}

                for field, start, end in (
                    ("#CHROM" if "#CHROM" in include else "CHROM", body_starts[0], first_tabs[0]),
                    ("ID", first_tabs[1] + 1, first_tabs[2]),
                    ("QUAL", first_tabs[4] + 1, first_tabs[5]),
                    ("FILTER", first_tabs[5] + 1, first_tabs[6]),
                    ("INFO", first_tabs[6] + 1, first_tabs[7]),
                ):
                    if field in include:
                        dynamic_arrays[field] = None
                        constants[field] = _decode_vcf_slice(raw, start, end)

                for field in ("REF", "ALT"):
                    if field in include:
                        dynamic_arrays[field] = None
                if "POS" in include:
                    arrays["POS"] = np.empty(n_records, dtype=np.int32)

                chunk_size = max(1, min(100_000, 2_000_000 // max(1, tabs_per_record)))

                for lo in range(0, n_records, chunk_size):
                    hi = min(n_records, lo + chunk_size)
                    byte_start = int(body_starts[lo])
                    byte_end = int(body_ends[hi - 1])
                    tabs = np.flatnonzero(raw[byte_start:byte_end] == ord("\t")) + byte_start
                    if tabs.size != (hi - lo) * tabs_per_record:
                        raise ValueError("VCF records do not all have the expected number of tab-delimited columns.")
                    tabs = tabs.reshape(hi - lo, tabs_per_record)

                    if return_dosage and not assume_biallelic:
                        _raise_if_multiallelic_alt_bytes(raw, tabs[:, 3] + 1, tabs[:, 4])
                    if n_selected:
                        non_diploid_chromosomes = (
                            _non_diploid_chromosome_mask_or_none(
                                _decode_fields(raw, body_starts[lo:hi], tabs[:, 0])
                            )
                            if detect_non_diploid
                            else None
                        )
                        genotypes = _write_gt_block(
                            raw, tabs, body_ends[lo:hi], n_samples_total, sample_idxs,
                            genotypes, lo, return_dosage, non_diploid_chromosomes,
                        )

                    if "POS" in include:
                        arrays["POS"][lo:hi] = _parse_ascii_ints(raw, tabs[:, 0] + 1, tabs[:, 1])
                    if "#CHROM" in include or "CHROM" in include:
                        chrom_field = "#CHROM" if "#CHROM" in include else "CHROM"
                        _assign_constant_or_decode(
                            raw,
                            body_starts[lo:hi],
                            tabs[:, 0],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field=chrom_field,
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "ID" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 1] + 1,
                            tabs[:, 2],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="ID",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "REF" in include:
                        _assign_allele_field(
                            raw,
                            tabs[:, 2] + 1,
                            tabs[:, 3],
                            arrays=dynamic_arrays,
                            field="REF",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "ALT" in include:
                        _assign_allele_field(
                            raw,
                            tabs[:, 3] + 1,
                            tabs[:, 4],
                            arrays=dynamic_arrays,
                            field="ALT",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "QUAL" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 4] + 1,
                            tabs[:, 5],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="QUAL",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "FILTER" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 5] + 1,
                            tabs[:, 6],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="FILTER",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "INFO" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 6] + 1,
                            tabs[:, 7],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="INFO",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )

                for field in dynamic_arrays:
                    arrays[field] = _finalize_string_array(dynamic_arrays, constants, field, n_records)

                return self._make_snpobject(
                    genotypes=genotypes,
                    sample_columns=sample_columns,
                    arrays=arrays,
                )
            finally:
                try:
                    del raw
                except UnboundLocalError:
                    pass
                mapped.close()

    def _read_block_gt_only_streaming(
        self,
        *,
        names: list[str],
        field_columns: list[str],
        sample_columns: list[str],
        sample_idxs: np.ndarray,
        region_filter: Optional[tuple[str, Optional[int], Optional[int]]],
        return_dosage: bool,
        assume_biallelic: bool,
        detect_non_diploid: bool,
    ) -> SNPObject:
        n_samples_total = len(names) - 9
        if n_samples_total < 0:
            raise ValueError("Malformed VCF header with fewer than 9 fixed columns.")

        include = set(field_columns)
        n_selected = len(sample_columns)
        tabs_per_record = 8 + n_samples_total
        # Wide blocks need a tab offset for every sample field.
        block_size = (1 if n_samples_total > 128 else 8) * 1024 * 1024

        # For gzip inputs we do not know n_records without a second pass. Keep
        # one growable genotype buffer and fill it directly, instead of keeping
        # all parsed genotype chunks and concatenating them at the end.
        gt_buffer: Optional[np.ndarray] = None
        gt_records = 0
        initial_gt_capacity = _initial_stream_capacity(self._filename)
        # Variant metadata is much smaller than the genotype matrix, so chunked
        # metadata assembly keeps this path simple without driving peak memory.
        array_chunks: dict[str, list[np.ndarray]] = {
            field: []
            for field in field_columns
            if field != "FORMAT"
        }

        with _open_vcf_binary(self._filename) as file:
            for line in file:
                if line.startswith(b"##"):
                    continue
                if line.startswith(b"#CHROM") or line.startswith(b"CHROM"):
                    break
                raise ValueError("Could not find VCF header line.")
            else:
                raise ValueError("Could not find VCF header line.")

            remainder = b""
            while True:
                chunk = file.read(block_size)
                if chunk:
                    # gzip reads can split records anywhere. Parse only complete
                    # lines in this block and carry the partial tail forward.
                    combined = remainder + chunk
                    cut = combined.rfind(b"\n")
                    if cut < 0:
                        remainder = combined
                        continue
                    block = combined[:cut + 1]
                    remainder = combined[cut + 1:]
                else:
                    block = remainder
                    remainder = b""

                if block:
                    raw = np.frombuffer(block, dtype=np.uint8)
                    line_ends = np.flatnonzero(raw == ord("\n"))
                    if raw.size and (line_ends.size == 0 or line_ends[-1] != raw.size - 1):
                        line_ends = np.concatenate((line_ends, np.asarray([raw.size], dtype=line_ends.dtype)))
                    line_starts = np.empty(line_ends.size, dtype=np.int64)
                    line_starts[0] = 0
                    line_starts[1:] = line_ends[:-1] + 1
                    content_ends = line_ends.astype(np.int64, copy=True)
                    if content_ends.size:
                        crlf = raw[content_ends - 1] == ord("\r")
                        content_ends[crlf] -= 1
                    height = int(line_ends.size)

                    if n_samples_total > 128 and not n_selected:
                        tabs = _first_tabs_by_line(block, line_starts, 9)
                    else:
                        tabs = np.flatnonzero(raw == ord("\t"))
                        if tabs.size != height * tabs_per_record:
                            raise ValueError("VCF records do not all have the expected number of tab-delimited columns.")
                        tabs = tabs.reshape(height, tabs_per_record)

                    if region_filter is not None:
                        keep = _region_mask_from_raw(
                            raw,
                            line_starts,
                            tabs[:, 0],
                            tabs[:, 0] + 1,
                            tabs[:, 1],
                            region_filter,
                        )
                        if not np.any(keep):
                            if not chunk:
                                break
                            continue
                        tabs = tabs[keep]
                        line_starts = line_starts[keep]
                        content_ends = content_ends[keep]
                        height = int(tabs.shape[0])

                    if return_dosage and not assume_biallelic:
                        _raise_if_multiallelic_alt_bytes(
                            raw,
                            tabs[:, 3] + 1,
                            tabs[:, 4],
                        )

                    if n_selected:
                        non_diploid_chromosomes = (
                            _non_diploid_chromosome_mask_or_none(
                                _decode_fields(raw, line_starts, tabs[:, 0])
                            )
                            if detect_non_diploid
                            else None
                        )
                        tail_shape = (n_selected,) if return_dosage else (n_selected, 2)
                        gt_buffer = _ensure_axis0_capacity(
                            gt_buffer, gt_records, height, tail_shape, np.int8, initial_gt_capacity,
                        )
                        gt_buffer = _write_gt_block(
                            raw, tabs, content_ends, n_samples_total, sample_idxs,
                            gt_buffer, gt_records, return_dosage, non_diploid_chromosomes,
                        )
                    gt_records += height

                    if "POS" in include:
                        array_chunks["POS"].append(_parse_ascii_ints(raw, tabs[:, 0] + 1, tabs[:, 1]))
                    if "#CHROM" in include or "CHROM" in include:
                        chrom_field = "#CHROM" if "#CHROM" in include else "CHROM"
                        array_chunks[chrom_field].append(_decode_fields(raw, line_starts, tabs[:, 0]))
                    if "ID" in include:
                        array_chunks["ID"].append(_decode_fields(raw, tabs[:, 1] + 1, tabs[:, 2]))
                    if "REF" in include:
                        array_chunks["REF"].append(_decode_allele_fields(raw, tabs[:, 2] + 1, tabs[:, 3]))
                    if "ALT" in include:
                        array_chunks["ALT"].append(_decode_allele_fields(raw, tabs[:, 3] + 1, tabs[:, 4]))
                    if "QUAL" in include:
                        array_chunks["QUAL"].append(_parse_vcf_qual_raw(raw, tabs[:, 4] + 1, tabs[:, 5]))
                    if "FILTER" in include:
                        array_chunks["FILTER"].append(_vcf_filter_pass_raw(raw, tabs[:, 5] + 1, tabs[:, 6]))
                    if "INFO" in include:
                        array_chunks["INFO"].append(_decode_fields(raw, tabs[:, 6] + 1, tabs[:, 7]))

                if not chunk:
                    break

        if gt_buffer is None:
            genotypes = _empty_genotype_array(gt_records, n_selected, return_dosage)
        else:
            genotypes = gt_buffer[:gt_records]

        arrays = {
            field: _concat_axis0_releasing(chunks)
            for field, chunks in array_chunks.items()
            if chunks
        }
        return self._make_snpobject(
            genotypes=genotypes,
            sample_columns=sample_columns,
            arrays=arrays,
        )

    def _read_simple_format_streaming(
        self,
        *,
        names: list[str],
        field_columns: list[str],
        sample_columns: list[str],
        sample_idxs: np.ndarray,
        region_filter: Optional[tuple[str, Optional[int], Optional[int]]],
        return_dosage: bool,
        assume_biallelic: bool,
        detect_non_diploid: bool,
    ) -> SNPObject:
        n_samples_total = len(names) - 9
        if n_samples_total < 0:
            raise ValueError("Malformed VCF header with fewer than 9 fixed columns.")

        include = set(field_columns)
        n_selected = len(sample_columns)
        tabs_per_record = 8 + n_samples_total
        if n_selected:
            max_sample_idx = int(sample_idxs.max())
            last_needed_tab = 8 + max_sample_idx
            if max_sample_idx < n_samples_total - 1:
                last_needed_tab = 9 + max_sample_idx
            n_tabs_needed = last_needed_tab + 1
        else:
            n_tabs_needed = 9
        block_size = (1 if n_samples_total > 128 and n_selected <= 16 else 8) * 1024 * 1024

        # Same growable output strategy as the GT-only path, but this parser
        # also handles simple FORMAT layouts where GT is first or follows DP.
        gt_buffer: Optional[np.ndarray] = None
        gt_records = 0
        initial_gt_capacity = _initial_stream_capacity(self._filename)
        # Metadata chunks are cheap relative to genotype calls and are released
        # during final axis-0 assembly.
        array_chunks: dict[str, list[np.ndarray]] = {
            field: []
            for field in field_columns
            if field != "FORMAT"
        }

        with _open_vcf_binary(self._filename) as file:
            for line in file:
                if line.startswith(b"##"):
                    continue
                if line.startswith(b"#CHROM") or line.startswith(b"CHROM"):
                    break
                raise ValueError("Could not find VCF header line.")
            else:
                raise ValueError("Could not find VCF header line.")

            remainder = b""
            while True:
                chunk = file.read(block_size)
                if chunk:
                    # Work on complete VCF records only; keep an incomplete line
                    # for the next read from the compressed stream.
                    combined = remainder + chunk
                    cut = combined.rfind(b"\n")
                    if cut < 0:
                        remainder = combined
                        continue
                    block = combined[:cut + 1]
                    remainder = combined[cut + 1:]
                else:
                    block = remainder
                    remainder = b""

                if block:
                    raw = np.frombuffer(block, dtype=np.uint8)
                    line_ends = np.flatnonzero(raw == ord("\n"))
                    if raw.size and (line_ends.size == 0 or line_ends[-1] != raw.size - 1):
                        line_ends = np.concatenate((line_ends, np.asarray([raw.size], dtype=line_ends.dtype)))
                    line_starts = np.empty(line_ends.size, dtype=np.int64)
                    line_starts[0] = 0
                    line_starts[1:] = line_ends[:-1] + 1
                    content_ends = line_ends.astype(np.int64, copy=True)
                    if content_ends.size:
                        crlf = raw[content_ends - 1] == ord("\r")
                        content_ends[crlf] -= 1
                    height = int(line_ends.size)

                    if n_samples_total <= 128:
                        tabs = np.flatnonzero(raw == ord("\t"))
                        if tabs.size != height * tabs_per_record:
                            raise ValueError("VCF records do not all have the expected number of tab-delimited columns.")
                        tabs = tabs.reshape(height, tabs_per_record)
                    else:
                        # Variable-width sample fields require tab positions up
                        # to the last selected sample, but not necessarily all
                        # samples in wide VCFs.
                        tabs = _first_tabs_by_line(block, line_starts, n_tabs_needed)

                    if region_filter is not None:
                        keep = _region_mask_from_raw(
                            raw,
                            line_starts,
                            tabs[:, 0],
                            tabs[:, 0] + 1,
                            tabs[:, 1],
                            region_filter,
                        )
                        if not np.any(keep):
                            if not chunk:
                                break
                            continue
                        tabs = tabs[keep]
                        line_starts = line_starts[keep]
                        content_ends = content_ends[keep]
                        height = int(tabs.shape[0])

                    if return_dosage and not assume_biallelic:
                        _raise_if_multiallelic_alt_bytes(
                            raw,
                            tabs[:, 3] + 1,
                            tabs[:, 4],
                        )

                    if n_selected:
                        non_diploid_chromosomes = (
                            _non_diploid_chromosome_mask_or_none(
                                _decode_fields(raw, line_starts, tabs[:, 0])
                            )
                            if detect_non_diploid
                            else None
                        )
                        # Keep this fallback vectorized by supporting only the
                        # simple layouts that give a fixed GT byte offset.
                        format_starts = tabs[:, 7] + 1
                        format_ends = tabs[:, 8]
                        if _field_is_gt_first(raw, format_starts, format_ends):
                            gt_offset = 0
                        elif _field_matches_value(raw, format_starts, format_ends, "DP:GT"):
                            gt_offset = 2
                        else:
                            raise ValueError("The simple FORMAT fast path requires GT first or DP:GT FORMAT fields.")

                        sample_starts, sample_ends = _sample_field_bounds(
                            tabs, content_ends, n_samples_total, sample_idxs,
                        )
                        gt_ends = sample_starts + gt_offset + 3
                        if np.any(gt_ends > sample_ends):
                            raise ValueError("The simple FORMAT fast path requires diploid GT calls.")
                        if gt_offset and not np.all(raw[sample_starts + gt_offset - 1] == ord(":")):
                            raise ValueError("The simple FORMAT fast path requires a fixed GT offset.")
                        has_suffix = gt_ends < sample_ends
                        if not np.all(raw[gt_ends[has_suffix]] == ord(":")):
                            raise ValueError("The simple FORMAT fast path requires single-digit GT alleles.")

                        sep = raw[sample_starts + gt_offset + 1]
                        if not np.all((sep == ord("|")) | (sep == ord("/"))):
                            raise ValueError("The simple FORMAT fast path requires diploid GT calls.")

                        first_allele = _ascii_gt_to_int(raw[sample_starts + gt_offset])
                        second_allele = _ascii_gt_to_int(raw[sample_starts + gt_offset + 2])
                        if return_dosage:
                            gt_buffer = _ensure_axis0_capacity(
                                gt_buffer,
                                gt_records,
                                height,
                                (n_selected,),
                                np.int8,
                                initial_gt_capacity,
                            )
                            gt_buffer[gt_records:gt_records + height] = sum_diploid_alleles(
                                first_allele,
                                second_allele,
                                missing_as_haploid=non_diploid_chromosomes,
                            )
                        else:
                            _raise_if_unphased_vcf_gt_separator(sep)
                            gt_buffer = _ensure_axis0_capacity(
                                gt_buffer,
                                gt_records,
                                height,
                                (n_selected, 2),
                                np.int8,
                                initial_gt_capacity,
                            )
                            gt_buffer[gt_records:gt_records + height, :, 0] = first_allele
                            gt_buffer[gt_records:gt_records + height, :, 1] = second_allele
                    gt_records += height

                    if "POS" in include:
                        array_chunks["POS"].append(_parse_ascii_ints(raw, tabs[:, 0] + 1, tabs[:, 1]))
                    if "#CHROM" in include or "CHROM" in include:
                        chrom_field = "#CHROM" if "#CHROM" in include else "CHROM"
                        array_chunks[chrom_field].append(_decode_fields(raw, line_starts, tabs[:, 0]))
                    if "ID" in include:
                        array_chunks["ID"].append(_decode_fields(raw, tabs[:, 1] + 1, tabs[:, 2]))
                    if "REF" in include:
                        array_chunks["REF"].append(_decode_allele_fields(raw, tabs[:, 2] + 1, tabs[:, 3]))
                    if "ALT" in include:
                        array_chunks["ALT"].append(_decode_allele_fields(raw, tabs[:, 3] + 1, tabs[:, 4]))
                    if "QUAL" in include:
                        array_chunks["QUAL"].append(_parse_vcf_qual_raw(raw, tabs[:, 4] + 1, tabs[:, 5]))
                    if "FILTER" in include:
                        array_chunks["FILTER"].append(_vcf_filter_pass_raw(raw, tabs[:, 5] + 1, tabs[:, 6]))
                    if "INFO" in include:
                        array_chunks["INFO"].append(_decode_fields(raw, tabs[:, 6] + 1, tabs[:, 7]))

                if not chunk:
                    break

        if gt_buffer is None:
            genotypes = _empty_genotype_array(gt_records, n_selected, return_dosage)
        else:
            genotypes = gt_buffer[:gt_records]

        arrays = {
            field: _concat_axis0_releasing(chunks)
            for field, chunks in array_chunks.items()
            if chunks
        }
        return self._make_snpobject(
            genotypes=genotypes,
            sample_columns=sample_columns,
            arrays=arrays,
        )

    def _read_block_gt_only(
        self,
        *,
        names: list[str],
        field_columns: list[str],
        sample_columns: list[str],
        sample_idxs: np.ndarray,
        region_filter: Optional[tuple[str, Optional[int], Optional[int]]],
        return_dosage: bool,
        assume_biallelic: bool,
        detect_non_diploid: bool,
    ) -> SNPObject:
        n_samples_total = len(names) - 9
        if n_samples_total < 0:
            raise ValueError("Malformed VCF header with fewer than 9 fixed columns.")

        include = set(field_columns)
        n_records = _count_vcf_records(self._filename, region_filter=region_filter, separator="\t")
        n_selected = len(sample_columns)
        genotypes = _empty_genotype_array(n_records, n_selected, return_dosage)

        if n_records == 0:
            return self._make_snpobject(
                genotypes=genotypes,
                sample_columns=sample_columns,
                arrays={},
            )

        with _open_vcf_binary(self._filename) as file:
            while True:
                line = file.readline()
                if not line:
                    raise ValueError("Could not find VCF header line.")
                if not line.startswith(b"##"):
                    break

            first_line = file.readline()
            if not first_line:
                return self._make_snpobject(
                    genotypes=genotypes,
                    sample_columns=sample_columns,
                        arrays={},
                    )

            first_raw = np.frombuffer(first_line.rstrip(b"\r\n"), dtype=np.uint8)
            first_tabs = np.flatnonzero(first_raw == ord("\t"))
            tabs_per_record = 8 + n_samples_total
            if first_tabs.size != tabs_per_record:
                raise ValueError("VCF records do not have the expected number of tab-delimited columns.")
            format_value = _decode_vcf_slice(first_raw, first_tabs[7] + 1, first_tabs[8])
            if sample_columns and format_value != "GT":
                raise ValueError("The block fast path requires GT-only sample fields.")

            arrays: dict[str, np.ndarray] = {}
            dynamic_arrays: dict[str, Optional[np.ndarray]] = {}
            constants: dict[str, str] = {}

            for field, start, end in (
                ("#CHROM" if "#CHROM" in include else "CHROM", 0, first_tabs[0]),
                ("ID", first_tabs[1] + 1, first_tabs[2]),
                ("QUAL", first_tabs[4] + 1, first_tabs[5]),
                ("FILTER", first_tabs[5] + 1, first_tabs[6]),
                ("INFO", first_tabs[6] + 1, first_tabs[7]),
            ):
                if field in include:
                    dynamic_arrays[field] = None
                    constants[field] = _decode_vcf_slice(first_raw, start, end)

            for field in ("REF", "ALT"):
                if field in include:
                    dynamic_arrays[field] = None
            if "POS" in include:
                arrays["POS"] = np.empty(n_records, dtype=np.int32)

            row = 0
            remainder = first_line
            # Wide blocks need a tab offset for every sample field.
            block_size = (1 if n_samples_total > 128 else 8) * 1024 * 1024

            while True:
                chunk = file.read(block_size)
                if chunk:
                    combined = remainder + chunk
                    cut = combined.rfind(b"\n")
                    if cut < 0:
                        remainder = combined
                        continue
                    block = combined[:cut + 1]
                    remainder = combined[cut + 1:]
                else:
                    block = remainder
                    remainder = b""

                if block:
                    raw = np.frombuffer(block, dtype=np.uint8)
                    line_ends = np.flatnonzero(raw == ord("\n"))
                    if raw.size and (line_ends.size == 0 or line_ends[-1] != raw.size - 1):
                        line_ends = np.concatenate((line_ends, np.asarray([raw.size], dtype=line_ends.dtype)))
                    line_starts = np.empty(line_ends.size, dtype=np.int64)
                    line_starts[0] = 0
                    line_starts[1:] = line_ends[:-1] + 1
                    content_ends = line_ends.astype(np.int64, copy=True)
                    if content_ends.size:
                        crlf = raw[content_ends - 1] == ord("\r")
                        content_ends[crlf] -= 1
                    height = int(line_ends.size)
                    lo = row
                    if region_filter is None and row + height > n_records:
                        raise ValueError("VCF contains more records than counted.")

                    if n_samples_total > 128 and not n_selected:
                        tabs = _first_tabs_by_line(block, line_starts, 9)
                    else:
                        tabs = np.flatnonzero(raw == ord("\t"))
                        if tabs.size != height * tabs_per_record:
                            raise ValueError("VCF records do not all have the expected number of tab-delimited columns.")
                        tabs = tabs.reshape(height, tabs_per_record)

                    if region_filter is not None:
                        keep = _region_mask_from_raw(
                            raw,
                            line_starts,
                            tabs[:, 0],
                            tabs[:, 0] + 1,
                            tabs[:, 1],
                            region_filter,
                        )
                        if not np.any(keep):
                            continue
                        tabs = tabs[keep]
                        line_starts = line_starts[keep]
                        content_ends = content_ends[keep]
                        height = int(tabs.shape[0])

                    hi = row + height
                    if hi > n_records:
                        raise ValueError("VCF contains more matching records than counted.")

                    if return_dosage and not assume_biallelic:
                        _raise_if_multiallelic_alt_bytes(
                            raw,
                            tabs[:, 3] + 1,
                            tabs[:, 4],
                        )

                    if n_selected:
                        non_diploid_chromosomes = (
                            _non_diploid_chromosome_mask_or_none(
                                _decode_fields(raw, line_starts, tabs[:, 0])
                            )
                            if detect_non_diploid
                            else None
                        )
                        genotypes = _write_gt_block(
                            raw, tabs, content_ends, n_samples_total, sample_idxs,
                            genotypes, lo, return_dosage, non_diploid_chromosomes,
                        )

                    if "POS" in include:
                        arrays["POS"][lo:hi] = _parse_ascii_ints(raw, tabs[:, 0] + 1, tabs[:, 1])
                    if "#CHROM" in include or "CHROM" in include:
                        chrom_field = "#CHROM" if "#CHROM" in include else "CHROM"
                        _assign_constant_or_decode(
                            raw,
                            line_starts,
                            tabs[:, 0],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field=chrom_field,
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "ID" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 1] + 1,
                            tabs[:, 2],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="ID",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "REF" in include:
                        _assign_allele_field(
                            raw,
                            tabs[:, 2] + 1,
                            tabs[:, 3],
                            arrays=dynamic_arrays,
                            field="REF",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "ALT" in include:
                        _assign_allele_field(
                            raw,
                            tabs[:, 3] + 1,
                            tabs[:, 4],
                            arrays=dynamic_arrays,
                            field="ALT",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "QUAL" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 4] + 1,
                            tabs[:, 5],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="QUAL",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "FILTER" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 5] + 1,
                            tabs[:, 6],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="FILTER",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    if "INFO" in include:
                        _assign_constant_or_decode(
                            raw,
                            tabs[:, 6] + 1,
                            tabs[:, 7],
                            arrays=dynamic_arrays,
                            constants=constants,
                            field="INFO",
                            lo=lo,
                            hi=hi,
                            n_records=n_records,
                        )
                    row = hi

                if not chunk:
                    break

            if row != n_records:
                raise ValueError("VCF contains fewer records than counted.")

        for field in dynamic_arrays:
            arrays[field] = _finalize_string_array(dynamic_arrays, constants, field, n_records)

        return self._make_snpobject(
            genotypes=genotypes,
            sample_columns=sample_columns,
            arrays=arrays,
        )

    def _read_pandas_chunks(
        self,
        *,
        names: list[str],
        field_columns: list[str],
        sample_columns: list[str],
        region_filter: Optional[tuple[str, Optional[int], Optional[int]]],
        return_dosage: bool,
        assume_biallelic: bool,
        detect_non_diploid: bool,
        separator: str,
    ) -> SNPObject:
        import pandas as pd

        n_records = _count_vcf_records(
            self._filename,
            region_filter=region_filter,
            separator=separator,
        )
        arrays: dict[str, np.ndarray] = {}
        for field in field_columns:
            if field == "FORMAT":
                continue
            arrays[field] = np.empty(n_records, dtype=np.int32 if field == "POS" else object)

        n_selected = len(sample_columns)
        genotypes = _empty_genotype_array(n_records, n_selected, return_dosage)

        filter_columns = []
        if region_filter is not None:
            chrom_column = "#CHROM" if "#CHROM" in names else "CHROM" if "CHROM" in names else None
            filter_columns = ["POS"]
            if chrom_column is not None:
                filter_columns.insert(0, chrom_column)
        elif detect_non_diploid:
            chrom_column = "#CHROM" if "#CHROM" in names else "CHROM" if "CHROM" in names else None
            filter_columns = [] if chrom_column is None else [chrom_column]

        parsing_columns = ["FORMAT"] if n_selected else []
        if return_dosage and not assume_biallelic:
            parsing_columns.append("ALT")
        usecols = list(dict.fromkeys(field_columns + sample_columns + filter_columns + parsing_columns))
        offset = 0
        for frame in pd.read_csv(
            self._filename,
            sep=separator,
            comment="#",
            names=names,
            usecols=usecols,
            chunksize=25_000,
            dtype=str,
            engine="c",
        ):
            if region_filter is not None:
                region_chrom, start, end = region_filter
                chrom_column = filter_columns[0]
                positions = frame["POS"].astype(np.int64)
                mask = frame[chrom_column].astype(str).eq(region_chrom)
                if start is not None:
                    mask = mask & (positions >= start)
                if end is not None:
                    mask = mask & (positions <= end)
                frame = frame.loc[mask]

            height = len(frame)
            if height == 0:
                continue
            if return_dosage and not assume_biallelic and frame["ALT"].str.contains(",", regex=False).any():
                raise ValueError(_MULTIALLELIC_DOSAGE_ERROR)
            if n_selected:
                non_diploid_chromosomes = None
                if detect_non_diploid and ("#CHROM" in frame.columns or "CHROM" in frame.columns):
                    chrom_column = "#CHROM" if "#CHROM" in frame.columns else "CHROM"
                    non_diploid_chromosomes = _non_diploid_chromosome_mask_or_none(
                        frame[chrom_column].to_numpy(dtype=object)
                    )
                calls = _parse_gt_sample_matrix(
                    frame[sample_columns].to_numpy(dtype=object),
                    frame["FORMAT"].to_numpy(dtype=object),
                    return_dosage=bool(return_dosage),
                    non_diploid_chromosomes=non_diploid_chromosomes,
                )
                if calls.itemsize > genotypes.itemsize:
                    genotypes = _widen_gt_buffer(genotypes, calls.dtype, offset)
                genotypes[offset:offset + height] = calls

            for field in field_columns:
                if field == "FORMAT":
                    continue
                dtype = np.int32 if field == "POS" else object
                arrays[field][offset:offset + height] = frame[field].to_numpy(dtype=dtype)
            offset += height

        if offset != n_records:
            arrays = {field: values[:offset] for field, values in arrays.items()}
            if n_selected:
                genotypes = genotypes[:offset]

        return self._make_snpobject(
            genotypes=genotypes,
            sample_columns=sample_columns,
            arrays=arrays,
        )

    def read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        region: Optional[str] = None,
        samples: Optional[Sequence[Union[str, int]]] = None,
        genotype_mode: GenotypeMode = "auto",
        chromosome_ploidy: Optional[str] = None,
        separator: Optional[str] = None,
        threads: int = 1,
        *,
        assume_biallelic: bool = False,
    ) -> SNPObject:
        """
        Read a VCF file into an :class:`~snputils.snp.genobj.snpobj.SNPObject`.

        By default, the reader loads the core VCF variant columns ``CHROM``,
        ``POS``, ``ID``, ``REF``, ``ALT``, ``QUAL``, and ``FILTER``, plus all
        sample genotype columns. Genotypes are read from the ``GT`` FORMAT
        field and returned as an ``int8`` array, widened to a signed integer type
        when a phased allele index exceeds 127. By default
        (``genotype_mode="auto"``), phased genotypes are kept separate with shape
        ``(n_variants, n_samples, 2)`` and unphased GT calls fall back to
        dosages with shape ``(n_variants, n_samples)``. With
        ``genotype_mode="dosage"``, the two alleles are converted to dosage. With
        ``genotype_mode="phased"``, unphased ``/`` GT calls are rejected because
        their allele order is not meaningful.

        Args:
            fields: VCF fixed columns to include, such as ``["CHROM", "POS",
                "ID"]``. Use ``"*"`` to include all fixed VCF columns,
                including ``INFO`` and ``FORMAT``. If ``None``, the default core
                fields are used.
            exclude_fields: Fixed VCF columns to exclude. This is mainly useful
                with ``fields="*"``; when ``fields`` is ``None``, it excludes
                columns from the default core field set.
            region: Optional genomic region to read. Accepts chromosome-only
                values such as ``"22"`` or inclusive 1-based intervals such as
                ``"22:100000-200000"``. Records are included when their POS is
                within the requested interval.
            samples: Optional sample subset. Provide sample IDs or zero-based
                sample indexes. If omitted, all samples are read; pass an empty
                sequence to read variant metadata without genotypes.
            genotype_mode: ``"dosage"`` returns biallelic ALT-copy counts
                (``0``, ``1``, or ``2``) and rejects multiallelic variants. ``"phased"`` keeps phased allele columns separate
                and rejects unphased GT calls. ``"auto"`` (default) preserves
                phased calls and falls back to dosage for unphased calls.
            assume_biallelic: Skip ALT checks for a VCF known to contain at most one
                ALT allele per record. A wrong assumption can produce incorrect genotypes.
            chromosome_ploidy:
                Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
                all selected variants should be treated as ordinary diploid/autosomal; this
                skips non-diploid chromosome checks and can be faster. The default None/"auto"
                preserves existing behavior.
            separator: Optional column separator. If omitted, the separator is
                detected from the VCF header. Tab-delimited files use optimized
                byte parsers when possible; other separators use the pandas
                chunked parser.
            threads: Number of BGZF decompression and native GT decoder threads.
                Values above 1 currently require a filename whose final two
                suffixes are exactly ``.vcf.gz``, plus an unfiltered explicit
                dosage or phased read with at most ``POS`` variant metadata and
                fixed-width diploid GT-only sample fields. Use ``threads=1``
                for haploid calls or other valid GT layouts. Dosage additionally
                requires ``chromosome_ploidy="autosomal"``. The default is 1.

        Returns:
            SNPObject: Object containing selected genotype, sample, and variant
            fields.
        """
        genotype_mode = normalize_genotype_mode(genotype_mode)
        _check_assume_biallelic(assume_biallelic)
        chromosome_ploidy_mode = _normalize_chromosome_ploidy(chromosome_ploidy)
        try:
            threads = operator.index(threads)
        except TypeError as exc:
            raise TypeError("VCFReader threads must be an integer.") from exc
        if threads < 1:
            raise ValueError("VCFReader threads must be at least 1.")
        if genotype_mode == "auto":
            if threads != 1:
                raise ValueError(
                    "Multithreaded VCF reading requires an explicit genotype_mode "
                    "of 'dosage' or 'phased'."
                )
            try:
                return self.read(
                    fields=fields,
                    exclude_fields=exclude_fields,
                    region=region,
                    samples=samples,
                    genotype_mode="phased",
                    chromosome_ploidy=chromosome_ploidy,
                    separator=separator,
                    threads=threads,
                    assume_biallelic=assume_biallelic,
                )
            except ValueError as exc:
                if _UNPHASED_VCF_PHASED_ERROR not in str(exc):
                    raise
                return self.read(
                    fields=fields,
                    exclude_fields=exclude_fields,
                    region=region,
                    samples=samples,
                    genotype_mode="dosage",
                    chromosome_ploidy=chromosome_ploidy,
                    separator=separator,
                    threads=threads,
                    assume_biallelic=assume_biallelic,
                )
        return_dosage = genotype_mode == "dosage"
        detect_non_diploid = return_dosage and chromosome_ploidy_mode != "autosomal"

        region_filter = _parse_vcf_region(region)
        names, detected_separator = _get_vcf_col_names_and_sep(
            str(self._filename),
            separator=separator,
        )
        field_columns, sample_columns, sample_idxs = _resolve_fast_vcf_columns(
            names,
            fields,
            exclude_fields,
            samples,
        )

        if threads != 1:
            if (
                any(field != "POS" for field in field_columns)
                or region_filter is not None
                or (return_dosage and chromosome_ploidy_mode != "autosomal")
                or detected_separator != "\t"
                or Path(self._filename).suffixes[-2:] != [".vcf", ".gz"]
            ):
                raise ValueError(
                    "Multithreaded VCF reading currently requires an unfiltered tab-delimited "
                    ".vcf.gz explicit dosage or phased read with at most POS metadata; dosage additionally "
                    "requires chromosome_ploidy='autosomal'."
                )
            return self._read_parallel_gt_only(
                names=names,
                field_columns=field_columns,
                sample_columns=sample_columns,
                sample_idxs=sample_idxs,
                return_dosage=return_dosage,
                threads=threads,
                assume_biallelic=assume_biallelic,
            )

        if detected_separator != "\t":
            return self._read_pandas_chunks(
                names=names,
                field_columns=field_columns,
                sample_columns=sample_columns,
                region_filter=region_filter,
                return_dosage=bool(return_dosage),
                detect_non_diploid=detect_non_diploid,
                assume_biallelic=assume_biallelic,
                separator=detected_separator,
            )

        try:
            if Path(self._filename).suffixes[-2:] in ([".vcf", ".gz"], [".vcf", ".zst"]):
                return self._read_block_gt_only_streaming(
                    names=names,
                    field_columns=field_columns,
                    sample_columns=sample_columns,
                    sample_idxs=sample_idxs,
                    region_filter=region_filter,
                    return_dosage=bool(return_dosage),
                    detect_non_diploid=detect_non_diploid,
                    assume_biallelic=assume_biallelic,
                )
            return self._read_block_gt_only(
                names=names,
                field_columns=field_columns,
                sample_columns=sample_columns,
                sample_idxs=sample_idxs,
                region_filter=region_filter,
                return_dosage=bool(return_dosage),
                detect_non_diploid=detect_non_diploid,
                assume_biallelic=assume_biallelic,
            )
        except ValueError as exc:
            log.debug("VCF fast block path unavailable for %s: %s", self._filename, exc)
            try:
                return self._read_simple_format_streaming(
                    names=names,
                    field_columns=field_columns,
                    sample_columns=sample_columns,
                    sample_idxs=sample_idxs,
                    region_filter=region_filter,
                    return_dosage=bool(return_dosage),
                    detect_non_diploid=detect_non_diploid,
                    assume_biallelic=assume_biallelic,
                )
            except ValueError as fallback_exc:
                log.debug(
                    "VCF simple FORMAT path unavailable for %s: %s",
                    self._filename,
                    fallback_exc,
                )
                return self._read_pandas_chunks(
                    names=names,
                    field_columns=field_columns,
                    sample_columns=sample_columns,
                    region_filter=region_filter,
                    return_dosage=bool(return_dosage),
                    detect_non_diploid=detect_non_diploid,
                    assume_biallelic=assume_biallelic,
                    separator=detected_separator,
                )

    def iter_read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        region: Optional[str] = None,
        samples: Optional[Sequence[Union[str, int]]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        sample_idxs: Optional[np.ndarray] = None,
        variant_ids: Optional[np.ndarray] = None,
        variant_idxs: Optional[np.ndarray] = None,
        genotype_mode: ExplicitGenotypeMode = "phased",
        chromosome_ploidy: Optional[str] = None,
        separator: Optional[str] = None,
        chunk_size: int = 10_000,
        threads: int = 1,
        *,
        assume_biallelic: bool = False,
    ) -> Iterator[SNPObject]:
        """
        Stream a VCF in variant chunks.

        assume_biallelic: Skip ALT checks for a VCF known to contain at most one
            ALT allele per record. A wrong assumption can produce incorrect genotypes.
        chromosome_ploidy:
            Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
            all selected variants should be treated as ordinary diploid/autosomal; this
            skips non-diploid chromosome checks and can be faster. The default None/"auto"
            preserves existing behavior.
        threads: Number of native GT decoder threads used for each chunk. Values
            above one require GT-only sample fields; dosage also requires
            ``chromosome_ploidy="autosomal"``.
        """
        genotype_mode = normalize_genotype_mode(genotype_mode, allow_auto=False)
        return_dosage = genotype_mode == "dosage"
        _check_assume_biallelic(assume_biallelic)
        chromosome_ploidy_mode = _normalize_chromosome_ploidy(chromosome_ploidy)
        detect_non_diploid = return_dosage and chromosome_ploidy_mode != "autosomal"
        try:
            threads = operator.index(threads)
        except TypeError as exc:
            raise TypeError("VCFReader threads must be an integer.") from exc
        if threads < 1:
            raise ValueError("VCFReader threads must be at least 1.")
        if threads != 1 and detect_non_diploid:
            raise ValueError(
                "Multithreaded VCF streaming dosage reads require "
                "chromosome_ploidy='autosomal'."
            )
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")
        if separator not in (None, "\t"):
            raise ValueError("VCFReader.iter_read only supports tab-delimited VCF files.")
        if samples is not None and (sample_ids is not None or sample_idxs is not None):
            raise ValueError("Only one of samples, sample_ids, and sample_idxs can be specified.")
        if sample_idxs is not None and sample_ids is not None:
            raise ValueError("Only one of sample_idxs and sample_ids can be specified.")
        if variant_idxs is not None and variant_ids is not None:
            raise ValueError("Only one of variant_idxs and variant_ids can be specified.")

        selected_samples: Optional[Sequence[Union[str, int]]]
        if samples is not None:
            selected_samples = samples
        elif sample_idxs is not None:
            selected_samples = [int(idx) for idx in np.asarray(sample_idxs).ravel()]
        else:
            selected_samples = None if sample_ids is None else list(sample_ids)

        region_filter = _parse_vcf_region(region)
        names = _vcf_header_columns(self._filename)
        field_columns, sample_columns, sample_indices = _resolve_fast_vcf_columns(
            names,
            fields,
            exclude_fields,
            selected_samples,
        )
        n_samples_total = len(names) - 9
        all_samples = np.array_equal(sample_indices, np.arange(n_samples_total))
        if threads != 1:
            from snputils.snp.io.read import _vcf

            native_sample_idxs = None if all_samples else sample_indices.tolist()
        wanted_variant_ids = None if variant_ids is None else set(map(str, np.asarray(variant_ids).ravel()))
        wanted_variant_idxs = None if variant_idxs is None else set(map(int, np.asarray(variant_idxs).ravel()))

        include = set(field_columns)
        records: dict[str, list[Any]] = {
            "chrom": [],
            "pos": [],
            "id": [],
            "ref": [],
            "alt": [],
            "qual": [],
            "filter": [],
            "info": [],
            "gt": [],
        }
        records_count = 0
        gt_lines: list[bytes] = []

        def flush() -> Optional[SNPObject]:
            nonlocal records_count
            if records_count == 0:
                return None
            if len(sample_columns) == 0:
                if return_dosage:
                    gt = np.empty((records_count, 0), dtype=np.int8)
                else:
                    gt = np.empty((records_count, 0, 2), dtype=np.int8)
            elif threads == 1:
                gt = np.stack(records["gt"], axis=0)
            else:
                gt_buffer, _, decoded_records = _vcf.decode_gt(
                    b"".join(gt_lines),
                    0,
                    n_samples_total,
                    native_sample_idxs,
                    bool(return_dosage),
                    threads,
                    assume_biallelic,
                )
                if decoded_records != records_count:
                    raise RuntimeError("Native VCF chunk decoder returned an unexpected record count.")
                gt = np.frombuffer(gt_buffer, dtype=np.int8)
                if return_dosage:
                    gt = gt.reshape(records_count, len(sample_columns))
                else:
                    gt = gt.reshape(records_count, len(sample_columns), 2)
            arrays: dict[str, np.ndarray] = {}
            if "REF" in include:
                arrays["REF"] = np.asarray(records["ref"], dtype=object)
            if "ALT" in include:
                arrays["ALT"] = np.asarray(records["alt"], dtype=object)
            if "#CHROM" in include or "CHROM" in include:
                arrays["#CHROM" if "#CHROM" in include else "CHROM"] = np.asarray(records["chrom"], dtype=object)
            if "FILTER" in include:
                arrays["FILTER"] = np.asarray(records["filter"], dtype=object)
            if "ID" in include:
                arrays["ID"] = np.asarray(records["id"], dtype=object)
            if "POS" in include:
                arrays["POS"] = np.asarray(records["pos"], dtype=np.int32)
            if "QUAL" in include:
                arrays["QUAL"] = np.asarray(records["qual"], dtype=object)
            if "INFO" in include:
                arrays["INFO"] = np.asarray(records["info"], dtype=object)
            snpobj = self._make_snpobject(
                genotypes=gt,
                sample_columns=sample_columns,
                arrays=arrays,
            )
            for values in records.values():
                values.clear()
            gt_lines.clear()
            records_count = 0
            return snpobj

        row_idx = 0
        with _open_vcf_binary(self._filename) as file:
            for line in file:
                if not line or line.startswith(b"#"):
                    continue
                parts = line.split(b"\t", 9)
                if len(parts) < 10:
                    raise ValueError("Malformed VCF record with fewer than 10 tab-delimited fields.")
                if wanted_variant_idxs is not None and row_idx not in wanted_variant_idxs:
                    row_idx += 1
                    continue
                if region_filter is not None and not _vcf_region_matches(
                    _decode_vcf_value(parts[0]),
                    int(parts[1]),
                    region_filter,
                ):
                    row_idx += 1
                    continue
                if wanted_variant_ids is not None and not _variant_id_matches(parts, wanted_variant_ids):
                    row_idx += 1
                    continue
                if return_dosage and not assume_biallelic and b"," in parts[4]:
                    raise ValueError(_MULTIALLELIC_DOSAGE_ERROR)
                if "#CHROM" in include or "CHROM" in include:
                    records["chrom"].append(_decode_vcf_value(parts[0]))
                if "POS" in include:
                    records["pos"].append(int(parts[1]))
                if "ID" in include:
                    records["id"].append(_decode_vcf_value(parts[2]))
                if "REF" in include:
                    records["ref"].append(_decode_vcf_value(parts[3]))
                if "ALT" in include:
                    records["alt"].append(_decode_vcf_value(parts[4]))
                if "QUAL" in include:
                    records["qual"].append(_decode_vcf_value(parts[5]))
                if "FILTER" in include:
                    records["filter"].append(_decode_vcf_value(parts[6]))
                if "INFO" in include:
                    records["info"].append(_decode_vcf_value(parts[7]))
                if sample_columns:
                    if threads == 1:
                        records["gt"].append(
                            _parse_gt_sample_bytes(
                                parts[9],
                                format_value=parts[8],
                                n_samples_total=n_samples_total,
                                sample_idxs=sample_indices,
                                all_samples=all_samples,
                                return_dosage=bool(return_dosage),
                                non_diploid_chromosome=(
                                    (
                                        _normalized_non_diploid_chromosome(_decode_vcf_value(parts[0])) is not None
                                    )
                                    if detect_non_diploid
                                    else None
                                ),
                            )
                        )
                    else:
                        if parts[8] != b"GT":
                            raise ValueError(
                                "Multithreaded VCF streaming requires GT-only sample fields."
                            )
                        gt_lines.append(line)
                records_count += 1
                if records_count >= chunk_size:
                    chunk = flush()
                    if chunk is not None:
                        yield chunk
                row_idx += 1

        chunk = flush()
        if chunk is not None:
            yield chunk


def _infer_col_data_types(names: List):
    """
    Infer data types for VCF columns.

    Args:
        names: List of column names.

    Returns:
        col_dtypes: Dictionary mapping column names to data types.
    """
    col_dtypes = {name: pl.Utf8 for name in names}
    if 'POS' in col_dtypes:
        col_dtypes['POS'] = pl.Int32
    if '#CHROM' in col_dtypes:
        col_dtypes['#CHROM'] = pl.String
    if 'CHROM' in col_dtypes:
        col_dtypes['CHROM'] = pl.String

    return col_dtypes


def _extract_columns(names: List[str], fields: List[str], exclude_fields: List[str],
                     samples: List[str]) -> List[str]:
    """
    Extracts columns based on specified `fields`, `exclude_fields` and `samples`.

    Args:
        names: List of column names.
        fields: Fields to extract data for. This parameter specifies which data fields
            from the VCF file should be included in the result. To extract all fields,
            provide just the string '*'.
        exclude_fields: Fields to exclude. E.g., for use in combination with fields='*'.
        samples: Selection of samples to extract calldata for. If provided, should be
            a list of strings giving sample identifiers. May also be a list of
            integers giving indices of selected samples.

    Returns:
        field_columns: List of field columns.
        sample_columns: List of sample columns.
        selected_column_idxs: List of selected column indices.
    """

    # Define standard field names in a VCF file
    field_names = ['#CHROM', 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']

    # Find the index of the first column that is not a standard field name
    first_sample_idx = next((i for i, col in enumerate(names) if col not in field_names), len(names))

    # Identify field columns as all columns before the first sample column
    field_columns = names[:first_sample_idx]

    if fields != '*' and fields is not None:
        # Filter field columns to contain those in `fields`
        selected_fields = set(fields)
        if 'CHROM' in selected_fields and '#CHROM' in names:
            selected_fields.add('#CHROM')
        if '#CHROM' in selected_fields and 'CHROM' in names:
            selected_fields.add('CHROM')
        field_columns = [col for col in field_columns if col in selected_fields]
    elif fields == '*' and exclude_fields is not None:
        excluded_fields = set(exclude_fields)
        if 'CHROM' in excluded_fields and '#CHROM' in names:
            excluded_fields.add('#CHROM')
        if '#CHROM' in excluded_fields and 'CHROM' in names:
            excluded_fields.add('CHROM')
        field_columns = [col for col in field_columns if col not in excluded_fields]

    # Sample columns are all columns starting from the first sample column
    sample_columns = names[first_sample_idx:]

    if samples is not None:
        if len(samples) == 0:
            sample_columns = []
        elif type(samples[0]) is int:
            sample_columns = list(np.array(sample_columns)[samples])
        else:
            selected_samples = set(samples)
            sample_columns = [col for col in sample_columns if col in selected_samples]

    # Create a dictionary mapping column names to their indices
    column_idx_map = {name: index for index, name in enumerate(names)}

    # Create a list of selected column indices
    selected_column_idxs = [column_idx_map[col] for col in field_columns + sample_columns]

    # Sort the selected column indices
    selected_column_idxs.sort()

    return field_columns, sample_columns, selected_column_idxs


@SNPBaseReader.register
class VCFReaderPolars(SNPBaseReader):
    """Reads a VCF file and processes it into a SNPObject."""

    def __init__(self, filename: Union[str, pathlib.Path]):
        super().__init__(filename)

    def _resolve_columns(
        self,
        fields: Optional[List[str]],
        exclude_fields: Optional[List[str]],
        samples: Optional[List[str]],
        separator: Optional[str],
    ) -> Tuple[List[str], List[str], List[int], Dict[str, pl.DataType], str]:
        """
        Resolve the field/sample column selections and parser schema for the VCF.
        """
        col_names, detected_separator = _get_vcf_col_names_and_sep(
            str(self._filename),
            separator=separator,
        )
        col_dtypes = _infer_col_data_types(col_names)
        field_columns, sample_columns, selected_column_idxs = _extract_columns(
            col_names,
            fields,
            exclude_fields,
            samples,
        )
        return field_columns, sample_columns, selected_column_idxs, col_dtypes, detected_separator

    def _parse_genotypes(
        self,
        vcf: pl.DataFrame,
        sample_columns: List[str],
        return_dosage: bool,
        non_diploid_chromosomes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not sample_columns:
            return _empty_genotype_array(vcf.height, 0, return_dosage)

        gt = vcf.select(sample_columns)
        formats = vcf["FORMAT"].unique().to_list()
        if formats != ["GT"]:
            # GT is optional per record. A null index yields a missing call for every sample.
            indexes = {fmt: _format_gt_index(fmt) for fmt in formats if fmt and "GT" in fmt.split(":")}
            index = pl.col("FORMAT").replace_strict(indexes, default=None, return_dtype=pl.Int64)
            gt = vcf.select(pl.col(sample_columns).str.split(":").list.get(index, null_on_oob=True))

        if not return_dosage and gt.select(pl.all().str.contains("/", literal=True).any()).to_numpy().any():
            raise ValueError(_UNPHASED_VCF_PHASED_ERROR)
        token = pl.all()
        if return_dosage:
            token = token.str.replace_all("/", "|", literal=True)
        pairs = gt.select(token.str.strip_prefix("|").str.splitn("|", 2))
        alleles = []
        for field in ("field_0", "field_1"):
            values = pl.all().struct.field(field).replace([".", "-", ""], "-1").fill_null("-1")
            try:
                allele = pairs.select(values.cast(pl.Int8).name.keep()).to_numpy()
            except pl.exceptions.InvalidOperationError:
                allele = pairs.select(values.cast(pl.Int64).name.keep()).to_numpy()
                allele = allele.astype(_allele_dtype(int(allele.max())), copy=False)
            alleles.append(allele)
        genotypes = np.stack(alleles, axis=2)
        if return_dosage:
            genotypes = sum_diploid_alleles(
                genotypes[:, :, 0],
                genotypes[:, :, 1],
                missing_as_haploid=non_diploid_chromosomes,
            )

        return genotypes

    def _dataframe_to_snpobject(
        self,
        vcf: pl.DataFrame,
        field_columns: List[str],
        sample_columns: List[str],
        return_dosage: bool,
        assume_biallelic: bool,
        detect_non_diploid: bool,
    ) -> SNPObject:
        if return_dosage and not assume_biallelic and "ALT" in vcf.columns:
            if vcf["ALT"].str.contains(",", literal=True).any():
                raise ValueError(_MULTIALLELIC_DOSAGE_ERROR)

        if "#CHROM" in vcf.columns:
            chrom_column = "#CHROM"
        elif "CHROM" in vcf.columns:
            chrom_column = "CHROM"
        else:
            chrom_column = None

        non_diploid_chromosomes = None
        if detect_non_diploid and chrom_column is not None:
            non_diploid_chromosomes = _non_diploid_chromosome_mask_or_none(
                vcf[chrom_column].to_numpy()
            )

        genotypes = self._parse_genotypes(
            vcf,
            sample_columns,
            bool(return_dosage),
            non_diploid_chromosomes=non_diploid_chromosomes,
        )

        return SNPObject(
            genotypes=genotypes,
            samples=np.asarray(sample_columns),
            variants_ref=vcf["REF"].to_numpy() if "REF" in field_columns else np.array([]),
            variants_alt=vcf["ALT"].to_numpy() if "ALT" in field_columns else np.array([]),
            variants_chrom=(
                vcf[chrom_column].to_numpy()
                if chrom_column is not None and chrom_column in field_columns
                else np.array([])
            ),
            variants_filter_pass=vcf["FILTER"].to_numpy() if "FILTER" in field_columns else np.array([]),
            variants_id=vcf["ID"].to_numpy() if "ID" in field_columns else np.array([]),
            variants_pos=vcf["POS"].to_numpy() if "POS" in field_columns else np.array([]),
            variants_qual=vcf["QUAL"].to_numpy() if "QUAL" in field_columns else np.array([]),
            variants_info=vcf["INFO"].to_numpy() if "INFO" in field_columns else np.array([]),
        )

    def read(self,
             fields: Optional[List[str]] = None,
             exclude_fields: Optional[List[str]] = None,
             region: Optional[str] = None,
             samples: Optional[List[str]] = None,
             genotype_mode: GenotypeMode = "auto",
             chromosome_ploidy: Optional[str] = None,
             separator: Optional[str] = None,
             threads: Optional[int] = None,
             *,
             assume_biallelic: bool = False,
             ) -> SNPObject:
        """
        Read a vcf file into a SNPObject.

        Args:
            fields: Fields to extract data for. This parameter specifies which data fields
                from the VCF file should be included in the result. Available options include
                'CHROM'/'#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', and
                'FORMAT'.
                To extract all fields, provide just the string '*' or the default None.
            exclude_fields: Fields to exclude for use in combination with fields='*'.
                Available options include 'CHROM'/'#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL',
                'FILTER', 'INFO', and 'FORMAT'.
            region: Genomic region to extract variants for. If provided, it should be a
                tabix-style region string, specifying a chromosome name and optionally
                beginning and end coordinates (e.g., '2L:100000-200000'). TODO
            samples: Selection of samples to extract calldata for. If provided, should be
                a list of strings giving sample identifiers. May also be a list of
                integers giving indices of selected samples.  If an empty list is provided,
                no samples are extracted.
            genotype_mode: ``"dosage"`` returns biallelic ALT-copy counts
                (``0``, ``1``, or ``2``) and rejects multiallelic variants.
                ``"phased"`` preserves phased allele calls and rejects unphased
                calls. ``"auto"`` preserves phased calls and falls back to
                dosage for unphased calls.
            assume_biallelic: Skip ALT checks for a VCF known to contain at most one
                ALT allele per record. A wrong assumption can produce incorrect genotypes.
            chromosome_ploidy:
                Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
                all selected variants should be treated as ordinary diploid/autosomal; this
                skips non-diploid chromosome checks and can be faster. The default None/"auto"
                preserves existing behavior.
            separator: Separator used in the pvar file. If None, the separator is automatically detected.
                If the automatic detection fails, please specify the separator manually.
            threads: Number of CSV parser threads on Polars 1.x. None lets Polars
                choose. Polars 2.x uses its process-wide thread pool for CSV reads.
                Set POLARS_MAX_THREADS before starting Python to size that pool.

        Returns:
            snpobj: SNPObject containing the data from the VCF file. The format and content
                of this object depend on the specified parameters and the content of the VCF file.
        """
        # TODO: add support for excluding GT
        if threads is not None:
            try:
                threads = operator.index(threads)
            except TypeError as exc:
                raise TypeError("VCFReaderPolars threads must be an integer or None.") from exc
            if threads < 1:
                raise ValueError("VCFReaderPolars threads must be at least 1.")
        genotype_mode = normalize_genotype_mode(genotype_mode)
        _check_assume_biallelic(assume_biallelic)
        chromosome_ploidy_mode = _normalize_chromosome_ploidy(chromosome_ploidy)
        if genotype_mode == "auto":
            try:
                return self.read(
                    fields=fields,
                    exclude_fields=exclude_fields,
                    region=region,
                    samples=samples,
                    genotype_mode="phased",
                    chromosome_ploidy=chromosome_ploidy,
                    separator=separator,
                    threads=threads,
                    assume_biallelic=assume_biallelic,
                )
            except ValueError as exc:
                if _UNPHASED_VCF_PHASED_ERROR not in str(exc):
                    raise
                return self.read(
                    fields=fields,
                    exclude_fields=exclude_fields,
                    region=region,
                    samples=samples,
                    genotype_mode="dosage",
                    chromosome_ploidy=chromosome_ploidy,
                    separator=separator,
                    threads=threads,
                    assume_biallelic=assume_biallelic,
                )
        return_dosage = genotype_mode == "dosage"
        detect_non_diploid = return_dosage and chromosome_ploidy_mode != "autosomal"

        log.info(f"Reading {self._filename}")

        try:
            field_columns, sample_columns, selected_column_idxs, col_dtypes, detected_separator = self._resolve_columns(
                fields=fields,
                exclude_fields=exclude_fields,
                samples=samples,
                separator=separator,
            )
            if sample_columns and "FORMAT" not in field_columns:
                selected_column_idxs = sorted(set(selected_column_idxs + [list(col_dtypes).index("FORMAT")]))
            if return_dosage and not assume_biallelic and "ALT" not in field_columns:
                selected_column_idxs = sorted(set(selected_column_idxs + [list(col_dtypes).index("ALT")]))
            if detect_non_diploid:
                chrom_column = "#CHROM" if "#CHROM" in col_dtypes else "CHROM" if "CHROM" in col_dtypes else None
                if chrom_column is not None:
                    chrom_idx = list(col_dtypes).index(chrom_column)
                    selected_column_idxs = sorted(set(selected_column_idxs + [chrom_idx]))

            thread_options = {}
            if _POLARS_CSV_HAS_N_THREADS:
                thread_options["n_threads"] = threads
            elif threads is not None:
                log.warning(
                    "Polars uses its process-wide thread pool for this read. "
                    "Set POLARS_MAX_THREADS=%d before starting Python to select its size.",
                    threads,
                )

            vcf = pl.read_csv(
                self._filename,
                comment_prefix="##",
                has_header=True,
                separator=detected_separator,
                # A projection needs one column to retain the row count.
                columns=selected_column_idxs or [0],
                schema_overrides=col_dtypes,
                **thread_options,
            )

            log.debug("vcf polars read")
            snpobj = self._dataframe_to_snpobject(
                vcf=vcf,
                field_columns=field_columns,
                sample_columns=sample_columns,
                return_dosage=bool(return_dosage),
                detect_non_diploid=detect_non_diploid,
                assume_biallelic=assume_biallelic,
            )

            log.info(f"Finished reading {self.filename}")

            return snpobj

        except Exception as e:
            # Auto mode handles this signal by retrying the Polars read in dosage mode.
            if isinstance(e, ValueError) and str(e) == _UNPHASED_VCF_PHASED_ERROR:
                raise
            log.warning(
                "Polars VCF parsing failed (%s). Falling back to default VCF reader.",
                e,
            )
            from snputils.snp.io.read import VCFReader

            # Instantiate a VCFReader object and read SNP data
            reader = VCFReader(self._filename)
            snpobj = reader.read(
                fields=fields,
                exclude_fields=exclude_fields,
                region=region,
                samples=samples,
                genotype_mode=genotype_mode,
                chromosome_ploidy=chromosome_ploidy,
                threads=1,
                assume_biallelic=assume_biallelic,
            )

            return snpobj

    def iter_read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        region: Optional[str] = None,
        samples: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        sample_idxs: Optional[np.ndarray] = None,
        variant_ids: Optional[np.ndarray] = None,
        variant_idxs: Optional[np.ndarray] = None,
        genotype_mode: ExplicitGenotypeMode = "phased",
        chromosome_ploidy: Optional[str] = None,
        separator: Optional[str] = None,
        chunk_size: int = 10_000,
        *,
        assume_biallelic: bool = False,
    ) -> Iterator[SNPObject]:
        """
        Stream a VCF in variant chunks using the Polars backend.

        Streaming parallelism uses Polars' process-wide thread pool.

        assume_biallelic: Skip ALT checks for a VCF known to contain at most one
            ALT allele per record. A wrong assumption can produce incorrect genotypes.
        chromosome_ploidy:
            Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
            all selected variants should be treated as ordinary diploid/autosomal; this
            skips non-diploid chromosome checks and can be faster. The default None/"auto"
            preserves existing behavior.
        """
        genotype_mode = normalize_genotype_mode(genotype_mode, allow_auto=False)
        return_dosage = genotype_mode == "dosage"
        _check_assume_biallelic(assume_biallelic)
        chromosome_ploidy_mode = _normalize_chromosome_ploidy(chromosome_ploidy)
        detect_non_diploid = return_dosage and chromosome_ploidy_mode != "autosomal"
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")
        if region is not None:
            raise NotImplementedError("VCFReaderPolars.iter_read does not support `region` yet.")
        if samples is not None and (sample_ids is not None or sample_idxs is not None):
            raise ValueError("Only one of samples, sample_ids, and sample_idxs can be specified.")
        if sample_idxs is not None and sample_ids is not None:
            raise ValueError("Only one of sample_idxs and sample_ids can be specified.")
        if variant_idxs is not None and variant_ids is not None:
            raise ValueError("Only one of variant_idxs and variant_ids can be specified.")

        if samples is not None:
            selected_samples = samples
        elif sample_idxs is not None:
            selected_samples = [int(idx) for idx in np.asarray(sample_idxs).ravel()]
        else:
            selected_samples = None if sample_ids is None else list(np.asarray(sample_ids).ravel())

        field_columns, sample_columns, _, col_dtypes, detected_separator = self._resolve_columns(
            fields=fields,
            exclude_fields=exclude_fields,
            samples=selected_samples,
            separator=separator,
        )

        selected_columns = field_columns + sample_columns
        filter_columns = ["FORMAT"] if sample_columns else []
        chrom_column = "#CHROM" if "#CHROM" in col_dtypes else "CHROM" if "CHROM" in col_dtypes else None
        if detect_non_diploid and chrom_column is not None:
            filter_columns.append(chrom_column)
        if variant_ids is not None:
            filter_columns.extend(["ID", "POS", "REF", "ALT"])
            if chrom_column is not None:
                filter_columns.append(chrom_column)
        if return_dosage and not assume_biallelic:
            filter_columns.append("ALT")
        scan_columns = list(dict.fromkeys(selected_columns + [col for col in filter_columns if col in col_dtypes]))
        wanted_variant_ids = None if variant_ids is None else np.asarray(variant_ids, dtype=str).ravel()
        wanted_variant_idxs = None if variant_idxs is None else np.asarray(variant_idxs, dtype=np.uint64).ravel()
        try:
            reader = (
                pl.scan_csv(
                    self._filename,
                    comment_prefix="##",
                    has_header=True,
                    separator=detected_separator,
                    schema_overrides=col_dtypes,
                )
                .select(scan_columns or [next(iter(col_dtypes))])
                .collect_batches(chunk_size=int(chunk_size))
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize VCF batched reader for {self._filename}: {exc}") from exc

        pending: Optional[pl.DataFrame] = None
        row_offset = 0

        for batch in reader:
            original_height = batch.height
            if wanted_variant_idxs is not None:
                row_idxs = np.arange(row_offset, row_offset + original_height, dtype=np.uint64)
                batch = batch.filter(np.isin(row_idxs, wanted_variant_idxs))
            row_offset += original_height

            if wanted_variant_ids is not None and batch.height > 0:
                id_expr = pl.col("ID").cast(pl.Utf8).is_in(wanted_variant_ids)
                if chrom_column is not None and all(col in batch.columns for col in (chrom_column, "POS", "REF", "ALT")):
                    computed_id_expr = pl.concat_str(
                        [
                            pl.col(chrom_column).cast(pl.Utf8),
                            pl.col("POS").cast(pl.Utf8),
                            pl.col("REF").cast(pl.Utf8),
                            pl.col("ALT").cast(pl.Utf8),
                        ],
                        separator=":",
                    ).is_in(wanted_variant_ids)
                    batch = batch.filter(id_expr | computed_id_expr)
                else:
                    batch = batch.filter(id_expr)

            if batch.height == 0:
                continue
            if pending is None:
                pending = batch
            else:
                pending = pl.concat([pending, batch], how="vertical_relaxed")

            while pending.height >= int(chunk_size):
                chunk_df = pending.slice(0, int(chunk_size))
                pending = pending.slice(int(chunk_size))
                yield self._dataframe_to_snpobject(
                    vcf=chunk_df,
                    field_columns=field_columns,
                    sample_columns=sample_columns,
                    return_dosage=bool(return_dosage),
                    detect_non_diploid=detect_non_diploid,
                    assume_biallelic=assume_biallelic,
                )

        if pending is not None and pending.height > 0:
            yield self._dataframe_to_snpobject(
                vcf=pending,
                field_columns=field_columns,
                sample_columns=sample_columns,
                return_dosage=bool(return_dosage),
                detect_non_diploid=detect_non_diploid,
                assume_biallelic=assume_biallelic,
            )
