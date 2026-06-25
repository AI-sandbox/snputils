import logging
from typing import Optional, List, Any, Union, Iterator, Tuple, Dict, Sequence
from pathlib import Path
import gzip
import csv
import mmap

import numpy as np
import polars as pl
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
import pathlib 
log = logging.getLogger(__name__)


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


def _empty_genotype_array(n_variants: int, n_samples: int, sum_strands: bool) -> np.ndarray:
    if sum_strands:
        return np.empty((n_variants, n_samples), dtype=np.int8)
    return np.empty((n_variants, n_samples, 2), dtype=np.int8)


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


def _parse_ascii_ints(raw: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    lengths = (ends - starts).astype(np.int64, copy=False)
    if lengths.size == 0:
        return np.array([], dtype=np.int32)
    width = int(lengths.max())
    columns = np.arange(width, dtype=np.int64)
    indexes = starts[:, None] + columns[None, :]
    mask = columns[None, :] < lengths[:, None]
    digits = np.where(mask, raw[indexes] - ord("0"), 0).astype(np.int64, copy=False)
    powers = 10 ** (lengths[:, None] - columns[None, :] - 1)
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


def _parse_gt_sample_bytes(
    sample_bytes: bytes,
    *,
    format_value: Optional[bytes] = None,
    n_samples_total: int,
    sample_idxs: np.ndarray,
    sum_strands: bool,
) -> np.ndarray:
    sample_bytes = sample_bytes.rstrip(b"\r\n")
    n_selected = int(sample_idxs.size)
    if n_selected == 0:
        return np.empty((0,), dtype=np.int8)

    expected_gt_only_len = n_samples_total * 4 - 1
    if (format_value is None or format_value == b"GT") and len(sample_bytes) == expected_gt_only_len:
        raw = np.frombuffer(sample_bytes, dtype=np.uint8)
        maternal = raw[0::4]
        paternal = raw[2::4]
        if sample_idxs.size != n_samples_total:
            maternal = maternal[sample_idxs]
            paternal = paternal[sample_idxs]
        maternal = _ascii_gt_to_int(maternal)
        paternal = _ascii_gt_to_int(paternal)
        if sum_strands:
            return maternal + paternal
        return np.stack((maternal, paternal), axis=1)

    gt_index = 0 if format_value is None else _format_gt_index(format_value)
    sample_fields = sample_bytes.split(b"\t")
    genotype = np.empty((n_selected, 2), dtype=np.int8)
    for out_idx, sample_idx in enumerate(sample_idxs):
        value = _sample_gt_token(sample_fields[int(sample_idx)], gt_index)
        genotype[out_idx] = _parse_simple_gt_token(value)
    if sum_strands:
        return genotype.sum(axis=1, dtype=np.int8)
    return genotype


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


def _parse_simple_gt_token(value: Union[str, bytes]) -> tuple[int, int]:
    if isinstance(value, bytes):
        if len(value) == 0 or value[0] in b".-":
            first = -1
        else:
            first = int(value[0] - ord("0"))
        if len(value) > 2 and value[2] not in b".-":
            second = int(value[2] - ord("0"))
        else:
            second = -1
        return first, second

    value = str(value)
    if len(value) == 0 or value[0] in ".-":
        first = -1
    else:
        first = int(value[0])
    if len(value) > 2 and value[2] not in ".-":
        second = int(value[2])
    else:
        second = -1
    return first, second


def _parse_gt_sample_matrix(
    sample_values: np.ndarray,
    format_values: np.ndarray,
    *,
    sum_strands: bool,
) -> np.ndarray:
    height, n_selected = sample_values.shape
    if all(_format_gt_is_first(value) for value in format_values):
        encoded = np.ascontiguousarray(sample_values.astype("S3", copy=False))
        raw_gt = encoded.view(np.uint8).reshape(height, n_selected, 3)
        maternal = _ascii_gt_to_int(raw_gt[:, :, 0])
        paternal = _ascii_gt_to_int(raw_gt[:, :, 2])
        paternal[(raw_gt[:, :, 1] == ord(":")) | (raw_gt[:, :, 1] == 0)] = -1
        if sum_strands:
            return maternal + paternal
        genotype = np.empty((height, n_selected, 2), dtype=np.int8)
        genotype[:, :, 0] = maternal
        genotype[:, :, 1] = paternal
        return genotype

    genotype = np.empty((height, n_selected, 2), dtype=np.int8)
    for row_idx, format_value in enumerate(format_values):
        gt_index = _format_gt_index(format_value)
        for col_idx, sample_field in enumerate(sample_values[row_idx]):
            genotype[row_idx, col_idx] = _parse_simple_gt_token(_sample_gt_token(sample_field, gt_index))
    if sum_strands:
        return genotype.sum(axis=2, dtype=np.int8)
    return genotype


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

    def _read_mmap_gt_only(
        self,
        *,
        names: list[str],
        field_columns: list[str],
        sample_columns: list[str],
        sample_idxs: np.ndarray,
        sum_strands: bool,
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
                        genotypes=_empty_genotype_array(0, len(sample_columns), sum_strands),
                        sample_columns=sample_columns,
                        arrays={},
                    )

                first_tabs = np.flatnonzero(raw[int(body_starts[0]):int(body_ends[0])] == ord("\t")) + int(body_starts[0])
                tabs_per_record = 8 + n_samples_total
                if first_tabs.size != tabs_per_record:
                    raise ValueError("VCF records do not have the expected number of tab-delimited columns.")
                format_value = _decode_vcf_slice(raw, first_tabs[7] + 1, first_tabs[8])
                sample_bytes_len = int(body_ends[0] - first_tabs[8] - 1)
                if sample_columns and (format_value != "GT" or sample_bytes_len != n_samples_total * 4 - 1):
                    raise ValueError("The memory-mapped fast path requires GT-only sample fields.")

                n_selected = len(sample_columns)
                genotypes = _empty_genotype_array(n_records, n_selected, sum_strands)

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

                sample_offsets = sample_idxs.astype(np.int64, copy=False) * 4
                chunk_size = max(1, min(100_000, 2_000_000 // max(1, tabs_per_record)))

                for lo in range(0, n_records, chunk_size):
                    hi = min(n_records, lo + chunk_size)
                    byte_start = int(body_starts[lo])
                    byte_end = int(body_ends[hi - 1])
                    tabs = np.flatnonzero(raw[byte_start:byte_end] == ord("\t")) + byte_start
                    if tabs.size != (hi - lo) * tabs_per_record:
                        raise ValueError("VCF records do not all have the expected number of tab-delimited columns.")
                    tabs = tabs.reshape(hi - lo, tabs_per_record)

                    if n_selected:
                        sample_starts = tabs[:, 8] + 1
                        offsets = sample_starts[:, None] + sample_offsets[None, :]
                        maternal = _ascii_gt_to_int(raw[offsets])
                        paternal = _ascii_gt_to_int(raw[offsets + 2])
                        if sum_strands:
                            genotypes[lo:hi] = maternal + paternal
                        else:
                            genotypes[lo:hi, :, 0] = maternal
                            genotypes[lo:hi, :, 1] = paternal

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
        sum_strands: bool,
    ) -> SNPObject:
        n_samples_total = len(names) - 9
        if n_samples_total < 0:
            raise ValueError("Malformed VCF header with fewer than 9 fixed columns.")

        include = set(field_columns)
        n_selected = len(sample_columns)
        tabs_per_record = 8 + n_samples_total
        sample_offsets = sample_idxs.astype(np.int64, copy=False) * 4
        block_size = (1 if n_samples_total > 128 and n_selected <= 16 else 8) * 1024 * 1024

        gt_chunks: list[np.ndarray] = []
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

                    if n_samples_total > 128:
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

                    if n_selected:
                        sample_starts = tabs[:, 8] + 1
                        if not _field_matches_value(raw, tabs[:, 7] + 1, tabs[:, 8], "GT"):
                            raise ValueError("The block fast path requires GT-only sample fields.")
                        expected_sample_bytes = n_samples_total * 4 - 1
                        if not np.all((content_ends - sample_starts) == expected_sample_bytes):
                            raise ValueError("The block fast path requires fixed-width diploid GT sample fields.")
                        offsets = sample_starts[:, None] + sample_offsets[None, :]
                        maternal = _ascii_gt_to_int(raw[offsets])
                        paternal = _ascii_gt_to_int(raw[offsets + 2])
                        if sum_strands:
                            gt_chunks.append(maternal + paternal)
                        else:
                            gt = np.empty((height, n_selected, 2), dtype=np.int8)
                            gt[:, :, 0] = maternal
                            gt[:, :, 1] = paternal
                            gt_chunks.append(gt)

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

        if n_selected == 0:
            first_chunk_list = next((chunks for chunks in array_chunks.values() if chunks), None)
            n_records = int(sum(chunk.shape[0] for chunk in first_chunk_list)) if first_chunk_list is not None else 0
            genotypes = _empty_genotype_array(n_records, 0, sum_strands)
        elif gt_chunks:
            genotypes = np.concatenate(gt_chunks, axis=0)
        elif sum_strands:
            genotypes = np.empty((0, n_selected), dtype=np.int8)
        else:
            genotypes = np.empty((0, n_selected, 2), dtype=np.int8)

        arrays = {
            field: np.concatenate(chunks, axis=0)
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
        sum_strands: bool,
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

        gt_chunks: list[np.ndarray] = []
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

                    if n_selected:
                        format_starts = tabs[:, 7] + 1
                        format_ends = tabs[:, 8]
                        if _field_is_gt_first(raw, format_starts, format_ends):
                            gt_offset = 0
                        elif _field_matches_value(raw, format_starts, format_ends, "DP:GT"):
                            gt_offset = 2
                        else:
                            raise ValueError("The simple FORMAT fast path requires GT first or DP:GT FORMAT fields.")

                        sample_starts = tabs[:, 8 + sample_idxs] + 1
                        if gt_offset:
                            sample_ends = np.empty_like(sample_starts)
                            for out_idx, sample_idx in enumerate(sample_idxs):
                                sample_idx = int(sample_idx)
                                if sample_idx == n_samples_total - 1:
                                    sample_ends[:, out_idx] = content_ends
                                else:
                                    sample_ends[:, out_idx] = tabs[:, 9 + sample_idx]
                            if not np.all(sample_ends - sample_starts >= gt_offset + 3):
                                raise ValueError("Sample fields are too short for the requested GT offset.")

                        sep = raw[sample_starts + gt_offset + 1]
                        if not np.all((sep == ord("|")) | (sep == ord("/"))):
                            raise ValueError("The simple FORMAT fast path requires diploid GT calls.")

                        maternal = _ascii_gt_to_int(raw[sample_starts + gt_offset])
                        paternal = _ascii_gt_to_int(raw[sample_starts + gt_offset + 2])
                        if sum_strands:
                            gt_chunks.append(maternal + paternal)
                        else:
                            gt = np.empty((height, n_selected, 2), dtype=np.int8)
                            gt[:, :, 0] = maternal
                            gt[:, :, 1] = paternal
                            gt_chunks.append(gt)

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

        if n_selected == 0:
            first_chunk_list = next((chunks for chunks in array_chunks.values() if chunks), None)
            n_records = int(sum(chunk.shape[0] for chunk in first_chunk_list)) if first_chunk_list is not None else 0
            genotypes = _empty_genotype_array(n_records, 0, sum_strands)
        elif gt_chunks:
            genotypes = np.concatenate(gt_chunks, axis=0)
        elif sum_strands:
            genotypes = np.empty((0, n_selected), dtype=np.int8)
        else:
            genotypes = np.empty((0, n_selected, 2), dtype=np.int8)

        arrays = {
            field: np.concatenate(chunks, axis=0)
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
        sum_strands: bool,
    ) -> SNPObject:
        n_samples_total = len(names) - 9
        if n_samples_total < 0:
            raise ValueError("Malformed VCF header with fewer than 9 fixed columns.")

        include = set(field_columns)
        n_records = _count_vcf_records(self._filename, region_filter=region_filter, separator="\t")
        n_selected = len(sample_columns)
        genotypes = _empty_genotype_array(n_records, n_selected, sum_strands)

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
            sample_bytes_len = int(first_raw.size - first_tabs[8] - 1)
            if sample_columns and (format_value != "GT" or sample_bytes_len != n_samples_total * 4 - 1):
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

            sample_offsets = sample_idxs.astype(np.int64, copy=False) * 4
            row = 0
            remainder = first_line
            block_size = (1 if n_samples_total > 128 and n_selected <= 16 else 8) * 1024 * 1024

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

                    if n_samples_total > 128:
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

                    if n_selected:
                        sample_starts = tabs[:, 8] + 1
                        if not _field_matches_value(raw, tabs[:, 7] + 1, tabs[:, 8], "GT"):
                            raise ValueError("The block fast path requires GT-only sample fields.")
                        expected_sample_bytes = n_samples_total * 4 - 1
                        if not np.all((content_ends - sample_starts) == expected_sample_bytes):
                            raise ValueError("The block fast path requires fixed-width diploid GT sample fields.")
                        offsets = sample_starts[:, None] + sample_offsets[None, :]
                        maternal = _ascii_gt_to_int(raw[offsets])
                        paternal = _ascii_gt_to_int(raw[offsets + 2])
                        if sum_strands:
                            genotypes[lo:hi] = maternal + paternal
                        else:
                            genotypes[lo:hi, :, 0] = maternal
                            genotypes[lo:hi, :, 1] = paternal

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
        sum_strands: bool,
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
        genotypes = _empty_genotype_array(n_records, n_selected, sum_strands)

        filter_columns = []
        if region_filter is not None:
            chrom_column = "#CHROM" if "#CHROM" in names else "CHROM"
            filter_columns = [chrom_column, "POS"]

        parsing_columns = ["FORMAT"] if n_selected else []
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
            if n_selected:
                genotypes[offset:offset + height] = _parse_gt_sample_matrix(
                    frame[sample_columns].to_numpy(dtype=object),
                    frame["FORMAT"].to_numpy(dtype=object),
                    sum_strands=bool(sum_strands),
                )

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
        sum_strands: bool = False,
        separator: Optional[str] = None,
    ) -> SNPObject:
        """
        Read a VCF file into an :class:`~snputils.snp.genobj.snpobj.SNPObject`.

        By default, the reader loads the core VCF variant columns ``CHROM``,
        ``POS``, ``ID``, ``REF``, ``ALT``, ``QUAL``, and ``FILTER``, plus all
        sample genotype columns. Genotypes are read from the ``GT`` FORMAT
        field and returned as an ``int8`` array. With ``sum_strands=False``,
        ``genotypes`` has shape ``(n_variants, n_samples, 2)``; with
        ``sum_strands=True``, the two alleles are summed into shape
        ``(n_variants, n_samples)``.

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
            sum_strands: If ``True``, sum the two diploid alleles per sample and
                return dosages in ``genotypes``. If ``False``, keep the two
                allele columns separate.
            separator: Optional column separator. If omitted, the separator is
                detected from the VCF header. Tab-delimited files use optimized
                byte parsers when possible; other separators use the pandas
                chunked parser.

        Returns:
            SNPObject: Object containing selected genotype, sample, and variant
            fields.
        """
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

        if detected_separator != "\t":
            return self._read_pandas_chunks(
                names=names,
                field_columns=field_columns,
                sample_columns=sample_columns,
                region_filter=region_filter,
                sum_strands=bool(sum_strands),
                separator=detected_separator,
            )

        try:
            if Path(self._filename).suffixes[-2:] == [".vcf", ".gz"]:
                use_streaming_gt_only = (
                    bool(sum_strands)
                    or not sample_columns
                    or not _first_record_is_fixed_width_gt_only(self._filename, len(names) - 9)
                )
                if use_streaming_gt_only:
                    return self._read_block_gt_only_streaming(
                        names=names,
                        field_columns=field_columns,
                        sample_columns=sample_columns,
                        sample_idxs=sample_idxs,
                        region_filter=region_filter,
                        sum_strands=bool(sum_strands),
                    )
                return self._read_block_gt_only(
                    names=names,
                    field_columns=field_columns,
                    sample_columns=sample_columns,
                    sample_idxs=sample_idxs,
                    region_filter=region_filter,
                    sum_strands=bool(sum_strands),
                )
            return self._read_block_gt_only(
                names=names,
                field_columns=field_columns,
                sample_columns=sample_columns,
                sample_idxs=sample_idxs,
                region_filter=region_filter,
                sum_strands=bool(sum_strands),
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
                    sum_strands=bool(sum_strands),
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
                    sum_strands=bool(sum_strands),
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
        sum_strands: bool = False,
        separator: Optional[str] = None,
        chunk_size: int = 10_000,
    ) -> Iterator[SNPObject]:
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

        def flush() -> Optional[SNPObject]:
            nonlocal records_count
            if records_count == 0:
                return None
            if len(sample_columns) == 0:
                if sum_strands:
                    gt = np.empty((records_count, 0), dtype=np.int8)
                else:
                    gt = np.empty((records_count, 0, 2), dtype=np.int8)
            else:
                gt = np.stack(records["gt"], axis=0)
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
                    records["gt"].append(
                        _parse_gt_sample_bytes(
                            parts[9],
                            format_value=parts[8],
                            n_samples_total=n_samples_total,
                            sample_idxs=sample_indices,
                            sum_strands=bool(sum_strands),
                        )
                    )
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
        sum_strands: bool,
    ) -> np.ndarray:
        if not sample_columns:
            return _empty_genotype_array(vcf.height, 0, sum_strands)

        # Process maternal strand:
        # Extract the first position from genotype, e.g., 0|1 -> 0.
        # Replace missing values codified as ".", "-", or "" with -1 for integer casting.
        genotype_maternal = (
            vcf[sample_columns]
            .select(pl.all().str.slice(0, length=1))
            .select(pl.all().replace({".": -1, "-": -1, "": -1}))
            .cast(pl.Int8)
        )

        # Process paternal strand:
        # Extract the third position from genotype, e.g., 0|1 -> 1.
        # Convert ":" to ".." such that if only maternal strand is present,
        # paternal strand is set to missing, e.g., 0:0.982 -> 0..0.982 -> . -> -1.
        genotype_paternal = (
            vcf[sample_columns]
            .select(pl.all().str.replace_all("-1", "."))
            .select(pl.all().str.replace(":", ".."))
            .select(pl.all().str.slice(2, length=1))
            .select(pl.all().replace({".": -1, "": -1}))
            .cast(pl.Int8)
        )

        genotypes = np.dstack((genotype_maternal, genotype_paternal))
        if sum_strands:
            genotypes = genotypes.sum(axis=2, dtype=np.int8)

        return genotypes

    def _dataframe_to_snpobject(
        self,
        vcf: pl.DataFrame,
        field_columns: List[str],
        sample_columns: List[str],
        sum_strands: bool,
    ) -> SNPObject:
        genotypes = self._parse_genotypes(vcf, sample_columns, bool(sum_strands))

        if "#CHROM" in vcf.columns:
            chrom_column = "#CHROM"
        elif "CHROM" in vcf.columns:
            chrom_column = "CHROM"
        else:
            chrom_column = None

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
             sum_strands: Optional[bool] = False,
             separator: Optional[str] = None
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
            sum_strands: True if the maternal and paternal strands are to be summed together,
                False if the strands are to be stored separately.
            separator: Separator used in the pvar file. If None, the separator is automatically detected.
                If the automatic detection fails, please specify the separator manually.

        Returns:
            snpobj: SNPObject containing the data from the VCF file. The format and content
                of this object depend on the specified parameters and the content of the VCF file.
        """
        # TODO: add support for excluding GT

        log.info(f"Reading {self._filename}")

        try:
            field_columns, sample_columns, selected_column_idxs, col_dtypes, detected_separator = self._resolve_columns(
                fields=fields,
                exclude_fields=exclude_fields,
                samples=samples,
                separator=separator,
            )

            # Read the VCF file into a Polars DataFrame
            vcf = pl.read_csv(
                self._filename,
                comment_prefix="##",
                has_header=True,
                separator=detected_separator,
                columns=selected_column_idxs,
                schema_overrides=col_dtypes,
            )

            log.debug("vcf polars read")

            snpobj = self._dataframe_to_snpobject(
                vcf=vcf,
                field_columns=field_columns,
                sample_columns=sample_columns,
                sum_strands=bool(sum_strands),
            )

            log.info(f"Finished reading {self.filename}")

            return snpobj

        except Exception as e:
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
                sum_strands=bool(sum_strands),
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
        sum_strands: Optional[bool] = False,
        separator: Optional[str] = None,
        chunk_size: int = 10_000,
    ) -> Iterator[SNPObject]:
        """
        Stream a VCF in variant chunks using the Polars backend.
        """
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
        filter_columns = []
        chrom_column = "#CHROM" if "#CHROM" in col_dtypes else "CHROM" if "CHROM" in col_dtypes else None
        if variant_ids is not None:
            filter_columns.extend(["ID", "POS", "REF", "ALT"])
            if chrom_column is not None:
                filter_columns.append(chrom_column)
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
                .select(scan_columns)
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
                    sum_strands=bool(sum_strands),
                )

        if pending is not None and pending.height > 0:
            yield self._dataframe_to_snpobject(
                vcf=pending,
                field_columns=field_columns,
                sample_columns=sample_columns,
                sum_strands=bool(sum_strands),
            )
