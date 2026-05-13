from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .fstats import _build_blocks, _prepare_inputs


_RDS_NA_REAL = bytes.fromhex("7ff00000000007a2")


@dataclass(frozen=True)
class QPExportResult:
    """Summary of a qpAdm/qpGraph/qpWave blocked-statistics export."""

    outdir: Path
    populations: Tuple[str, ...]
    statistics: Tuple[str, ...]
    n_blocks: int
    n_snps: int
    files: Tuple[Path, ...]


def export_qp(
    data: Union[Any, Tuple[np.ndarray, np.ndarray, List[str]]],
    outdir: Union[str, Path],
    *,
    sample_labels: Optional[Sequence[str]] = None,
    populations: Optional[Sequence[str]] = None,
    tools: Sequence[str] = ("qpAdm", "qpGraph", "qpWave"),
    block_size: int = 5000,
    blocks: Optional[np.ndarray] = None,
    apply_correction: bool = True,
    overwrite: bool = False,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
    pseudohaploid: Union[bool, int] = False,
) -> QPExportResult:
    """
    Export blocked pairwise statistics for qpAdm, qpGraph, and qpWave.

    The output directory contains one file per population pair plus block-length
    files. qpGraph consumes the blocked ``f2`` files. qpAdm and qpWave consume
    the allele-product blocks (``ap``) and form their f4 matrices from them.

    Args:
        data: Either a ``SNPObject`` or a tuple ``(afs, counts, pops)`` where
            ``afs`` and ``counts`` have shape ``(n_snps, n_pops)``.
        outdir: Destination directory for the blocked-statistics files.
        sample_labels: Population label per sample when ``data`` is a
            ``SNPObject``. If omitted, the same defaults as ``f2`` are used.
        populations: Optional subset/order of populations to export. Defaults
            to all populations in the prepared allele-frequency table.
        tools: Any subset of ``{"qpAdm", "qpGraph", "qpWave"}``. The default
            writes everything needed by all three tools.
        block_size: Number of SNPs per jackknife block. Ignored if ``blocks``
            is provided.
        blocks: Optional explicit block label per SNP.
        apply_correction: Apply the small-sample f2 correction. SNPs with
            haplotype count <= 1 in either population are excluded from f2
            blocks when correction is enabled.
        overwrite: Replace existing files if ``True``. Existing target files
            raise ``FileExistsError`` by default.
        ancestry, laiobj, pseudohaploid: Passed through to the allele-frequency
            aggregation used by the native f-statistics implementation.

    Returns:
        ``QPExportResult`` with the exported populations, statistics, and files.
    """
    requested = _normalize_tools(tools)
    write_f2 = "qpgraph" in requested
    write_ap = bool({"qpadm", "qpwave"} & requested)
    if not write_f2 and not write_ap:
        raise ValueError("At least one of qpAdm, qpGraph, or qpWave must be requested.")

    afs, counts, pops = _prepare_inputs(
        data,
        sample_labels,
        ancestry=ancestry,
        laiobj=laiobj,
        pseudohaploid=pseudohaploid,
    )
    afs = np.asarray(afs, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if afs.ndim != 2 or counts.shape != afs.shape:
        raise ValueError("Prepared allele frequencies and counts must be 2D arrays with matching shapes.")

    pops = [str(pop) for pop in pops]
    if populations is not None:
        pops, keep = _subset_populations(pops, populations)
        afs = afs[:, keep]
        counts = counts[:, keep]
    _validate_population_names(pops)

    n_snps = int(afs.shape[0])
    block_ids, block_lengths = _build_blocks(n_snps, blocks, block_size)
    block_ids = np.asarray(block_ids, dtype=np.int64)
    block_lengths = np.asarray(block_lengths, dtype=np.int32)
    n_blocks = int(block_lengths.size)

    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    statistics: List[str] = []
    if write_f2:
        path = outpath / "block_lengths_f2.rds"
        _write_rds_int_vector(path, block_lengths, overwrite=overwrite)
        files.append(path)
        statistics.append("f2")
    if write_ap:
        path = outpath / "block_lengths_ap.rds"
        _write_rds_int_vector(path, block_lengths, overwrite=overwrite)
        files.append(path)
        statistics.append("ap")

    for i, j in _population_pair_indices(pops):
        pair_dir, pair_name = _pair_path_parts(pops[i], pops[j])
        dest_dir = outpath / pair_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        if write_f2:
            f2_est, f2_counts = _pair_f2_blocks(
                afs[:, i],
                afs[:, j],
                counts[:, i],
                counts[:, j],
                block_ids,
                block_lengths,
                apply_correction=apply_correction,
                force_zero=bool(i == j),
            )
            path = dest_dir / f"{pair_name}_f2.rds"
            _write_pair_matrix(path, f2_est, f2_counts, "f2", overwrite=overwrite)
            files.append(path)

        if write_ap:
            ap_est, ap_counts = _pair_ap_blocks(
                afs[:, i],
                afs[:, j],
                counts[:, i],
                counts[:, j],
                block_ids,
                block_lengths,
            )
            path = dest_dir / f"{pair_name}_ap.rds"
            _write_pair_matrix(path, ap_est, ap_counts, "ap", overwrite=overwrite)
            files.append(path)

    manifest_path = outpath / "qp_export.json"
    _write_manifest(
        manifest_path,
        populations=pops,
        requested_tools=tuple(sorted(requested)),
        statistics=tuple(statistics),
        block_lengths=block_lengths,
        n_snps=n_snps,
        apply_correction=apply_correction,
        overwrite=overwrite,
    )
    files.append(manifest_path)

    return QPExportResult(
        outdir=outpath,
        populations=tuple(pops),
        statistics=tuple(statistics),
        n_blocks=n_blocks,
        n_snps=n_snps,
        files=tuple(files),
    )


def _normalize_tools(tools: Sequence[str]) -> set[str]:
    if isinstance(tools, str):
        tools = (tools,)
    aliases = {
        "qpadm": "qpadm",
        "qpgraph": "qpgraph",
        "qpwave": "qpwave",
    }
    normalized: set[str] = set()
    for tool in tools:
        key = str(tool).replace("_", "").replace("-", "").lower()
        if key not in aliases:
            raise ValueError(f"Unsupported qp tool {tool!r}. Expected qpAdm, qpGraph, or qpWave.")
        normalized.add(aliases[key])
    return normalized


def _subset_populations(pops: Sequence[str], populations: Sequence[str]) -> Tuple[List[str], List[int]]:
    requested = [str(pop) for pop in populations]
    if len(set(requested)) != len(requested):
        raise ValueError("'populations' contains duplicate labels.")
    name_to_idx = {pop: idx for idx, pop in enumerate(pops)}
    missing = [pop for pop in requested if pop not in name_to_idx]
    if missing:
        raise ValueError(f"Population(s) not found: {', '.join(missing)}")
    return requested, [name_to_idx[pop] for pop in requested]


def _validate_population_names(pops: Sequence[str]) -> None:
    if len(set(pops)) != len(pops):
        raise ValueError("Population labels must be unique.")
    for pop in pops:
        if pop == "":
            raise ValueError("Population labels must not be empty.")
        if "/" in pop or "\\" in pop:
            raise ValueError(f"Population label {pop!r} cannot contain path separators.")


def _population_pair_indices(pops: Sequence[str]) -> Iterable[Tuple[int, int]]:
    seen: set[Tuple[str, str]] = set()
    for i, pop1 in enumerate(pops):
        for j, pop2 in enumerate(pops):
            pair = _ordered_pair(pop1, pop2)
            if pair in seen:
                continue
            seen.add(pair)
            yield i, j


def _ordered_pair(pop1: str, pop2: str) -> Tuple[str, str]:
    return (pop1, pop2) if pop1.encode("utf-8") <= pop2.encode("utf-8") else (pop2, pop1)


def _pair_path_parts(pop1: str, pop2: str) -> Tuple[str, str]:
    return _ordered_pair(pop1, pop2)


def _pair_f2_blocks(
    p1: np.ndarray,
    p2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    block_ids: np.ndarray,
    block_lengths: np.ndarray,
    *,
    apply_correction: bool,
    force_zero: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(p1) & np.isfinite(p2)
    if apply_correction:
        valid &= (n1 > 1) & (n2 > 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (p1 - p2) ** 2 - (p1 * (1.0 - p1)) / (n1 - 1.0) - (p2 * (1.0 - p2)) / (n2 - 1.0)
    else:
        valid &= (n1 > 0) & (n2 > 0)
        with np.errstate(invalid="ignore"):
            values = (p1 - p2) ** 2
    values = np.where(valid, values, 0.0)
    estimates, counts = _block_means(values, valid, block_ids, block_lengths)
    if force_zero:
        estimates = np.zeros_like(estimates)
    return estimates, counts


def _pair_ap_blocks(
    p1: np.ndarray,
    p2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    block_ids: np.ndarray,
    block_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(p1) & np.isfinite(p2) & (n1 > 0) & (n2 > 0)
    with np.errstate(invalid="ignore"):
        values = (p1 * p2 + (1.0 - p1) * (1.0 - p2)) / 2.0
    values = np.where(valid, values, 0.0)
    return _block_means(values, valid, block_ids, block_lengths)


def _block_means(
    values: np.ndarray,
    valid: np.ndarray,
    block_ids: np.ndarray,
    block_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_blocks = int(block_lengths.size)
    sums = np.bincount(block_ids, weights=values.astype(float, copy=False), minlength=n_blocks)
    valid_counts = np.bincount(block_ids, weights=valid.astype(float, copy=False), minlength=n_blocks)
    estimates = np.full(n_blocks, np.nan, dtype=float)
    np.divide(sums, valid_counts, out=estimates, where=valid_counts > 0)
    count_fraction = valid_counts / block_lengths.astype(float, copy=False)
    return estimates, count_fraction


def _write_pair_matrix(
    path: Path,
    estimates: np.ndarray,
    counts: np.ndarray,
    stat_name: str,
    *,
    overwrite: bool,
) -> None:
    matrix = np.column_stack([estimates.astype(float, copy=False), counts.astype(float, copy=False)])
    _write_rds_numeric_matrix(path, matrix, (stat_name, "counts"), overwrite=overwrite)


def _write_manifest(
    path: Path,
    *,
    populations: Sequence[str],
    requested_tools: Sequence[str],
    statistics: Sequence[str],
    block_lengths: np.ndarray,
    n_snps: int,
    apply_correction: bool,
    overwrite: bool,
) -> None:
    _ensure_writable(path, overwrite)
    payload: Dict[str, Any] = {
        "format": "snputils-qp-blocks-v1",
        "tools": list(requested_tools),
        "statistics": list(statistics),
        "populations": list(populations),
        "n_populations": len(populations),
        "n_snps": int(n_snps),
        "n_blocks": int(block_lengths.size),
        "block_lengths": [int(x) for x in block_lengths],
        "apply_correction": bool(apply_correction),
        "layout": {
            "block_lengths": "block_lengths_{stat}.rds",
            "pair": "{pop1}/{pop2}_{stat}.rds",
            "pair_columns": ["{stat}", "counts"],
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")


def _write_rds_int_vector(path: Path, values: np.ndarray, *, overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    buf = bytearray()
    _write_rds_header(buf)
    _write_int_vector(buf, [int(x) for x in np.asarray(values).ravel()])
    path.write_bytes(bytes(buf))


def _write_rds_numeric_matrix(
    path: Path,
    matrix: np.ndarray,
    colnames: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    _ensure_writable(path, overwrite)
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("RDS matrix payload must be 2D.")
    if len(colnames) != arr.shape[1]:
        raise ValueError("'colnames' length must match the matrix column count.")

    buf = bytearray()
    _write_rds_header(buf)
    attrs = [
        ("dim", ("int", [int(arr.shape[0]), int(arr.shape[1])])),
        ("dimnames", ("list", [None, ("str", [str(x) for x in colnames])])),
    ]
    _write_real_vector(buf, arr.reshape(-1, order="F"), attrs=attrs)
    path.write_bytes(bytes(buf))


def _write_rds_header(buf: bytearray) -> None:
    # Version-2 XDR serialization is stable and readable by base R without
    # requiring any package-specific writer.
    buf.extend(b"X\n")
    _write_int(buf, 2)
    _write_int(buf, 0x00040302)
    _write_int(buf, 0x00020300)


def _write_object(buf: bytearray, value: Any) -> None:
    if value is None:
        _write_int(buf, 254)
        return
    if isinstance(value, tuple) and len(value) == 2:
        kind, payload = value
        if kind == "int":
            _write_int_vector(buf, payload)
            return
        if kind == "str":
            _write_string_vector(buf, payload)
            return
        if kind == "list":
            _write_list_vector(buf, payload)
            return
    raise TypeError(f"Unsupported RDS object payload: {value!r}")


def _write_int_vector(buf: bytearray, values: Sequence[int]) -> None:
    _write_int(buf, 13)
    _write_int(buf, len(values))
    for value in values:
        _write_int(buf, int(value))


def _write_real_vector(
    buf: bytearray,
    values: Sequence[float],
    *,
    attrs: Optional[Sequence[Tuple[str, Any]]] = None,
) -> None:
    flags = 14 | (512 if attrs else 0)
    _write_int(buf, flags)
    _write_int(buf, len(values))
    for value in values:
        _write_double(buf, float(value))
    if attrs:
        _write_pairlist(buf, list(attrs))


def _write_string_vector(buf: bytearray, values: Sequence[str]) -> None:
    _write_int(buf, 16)
    _write_int(buf, len(values))
    for value in values:
        _write_charsxp(buf, str(value))


def _write_list_vector(buf: bytearray, values: Sequence[Any]) -> None:
    _write_int(buf, 19)
    _write_int(buf, len(values))
    for value in values:
        _write_object(buf, value)


def _write_pairlist(buf: bytearray, attrs: Sequence[Tuple[str, Any]]) -> None:
    if not attrs:
        _write_int(buf, 254)
        return
    name, value = attrs[0]
    _write_int(buf, 2 | 1024)
    _write_symbol(buf, name)
    _write_object(buf, value)
    _write_pairlist(buf, attrs[1:])


def _write_symbol(buf: bytearray, name: str) -> None:
    _write_int(buf, 1)
    _write_charsxp(buf, name)


def _write_charsxp(buf: bytearray, value: str) -> None:
    payload = value.encode("utf-8")
    _write_int(buf, 9 | 0x40000)
    _write_int(buf, len(payload))
    buf.extend(payload)


def _write_int(buf: bytearray, value: int) -> None:
    buf.extend(struct.pack(">i", int(value)))


def _write_double(buf: bytearray, value: float) -> None:
    if math.isnan(value):
        buf.extend(_RDS_NA_REAL)
    else:
        buf.extend(struct.pack(">d", value))


__all__ = ["QPExportResult", "export_qp"]
