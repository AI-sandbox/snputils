from __future__ import annotations

import logging
import os
import pathlib
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from snputils.ancestry.genobj.local import LocalAncestryObject

log = logging.getLogger(__name__)


def _load_color_map(color_map_path: str) -> Dict[str, str]:
    """
    Load color mappings from a TSV file.

    Args:
        color_map_path: Path to a TSV file with ``CODE``, ``NAME``, and
            ``COLOR`` columns.

    Returns:
        Dict[str, str]: Mapping from ancestry codes/names to hex color strings.
    """
    if color_map_path and os.path.exists(color_map_path):
        color_map_df = pd.read_csv(color_map_path, sep="\t")
        color_map: Dict[str, str] = {}
        for _, row in color_map_df.iterrows():
            color_map[row["CODE"]] = row["COLOR"]
            color_map[row["NAME"]] = row["COLOR"]
        return color_map
    return {}


def _generate_hex_colors(n: int) -> Dict[int, str]:
    """
    Generate hex color strings for integers ``0`` through ``n``.

    Uses the ``snputils_palette`` throughout, cycling when more colors are needed.

    Args:
        n: The maximum integer value (inclusive).

    Returns:
        Dict[int, str]: Mapping from integer index to hex color string.
    """
    from snputils.visualization.constants import get_palette_color  # lazy – avoids circular import

    return {i: get_palette_color(i) for i in range(n + 1)}


def _get_bed_data(
    msp_df: pd.DataFrame,
    sample: str,
    pop_order: Optional[Dict] = None,
) -> Dict:
    """
    Transform an MSP DataFrame into a BED-like dict for a single haplotype.

    Args:
        msp_df: DataFrame containing MSP file data.
        sample: Column name for the haplotype to process.
        pop_order: Optional mapping from numeric population code to label.

    Returns:
        Dict: BED-format data arrays for the haplotype.
    """
    ancestry_label = (
        lambda pop_numeric: pop_numeric if pop_order is None else pop_order[pop_numeric]
    )

    chm, spos, sgpos = [[val] for val in msp_df[["#chm", "spos", "sgpos"]].iloc[0]]
    epos, egpos = [], []
    anc = msp_df[sample].iloc[0]
    ancestry_labels = [ancestry_label(anc)]

    for i, row_anc in enumerate(msp_df[sample].iloc[1:]):
        row = i + 1
        if row_anc != anc:
            anc = row_anc
            epos.append(msp_df["epos"].iloc[row - 1])
            egpos.append(msp_df["egpos"].iloc[row - 1])
            chm.append(msp_df["#chm"].iloc[row])
            ancestry_labels.append(ancestry_label(row_anc))
            spos.append(msp_df["spos"].iloc[row])
            sgpos.append(msp_df["sgpos"].iloc[row])

    epos.append(msp_df["epos"].iloc[-1])
    egpos.append(msp_df["egpos"].iloc[-1])

    n = len(chm)
    return {
        "#chr": np.array([str(c) for c in chm]),
        "start": np.array(spos).astype(int),
        "stop": np.array(epos).astype(int),
        "feature": np.zeros(n, dtype=int),
        "size": np.ones(n, dtype=int),
        "ancestry": ancestry_labels,
    }


def _fill_missing_segments(bed_df: pd.DataFrame, build: str) -> pd.DataFrame:
    """
    Insert grey filler segments for unassigned regions of each chromosome.

    For every chromosome and haplotype copy, adds ``#8B8982`` segments
    before the first assigned region and after the last one so that the
    full chromosome length is covered.

    Args:
        bed_df: BED DataFrame with at least ``#chr``, ``start``, ``stop``,
            and ``chrCopy`` columns.
        build: Genome build key (``'hg37'`` or ``'hg38'``) used to look up
            chromosome sizes from
            :data:`~snputils.visualization.constants.CHROM_SIZES`.

    Returns:
        pandas.DataFrame: DataFrame with gap segments appended and sorted by
            chromosome, copy, and start position.
    """
    from snputils.visualization.constants import CHROM_SIZES  # lazy – avoids circular import

    new_segments = []
    for chrom, size in CHROM_SIZES[build].items():
        for chr_copy in [1, 2]:
            chrom_df = bed_df[
                (bed_df["#chr"].astype(str) == str(chrom))
                & (bed_df["chrCopy"] == chr_copy)
            ]
            if chrom_df.empty:
                continue
            first_start = chrom_df["start"].min()
            last_end = chrom_df["stop"].max()
            if first_start > 0:
                new_segments.append(
                    {
                        "#chr": chrom,
                        "start": 0,
                        "stop": first_start - 1,
                        "feature": 0,
                        "size": 1,
                        "color": "#8B8982",
                        "chrCopy": chr_copy,
                    }
                )
            if last_end < size:
                new_segments.append(
                    {
                        "#chr": chrom,
                        "start": last_end + 1,
                        "stop": size,
                        "feature": 0,
                        "size": 1,
                        "color": "#8B8982",
                        "chrCopy": chr_copy,
                    }
                )

    filled = pd.concat([bed_df, pd.DataFrame(new_segments)], ignore_index=True)
    filled.sort_values(by=["#chr", "chrCopy", "start"], inplace=True)
    return filled


def _fill_marker_gaps(
    bed_df: pd.DataFrame,
    build: str,
    extend_terminal: bool = False,
) -> pd.DataFrame:
    """
    Extend each painted segment to the next painted segment on the same
    chromosome copy so marker-sparse intervals do not appear as missing LAI.
    """
    from snputils.visualization.constants import CHROM_SIZES  # lazy – avoids circular import

    if bed_df.empty:
        return bed_df

    out = bed_df.copy()
    out.sort_values(by=["#chr", "chrCopy", "start"], inplace=True)
    for _, idx in out.groupby(["#chr", "chrCopy"], sort=False).groups.items():
        idx = list(idx)
        for current, nxt in zip(idx[:-1], idx[1:]):
            next_start = int(out.at[nxt, "start"])
            out.at[current, "stop"] = max(int(out.at[current, "stop"]), next_start - 1)

        if extend_terminal:
            last = idx[-1]
            chrom = str(out.at[last, "#chr"])
            chrom_key = chrom[3:] if chrom.lower().startswith("chr") else chrom
            chrom_size = CHROM_SIZES.get(build, {}).get(chrom_key)
            if chrom_size is not None:
                out.at[last, "stop"] = max(int(out.at[last, "stop"]), int(chrom_size))

    return out


def _sanitize_name(text: str) -> str:
    """
    Convert a string to a filesystem-safe identifier.

    Args:
        text: Input string.

    Returns:
        str: Lowercase string containing only alphanumeric characters and
            underscores.
    """
    text = text.lower().replace(" ", "_")
    return "".join(c for c in text if c.isalnum() or c == "_")


def msp_files_to_bed(
    msp_files: List[str],
    root: Union[str, pathlib.Path],
    sample_ids: Optional[List[str]] = None,
    sample_from: int = 0,
    max_sample_count: int = -1,
    num_labels: int = 8,
    build: str = "hg37",
    color_map: Optional[Union[str, Dict]] = None,
    fill_empty: bool = False,
    fill_marker_gaps: bool = False,
) -> List[str]:
    """
    Convert a list of MSP files to per-sample BED files.

    All MSP files are expected to share the same set of samples (same
    column layout). Each diploid sample is expanded into two haplotype
    rows that are combined across all chromosomes/files into a single BED
    file.

    Args:
        msp_files: Paths to the MSP files to process.
        root: Directory where the BED files will be written.
        sample_ids: Explicit list of sample identifiers to process. When
            provided, *sample_from* and *max_sample_count* are ignored.
        sample_from: Index of the first sample to process (0-based).
            Ignored when *sample_ids* is provided.
        max_sample_count: Maximum number of samples to process; ``-1``
            processes all. Ignored when *sample_ids* is provided.
        num_labels: Number of distinct colors to generate when *color_map*
            is ``None``.
        build: Genome build version (``'hg37'`` or ``'hg38'``).
        color_map: Either a path to a TSV color-map file (with ``CODE``,
            ``NAME``, and ``COLOR`` columns), or a ``{int: hex_color}``
            dict mapping numeric ancestry codes to hex strings. Uses the
            default snputils palette when ``None``.
        fill_empty: If True, insert grey filler segments for unassigned
            chromosome regions.
        fill_marker_gaps: If True, extend painted segments through
            inter-marker gaps until the next segment on the same chromosome
            copy. This avoids rendering sparse marker intervals as missing
            ancestry.

    Returns:
        List[str]: Paths to the generated BED files, one per sample.
    """
    root = pathlib.Path(root)
    all_files: List[str] = []

    with open(msp_files[0]) as f:
        _ = f.readline()
        second_line = f.readline()
    header = second_line.split("\t")
    samples = header[6:]
    paired_samples = [samples[i : i + 2] for i in range(0, len(samples), 2)]

    # Filter by explicit sample IDs when provided
    if sample_ids is not None:
        sid_set = set(sample_ids)
        paired_samples = [
            pair for pair in paired_samples
            if pair[0].rsplit(".", 1)[0] in sid_set
        ]
    else:
        sample_count = (
            min(max_sample_count, len(paired_samples))
            if max_sample_count > 0
            else len(paired_samples)
        )
        paired_samples = paired_samples[sample_from : sample_from + sample_count]

    for sample in paired_samples:
        aggregated_dfs = []
        for msp_file in msp_files:
            msp_df = pd.read_csv(msp_file, sep="\t", comment="#", names=header)

            with open(msp_file, "r") as fh:
                header_line = (
                    fh.readline().strip().replace("#Subpopulation order/codes: ", "")
                )
            original_mapping = {
                label.split("=")[0]: int(label.split("=")[1])
                for label in header_line.split("\t")
            }

            if isinstance(color_map, str):
                _color_dict = _load_color_map(color_map)
                color_dict: Dict[int, str] = {
                    original_mapping[k]: _color_dict[k]
                    for k in original_mapping.keys()
                }
            elif isinstance(color_map, dict):
                color_dict = color_map
            else:
                color_dict = _generate_hex_colors(num_labels)

            dfs = []
            for haploid in sample:
                last_value = haploid.split(".")[-1].strip()
                sample_bed_data = _get_bed_data(msp_df, haploid, pop_order=None)
                sample_bed_df = pd.DataFrame(sample_bed_data)
                sample_bed_df["color"] = sample_bed_df["ancestry"].replace(color_dict)
                sample_bed_df["chrCopy"] = int(last_value) + 1
                sample_bed_df.drop(columns=["ancestry"], inplace=True)
                dfs.append(sample_bed_df)

            aggregated_dfs.append(pd.concat(dfs, ignore_index=True))

        final_sample_df = pd.concat(aggregated_dfs, ignore_index=True)

        if fill_marker_gaps:
            final_sample_df = _fill_marker_gaps(
                final_sample_df,
                build=build,
                extend_terminal=fill_empty,
            )

        if fill_empty:
            final_sample_df = _fill_missing_segments(final_sample_df, build)

        sample_name = _sanitize_name(sample[0].rsplit(".", 1)[0])
        sample_bed_file = str(root / f"{sample_name}.bed")
        final_sample_df.to_csv(sample_bed_file, sep="\t", index=False)
        all_files.append(sample_bed_file)

    return all_files


def laiobj_sample_to_bed_df(
    laiobj: LocalAncestryObject,
    sample_id: str,
    color_map: Optional[Union[str, Dict]] = None,
    num_labels: int = 8,
    fill_empty: bool = True,
    build: str = "hg38",
    fill_marker_gaps: bool = False,
) -> pd.DataFrame:
    """
    Convert a single sample from a
    :class:`~snputils.ancestry.genobj.local.LocalAncestryObject` to a
    BED-format DataFrame.

    Consecutive windows that share the same ancestry code are merged into
    a single segment. Both haplotype copies are included with ``chrCopy``
    values of ``1`` and ``2``.

    Args:
        laiobj: A
            :class:`~snputils.ancestry.genobj.local.LocalAncestryObject`
            instance. Must have ``chromosomes`` and ``physical_pos``
            populated.
        sample_id: Sample identifier as it appears in ``laiobj.samples``.
        color_map: A TSV filename or a ``{int: hex_color}`` dict mapping
            numeric ancestry codes to hex color strings. Uses the default
            snputils palette when ``None``.
        num_labels: Number of distinct colors to generate when *color_map*
            is ``None``.
        fill_empty: If True, insert grey filler segments for unassigned
            chromosome regions.
        build: Genome build version (``'hg37'`` or ``'hg38'``), used when
            *fill_empty* or terminal marker-gap filling needs chromosome sizes.
        fill_marker_gaps: If True, extend painted segments through
            inter-marker gaps until the next segment on the same chromosome
            copy. This avoids rendering sparse marker intervals as missing
            ancestry.

    Returns:
        pandas.DataFrame: BED DataFrame with columns ``#chr``, ``start``,
            ``stop``, ``feature``, ``size``, ``color``, and ``chrCopy``.

    Raises:
        ValueError: If *sample_id* is not found in ``laiobj.samples``, or
            if ``chromosomes`` or ``physical_pos`` are ``None``.
    """
    samples = laiobj.samples
    if samples is None or sample_id not in samples:
        raise ValueError(
            f"Sample '{sample_id}' not found in laiobj.samples: {samples}"
        )
    if laiobj.chromosomes is None:
        raise ValueError(
            "laiobj.chromosomes must be set to generate a chromosome painting."
        )
    if laiobj.physical_pos is None:
        raise ValueError(
            "laiobj.physical_pos must be set to generate a chromosome painting."
        )

    sample_idx = samples.index(sample_id)
    hap_indices = [sample_idx * 2, sample_idx * 2 + 1]
    chromosomes = laiobj.chromosomes
    physical_pos = laiobj.physical_pos
    lai_array = laiobj.lai
    n_windows = laiobj.n_windows

    if isinstance(color_map, str):
        color_dict: Dict = _load_color_map(color_map)
    elif isinstance(color_map, dict):
        color_dict = color_map
    else:
        n_codes = int(lai_array.max()) + 1
        color_dict = _generate_hex_colors(max(n_codes - 1, num_labels - 1))

    rows = []
    for chr_copy, hap_idx in enumerate(hap_indices, start=1):
        hap_lai = lai_array[:, hap_idx]
        seg_start = 0
        for w in range(1, n_windows):
            boundary = (hap_lai[w] != hap_lai[w - 1]) or (
                chromosomes[w] != chromosomes[seg_start]
            )
            if boundary:
                rows.append(
                    {
                        "#chr": str(chromosomes[seg_start]),
                        "start": int(physical_pos[seg_start, 0]),
                        "stop": int(physical_pos[w - 1, 1]),
                        "feature": 0,
                        "size": 1,
                        "color": color_dict.get(int(hap_lai[seg_start]), "#000000"),
                        "chrCopy": chr_copy,
                    }
                )
                seg_start = w
        rows.append(
            {
                "#chr": str(chromosomes[seg_start]),
                "start": int(physical_pos[seg_start, 0]),
                "stop": int(physical_pos[n_windows - 1, 1]),
                "feature": 0,
                "size": 1,
                "color": color_dict.get(int(hap_lai[seg_start]), "#000000"),
                "chrCopy": chr_copy,
            }
        )

    bed_df = pd.DataFrame(rows)

    if fill_marker_gaps:
        bed_df = _fill_marker_gaps(
            bed_df,
            build=build,
            extend_terminal=fill_empty,
        )

    if fill_empty:
        bed_df = _fill_missing_segments(bed_df, build)

    return bed_df
