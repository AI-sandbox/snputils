"""
Generate chromosome painting visualizations.

The single public entry point is :func:`chromosome_painting`, which accepts a
:class:`~snputils.ancestry.genobj.local.LocalAncestryObject`, one or more MSP
files, or one or more BED files and dispatches to the appropriate internal
pipeline.

Adapts the drawing logic from Tagore (https://github.com/jordanlab/tagore),
reimplementing it internally and using CairoSVG (instead of RSVG) for SVG
conversion.
"""

from __future__ import annotations

import logging
import os
import pathlib
import pickle
import re
import tempfile
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from snputils.ancestry.io.local.write.msp_to_bed import (
    laiobj_sample_to_bed_df,
    msp_files_to_bed,
)
from snputils.visualization.constants import CHROM_SIZES, COORDINATES

if TYPE_CHECKING:
    from snputils.ancestry.genobj.local import LocalAncestryObject

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal SVG helpers
# ---------------------------------------------------------------------------


def _load_svg_base() -> Tuple[str, str]:
    base_svg_path = pathlib.Path(__file__).parent / "_data" / "base.svg.p"
    try:
        with open(base_svg_path, "rb") as fh:
            svg_header, svg_footer = pickle.load(fh)
        return svg_header, svg_footer
    except Exception as e:
        log.error(f"Failed to load base SVG file from {base_svg_path}: {e}")
        raise


def _draw_svg(
    input_bed: str,
    output_svg: str,
    build: str,
    svg_header: str,
    svg_footer: str,
    verbose: bool = False,
) -> None:
    if verbose:
        log.info(f"Drawing SVG for {input_bed}")
    try:
        with open(input_bed, "r") as bed_fh, open(output_svg, "w") as svg_fh:
            svg_fh.write(svg_header)
            polygons = ""
            line_num = 1
            for line in bed_fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip().split("\t")
                if len(parts) != 7:
                    raise ValueError(
                        f"Line {line_num} in {input_bed} does not have 7 columns."
                    )
                chrm, start, stop, feature, size, col, chrcopy = parts
                chrm = chrm.replace("chr", "")
                try:
                    start = int(start)
                    stop = int(stop)
                    size = float(size)
                    feature = int(feature)
                    chrcopy = int(chrcopy)
                except ValueError as e:
                    raise ValueError(f"Conversion error on line {line_num}: {e}") from e
                if not (0 <= size <= 1):
                    log.warning(
                        f"Feature size {size} on line {line_num} unclear. Defaulting to 1."
                    )
                    size = 1
                if not re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", col):
                    log.warning(
                        f"Feature color {col} on line {line_num} unclear. Defaulting to #000000."
                    )
                    col = "#000000"
                if chrcopy not in [1, 2]:
                    log.warning(
                        f"Feature chromosome copy {chrcopy} on line {line_num} unclear. Skipping..."
                    )
                    line_num += 1
                    continue

                if feature == 0:  # Rectangle
                    feat_start = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    feat_end = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    width = COORDINATES[chrm]["width"] * size / 2
                    x_pos = (
                        COORDINATES[chrm]["cx"] - width
                        if chrcopy == 1
                        else COORDINATES[chrm]["cx"]
                    )
                    svg_fh.write(
                        f'<rect x="{x_pos}" y="{COORDINATES[chrm]["cy"] + feat_start}"'
                        f' fill="{col}" width="{width}" height="{feat_end - feat_start}"/>\n'
                    )
                elif feature == 1:  # Circle
                    feat_start = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    feat_end = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    radius = COORDINATES[chrm]["width"] * size / 4
                    x_pos = (
                        COORDINATES[chrm]["cx"] - COORDINATES[chrm]["width"] / 4
                        if chrcopy == 1
                        else COORDINATES[chrm]["cx"] + COORDINATES[chrm]["width"] / 4
                    )
                    svg_fh.write(
                        f'<circle fill="{col}" cx="{x_pos}"'
                        f' cy="{COORDINATES[chrm]["cy"] + (feat_start + feat_end) / 2}"'
                        f' r="{radius}"/>\n'
                    )
                elif feature == 2:  # Triangle
                    feat_start = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    feat_end = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    if chrcopy == 1:
                        x_pos = COORDINATES[chrm]["cx"] - COORDINATES[chrm]["width"] / 2
                        sx_pos = 38.2 * size
                    else:
                        x_pos = COORDINATES[chrm]["cx"] + COORDINATES[chrm]["width"] / 2
                        sx_pos = -38.2 * size
                    y_pos = COORDINATES[chrm]["cy"] + (feat_start + feat_end) / 2
                    sy_pos = 21.5 * size
                    polygons += (
                        f'<polygon fill="{col}" points="'
                        f'{x_pos - sx_pos},{y_pos - sy_pos} '
                        f'{x_pos},{y_pos} '
                        f'{x_pos - sx_pos},{y_pos + sy_pos}"/>\n'
                    )
                elif feature == 3:  # Line
                    y_pos1 = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    y_pos2 = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    y_pos = (y_pos1 + y_pos2) / 2 + COORDINATES[chrm]["cy"]
                    if chrcopy == 1:
                        x1 = COORDINATES[chrm]["cx"] - COORDINATES[chrm]["width"] / 2
                        x2 = COORDINATES[chrm]["cx"]
                    else:
                        x1 = COORDINATES[chrm]["cx"]
                        x2 = COORDINATES[chrm]["cx"] + COORDINATES[chrm]["width"] / 2
                    svg_fh.write(
                        f'<line fill="none" stroke="{col}" stroke-miterlimit="10"'
                        f' x1="{x1}" y1="{y_pos}" x2="{x2}" y2="{y_pos}"/>\n'
                    )
                else:
                    log.warning(
                        f"Feature type {feature} unclear on line {line_num}. Skipping..."
                    )
                line_num += 1
            svg_fh.write(svg_footer)
            svg_fh.write(polygons)
            svg_fh.write("</svg>")
    except Exception as e:
        log.error(f"Error drawing SVG from {input_bed}: {e}")
        raise


# ---------------------------------------------------------------------------
# Internal per-file painting
# ---------------------------------------------------------------------------


def _paint_bed_file(
    bed_file: str,
    output_prefix: pathlib.Path,
    build: str,
    output_format: str,
    force: bool,
    verbose: bool,
    show: bool,
) -> str:
    """Render a single BED file to SVG and then to the requested format."""
    svg_file = str(output_prefix) + ".svg"
    output_file = str(output_prefix) + f".{output_format}"

    svg_header, svg_footer = _load_svg_base()

    if force or not os.path.exists(svg_file):
        if verbose:
            log.info(f"Generating SVG from {bed_file}")
        _draw_svg(
            input_bed=bed_file,
            output_svg=svg_file,
            build=build,
            svg_header=svg_header,
            svg_footer=svg_footer,
            verbose=verbose,
        )

    if force or not os.path.exists(output_file):
        if verbose:
            log.info(f"Converting SVG to {output_format.upper()}")
        import cairosvg

        if output_format == "png":
            cairosvg.svg2png(url=svg_file, write_to=output_file)
        else:
            cairosvg.svg2pdf(url=svg_file, write_to=output_file)

        if verbose:
            log.info(f"Saved {output_file}")

        if show and output_format == "png":
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt

            img = mpimg.imread(output_file)
            plt.figure(figsize=(12, 12))
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.show()
        elif show and output_format == "pdf":
            log.warning("show=True is only supported for PNG output")

    return output_file


# ---------------------------------------------------------------------------
# Internal dispatch implementations
# ---------------------------------------------------------------------------


def _paint_from_laiobj(
    laiobj: LocalAncestryObject,
    sample_ids: List[str],
    painting_dir: pathlib.Path,
    build: str,
    color_map: Optional[Union[str, Dict]],
    num_labels: int,
    fill_empty: bool,
    output_format: str,
    force: bool,
    verbose: bool,
    show: bool,
) -> List[str]:
    output_files: List[str] = []
    for sid in sample_ids:
        if verbose:
            log.info(f"Generating chromosome painting for sample '{sid}'")
        bed_df = laiobj_sample_to_bed_df(
            laiobj=laiobj,
            sample_id=sid,
            color_map=color_map,
            num_labels=num_labels,
            fill_empty=fill_empty,
            build=build,
        )
        safe_name = sid.replace(".", "_").replace("/", "_")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bed", prefix=f"{safe_name}_", delete=False
        ) as tmp:
            bed_df.to_csv(tmp, sep="\t", index=False)
            tmp_path = tmp.name
        try:
            output_file = _paint_bed_file(
                bed_file=tmp_path,
                output_prefix=painting_dir / safe_name,
                build=build,
                output_format=output_format,
                force=force,
                verbose=verbose,
                show=show,
            )
            output_files.append(output_file)
        finally:
            try:
                os.remove(tmp_path)
            except OSError as e:
                log.warning(f"Could not remove temporary BED file {tmp_path}: {e}")
    return output_files


def _paint_from_msp(
    msp_files: List[str],
    sample_ids: Optional[List[str]],
    painting_dir: pathlib.Path,
    build: str,
    color_map: Optional[Union[str, Dict]],
    num_labels: int,
    fill_empty: bool,
    output_format: str,
    force: bool,
    verbose: bool,
    keep_bed_files: bool,
    show: bool,
) -> List[str]:
    bed_dir = painting_dir.parent / "bed_files"
    os.makedirs(bed_dir, exist_ok=True)

    bed_files = msp_files_to_bed(
        msp_files=msp_files,
        root=bed_dir,
        sample_ids=sample_ids,
        color_map=color_map,
        num_labels=num_labels,
        build=build,
        fill_empty=fill_empty,
    )

    output_files: List[str] = []
    for bed_file in bed_files:
        sample_name = pathlib.Path(bed_file).stem
        output_file = _paint_bed_file(
            bed_file=bed_file,
            output_prefix=painting_dir / sample_name,
            build=build,
            output_format=output_format,
            force=force,
            verbose=verbose,
            show=show,
        )
        output_files.append(output_file)
        if not keep_bed_files:
            try:
                os.remove(bed_file)
            except OSError as e:
                log.warning(f"Could not remove BED file {bed_file}: {e}")

    if not keep_bed_files:
        try:
            os.rmdir(bed_dir)
        except OSError:
            pass

    return output_files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chromosome_painting(
    source: Union[
        LocalAncestryObject,
        str,
        pathlib.Path,
        List[Union[str, pathlib.Path]],
    ],
    output_dir: Union[str, pathlib.Path],
    sample_id: Optional[Union[str, List[str]]] = None,
    build: str = "hg38",
    color_map: Optional[Union[str, Dict]] = None,
    num_labels: int = 8,
    fill_empty: bool = True,
    output_format: str = "png",
    force: bool = True,
    verbose: bool = False,
    show: bool = False,
    keep_bed_files: bool = False,
) -> List[str]:
    """
    Generate chromosome paintings from a local ancestry source.

    Accepts a
    :class:`~snputils.ancestry.genobj.local.LocalAncestryObject`, one or
    more MSP files, or one or more BED files and dispatches to the
    appropriate internal pipeline automatically.

    **Source types**

    - :class:`~snputils.ancestry.genobj.local.LocalAncestryObject` —
      in-memory LAI data; ``chromosomes`` and ``physical_pos`` must be
      populated.
    - ``str`` / ``pathlib.Path`` ending with ``.msp`` or ``.msp.tsv`` —
      a single MSP file; also accepts a ``list`` of such paths spanning
      multiple chromosomes.
    - ``str`` / ``pathlib.Path`` ending with ``.bed`` — one pre-formatted
      BED file; also accepts a ``list`` to paint multiple samples at once.

    **Selecting samples**

    - ``sample_id=None`` (default) — paint every sample in the source.
    - ``sample_id="0001"`` — paint only the sample whose ID is ``"0001"``.
    - ``sample_id=["0001", "0002"]`` — paint a subset.

    ``sample_id`` is not applicable to BED sources (a BED file already
    represents one sample); it is silently ignored when BED files are
    provided.

    Args:
        source: The data source; see description above.
        output_dir: Directory where output files will be saved.
        sample_id: Sample identifier(s) to paint. ``None`` paints all
            samples. Accepts a single string or a list of strings.
        build: Genome build version (``'hg37'`` or ``'hg38'``).
        color_map: A TSV filename or a ``{int: hex_color}`` dict mapping
            numeric ancestry codes to hex color strings. Uses the default
            snputils palette when ``None``.
        num_labels: Number of distinct colors to generate when *color_map*
            is ``None``.
        fill_empty: If True, fill unassigned chromosome regions with a
            neutral grey color.
        output_format: Output format, ``'png'`` or ``'pdf'``.
        force: If True, overwrite existing output files.
        verbose: If True, emit progress log messages.
        show: If True, display each PNG in a matplotlib figure (PNG only).
        keep_bed_files: If True, retain intermediate BED files generated
            from MSP sources.

    Returns:
        List[str]: Paths to the generated output files, one per sample.

    Raises:
        ValueError: If the source type cannot be determined from the file
            extension, or if a requested *sample_id* is not found.

    Examples:
        Paint all samples from a LAI object::

            su.viz.chromosome_painting(lai, "paintings/")

        Paint a single sample::

            su.viz.chromosome_painting(lai, "paintings/", sample_id="0001")

        Paint a subset from MSP files::

            su.viz.chromosome_painting(
                ["chr1.msp", "chr2.msp"],
                "paintings/",
                sample_id=["0001", "0002"],
            )
    """
    # Lazy import to avoid circular dependency at module level
    from snputils.ancestry.genobj.local import LocalAncestryObject as _LAIObj

    output_dir = pathlib.Path(output_dir)
    painting_dir = output_dir / "paintings"
    os.makedirs(painting_dir, exist_ok=True)

    output_format = output_format.lower()
    if output_format not in ("png", "pdf"):
        raise ValueError("output_format must be 'png' or 'pdf'")

    # Normalize sample_id to Optional[List[str]]
    sample_ids: Optional[List[str]]
    if sample_id is None:
        sample_ids = None
    elif isinstance(sample_id, str):
        sample_ids = [sample_id]
    else:
        sample_ids = list(sample_id)

    # ---- LAI object --------------------------------------------------------
    if isinstance(source, _LAIObj):
        if sample_ids is None:
            if source.samples is None:
                raise ValueError(
                    "laiobj.samples is None; populate it or pass an explicit sample_id."
                )
            sample_ids = list(source.samples)
        return _paint_from_laiobj(
            laiobj=source,
            sample_ids=sample_ids,
            painting_dir=painting_dir,
            build=build,
            color_map=color_map,
            num_labels=num_labels,
            fill_empty=fill_empty,
            output_format=output_format,
            force=force,
            verbose=verbose,
            show=show,
        )

    # ---- File / list of files ----------------------------------------------
    paths = [source] if not isinstance(source, list) else source
    paths = [pathlib.Path(p) for p in paths]

    def _ext(p: pathlib.Path) -> str:
        name = p.name.lower()
        if name.endswith(".msp.tsv"):
            return "msp"
        if name.endswith(".msp"):
            return "msp"
        if name.endswith(".bed"):
            return "bed"
        raise ValueError(
            f"Cannot determine source type from '{p}'. "
            "Expected a .bed, .msp, or .msp.tsv file."
        )

    kind = _ext(paths[0])

    if kind == "msp":
        return _paint_from_msp(
            msp_files=[str(p) for p in paths],
            sample_ids=sample_ids,
            painting_dir=painting_dir,
            build=build,
            color_map=color_map,
            num_labels=num_labels,
            fill_empty=fill_empty,
            output_format=output_format,
            force=force,
            verbose=verbose,
            keep_bed_files=keep_bed_files,
            show=show,
        )

    # kind == "bed"
    if sample_ids is not None:
        log.warning(
            "sample_id is ignored when BED files are provided directly. "
            "Each BED file is painted as-is."
        )
    output_files: List[str] = []
    for bed_path in paths:
        output_file = _paint_bed_file(
            bed_file=str(bed_path),
            output_prefix=painting_dir / bed_path.stem,
            build=build,
            output_format=output_format,
            force=force,
            verbose=verbose,
            show=show,
        )
        output_files.append(output_file)
    return output_files
