"""
chromosome_painting.py

This script generates chromosome paintings by adapting the visualization logic from
Tagore (https://github.com/jordanlab/tagore). Instead of calling the external
Tagore utility, it reimplements the drawing logic internally and uses CairoSVG, 
replacing RSVG, to convert the generated SVG.
"""

import logging
import os
import pathlib
import pickle
import re
import sys
from typing import List, Optional, Union, Dict, Tuple
import cairosvg

from snputils.ancestry.io.local.write.msp_to_bed import msp_files_to_bed
from snputils.visualization.constants import CHROM_SIZES, COORDINATES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_svg_base() -> Tuple[str, str]:
    """
    Load the SVG header and footer from the base.svg.p file.

    The base.svg.p file is expected to be in the same directory as this script
    and should contain a pickled tuple of two strings: the SVG header and footer.

    Returns:
        Tuple[str, str]: A tuple containing (svg_header, svg_footer) strings
            needed to construct the complete SVG document.

    Raises:
        Exception: If the file cannot be found or loaded properly.
    """
    base_svg_path = pathlib.Path(__file__).parent / "base.svg.p"
    try:
        with open(base_svg_path, "rb") as fh:
            svg_header, svg_footer = pickle.load(fh)
        return svg_header, svg_footer
    except Exception as e:
        log.error(f"Failed to load base SVG file from {base_svg_path}: {e}")
        raise


def draw_svg(
    input_bed: str,
    output_svg: str,
    build: str,
    svg_header: str,
    svg_footer: str,
    verbose: bool = False,
) -> None:
    """
    Read the BED file and create an SVG file with the chromosome painting.

    Args:
        input_bed: Path to the BED file.
        output_svg: Path for the output SVG file.
        build: Genome build version (e.g. 'hg37' or 'hg38').
        svg_header: SVG header content.
        svg_footer: SVG footer content.
        verbose: If True, output verbose logging.
    """
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
                    log.error(f"Line {line_num} in {input_bed} does not have 7 columns.")
                    sys.exit(1)
                chrm, start, stop, feature, size, col, chrcopy = parts
                chrm = chrm.replace("chr", "")
                try:
                    start = int(start)
                    stop = int(stop)
                    size = float(size)
                    feature = int(feature)
                    chrcopy = int(chrcopy)
                except ValueError as e:
                    log.error(f"Conversion error on line {line_num}: {e}")
                    sys.exit(1)
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

                # Draw based on feature type
                if feature == 0:  # Rectangle
                    feat_start = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    feat_end = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    width = COORDINATES[chrm]["width"] * size / 2
                    x_pos = (
                        COORDINATES[chrm]["cx"] - width
                        if chrcopy == 1
                        else COORDINATES[chrm]["cx"]
                    )
                    y_pos = COORDINATES[chrm]["cy"] + feat_start
                    height = feat_end - feat_start
                    svg_fh.write(
                        f'<rect x="{x_pos}" y="{y_pos}" fill="{col}" width="{width}" height="{height}"/>\n'
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
                    y_pos = COORDINATES[chrm]["cy"] + (feat_start + feat_end) / 2
                    svg_fh.write(
                        f'<circle fill="{col}" cx="{x_pos}" cy="{y_pos}" r="{radius}"/>\n'
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
                        f'<polygon fill="{col}" points="{x_pos-sx_pos},{y_pos-sy_pos} {x_pos},{y_pos} {x_pos-sx_pos},{y_pos+sy_pos}"/>\n'
                    )
                elif feature == 3:  # Line
                    y_pos1 = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    y_pos2 = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[build][chrm]
                    y_pos = (y_pos1 + y_pos2) / 2 + COORDINATES[chrm]["cy"]
                    if chrcopy == 1:
                        x_pos1 = COORDINATES[chrm]["cx"] - COORDINATES[chrm]["width"] / 2
                        x_pos2 = COORDINATES[chrm]["cx"]
                        svg_fh.write(
                            f'<line fill="none" stroke="{col}" stroke-miterlimit="10" x1="{x_pos1}" y1="{y_pos}" x2="{x_pos2}" y2="{y_pos}"/>\n'
                        )
                    else:
                        x_pos1 = COORDINATES[chrm]["cx"]
                        x_pos2 = COORDINATES[chrm]["cx"] + COORDINATES[chrm]["width"] / 2
                        svg_fh.write(
                            f'<line fill="none" stroke="{col}" stroke-miterlimit="10" x1="{x_pos1}" y1="{y_pos}" x2="{x_pos2}" y2="{y_pos}"/>\n'
                        )
                else:
                    log.warning(f"Feature type {feature} unclear on line {line_num}. Skipping...")
                line_num += 1
            svg_fh.write(svg_footer)
            svg_fh.write(polygons)
            svg_fh.write("</svg>")
    except Exception as e:
        log.error(f"Error drawing SVG from {input_bed}: {e}")
        raise


def chromosome_painting_from_bed(
    bed_file: Union[str, pathlib.Path],
    output_prefix: Union[str, pathlib.Path],
    build: str = "hg37",
    output_format: str = "png",
    force: bool = True,
    verbose: bool = True,
    show: bool = False,
) -> str:
    """
    Generate a chromosome painting from a BED file.

    This function creates an SVG file from the BED input and then converts it to the specified format.
    Use this function if you already have a properly formatted BED file.

    Args:
        bed_file: Path to the BED file.
        output_prefix: Output prefix (SVG and output file will be saved as <output_prefix>.svg and <output_prefix>.[png|pdf]).
        build: Genome build version ('hg37' or 'hg38').
        output_format: Output format, either 'png' or 'pdf'.
        force: Overwrite existing files.
        verbose: If True, show verbose output.
        show: If True, show the output file (only works for PNG format).

    Returns:
        str: Path to the generated output file.

    Raises:
        ValueError: If output_format is not 'png' or 'pdf'.
    """
    bed_file = str(bed_file)
    output_prefix = str(output_prefix)
    svg_file = f"{output_prefix}.svg"
    output_format = output_format.lower()
    if output_format not in ["png", "pdf"]:
        raise ValueError("output_format must be either 'png' or 'pdf'")
    output_file = f"{output_prefix}.{output_format}"

    # Load SVG header and footer from base.svg.p
    svg_header, svg_footer = load_svg_base()

    if not force and os.path.exists(svg_file):
        log.info(f"Found existing SVG file at {svg_file}, skipping generation")
    else:
        if verbose:
            log.info(f"Generating SVG painting from {bed_file}")
        # Create the SVG painting from the BED file
        draw_svg(
            input_bed=bed_file,
            output_svg=svg_file,
            build=build,
            svg_header=svg_header,
            svg_footer=svg_footer,
            verbose=verbose,
        )

    if not force and os.path.exists(output_file):
        log.info(f"Found existing {output_format.upper()} file at {output_file}, skipping conversion")
    else:
        if verbose:
            log.info(f"Converting SVG to {output_format.upper()} format")
        # Convert the SVG to the desired format
        try:
            if output_format == "png":
                cairosvg.svg2png(url=svg_file, write_to=output_file)
            else:  # pdf
                cairosvg.svg2pdf(url=svg_file, write_to=output_file)
            
            if verbose:
                log.info(f"Successfully generated {output_format.upper()} file at {output_file}")
            
            if show and output_format == "png":
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                img = mpimg.imread(output_file)
                plt.figure(figsize=(12, 12))
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.show()
            elif show and output_format == "pdf":
                log.warning("Show functionality is only supported for PNG format")
        except Exception as e:
            log.error(f"Failed to convert SVG to {output_format.upper()} format: {e}")
            raise

    return output_file


def chromosome_paintings_from_msp_files(
    msp_files: List[str],
    output_dir: Union[str, pathlib.Path],
    build: str = "hg38",
    color_map: Optional[Union[str, Dict]] = None,
    num_labels: int = 8,
    fill_empty: bool = True,
    sample_from: int = 0,
    max_sample_count: int = -1,
    force: bool = True,
    verbose: bool = False,
    keep_bed_files: bool = False,
    show: bool = False,
    output_format: str = "png",
) -> List[str]:
    """
    Generate chromosome paintings from MSP (Local Ancestry Inference) files.
    This function first converts the MSP data to BED format and then creates
    SVG/PNG/PDF paintings.

    Use this function if you have MSP files from Local Ancestry Inference that
    need to be converted to BED format before painting.

    Args:
        msp_files: List of paths to MSP files.
        output_dir: Directory where output files will be saved.
        build: Genome build version (e.g. 'hg37' or 'hg38').
        color_map: A TSV filename or dict mapping ancestry names to colors.
        num_labels: Number of labels/colors to use in the color mapping, if a color map is not provided.
        fill_empty: Whether to fill empty chromosome regions.
        sample_from: Index of first sample to process.
        max_sample_count: Maximum number of samples to process (-1 for all).
        force: Overwrite existing files.
        verbose: Verbose output flag.
        keep_bed_files: Whether to keep intermediate BED files.
        show: If True, show the output file (only works for PNG format).
        output_format: Output format, either 'png' or 'pdf'.

    Returns:
        List of paths to generated output files.
    """
    output_dir = pathlib.Path(output_dir)
    bed_dir = output_dir / "bed_files"
    painting_dir = output_dir / "paintings"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bed_dir, exist_ok=True)
    os.makedirs(painting_dir, exist_ok=True)

    bed_files = msp_files_to_bed(
        msp_files=msp_files,
        root=bed_dir,
        sample_from=sample_from,
        max_sample_count=max_sample_count,
        color_map=color_map,
        num_labels=num_labels,
        build=build,
        fill_empty=fill_empty,
    )

    output_files = []
    for bed_file in bed_files:
        bed_path = pathlib.Path(bed_file)
        sample_name = bed_path.stem
        output_prefix = painting_dir / sample_name

        output_file = chromosome_painting_from_bed(
            bed_file=bed_file, 
            output_prefix=output_prefix, 
            build=build,
            output_format=output_format,
            force=force,
            verbose=verbose,
            show=show,
        )
        output_files.append(output_file)

        # Remove intermediate BED file if not needed
        if not keep_bed_files:
            try:
                os.remove(bed_file)
            except OSError as e:
                log.warning(f"Could not remove BED file {bed_file}: {e}")
    if not keep_bed_files:
        try:
            os.rmdir(bed_dir)
        except OSError:
            log.warning(f"Could not remove BED directory {bed_dir}, it may not be empty")

    return output_files
