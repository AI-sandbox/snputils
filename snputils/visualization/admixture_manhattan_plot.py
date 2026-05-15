import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt

from ._figure_export import default_savefig_kwargs, scatter_rasterized_for_path

def manhattan_plot(
    input_file: str, 
    colors: list,
    significance_threshold: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    fontsize: Optional[Dict[str, float]] = None,
    save: Optional[bool] = None,
    output_filename: Optional[str] = None,
):
    """
    Generates a Manhattan plot from an input file the results of an admixture mapping association. 
    The plot is highly customizable, allowing users to specify colors for each chromosome and apply bonferroni correction for p-values.

    Args:
        input_file: Path to the input file containing columns '#CHROM', 'POS', and 'P'. default plik2 output files are supported.
        colors: List of colors to apply to each chromosome. the chromosome number is used as an index to select the color.
        significance_threshold: Significance threshold for p-values. Default is 0.05.
        figsize: Optional tuple to specify figure dimensions (width, height).
        title: Plot title. If None, no title is shown.
        fontsize: Dictionary with font sizes for title, labels, and legend.
        save: If True, saves the plot to a file. If None, the plot is not saved.
        output_filename: Filename for saving the plot (``.pdf``, ``.svg``, ``.png``, ...).
    """
    # Read the input file
    df = pd.read_csv(input_file, sep='\t')

    # Calculate the maximum distance within each chromosome to scale absolute positions
    max_distance = 0
    for chrom, chrom_data in df.groupby('#CHROM'):
        chrom_max_pos = chrom_data['POS'].max()
        if chrom_max_pos > max_distance:
            max_distance = chrom_max_pos

    # Calculate absolute positions for each SNP
    df['ABS_POS'] = df['POS'] + max_distance * df['#CHROM']

    # Bonferroni threshold
    bonferroni_threshold = significance_threshold / len(df)

    # Create the plot
    plt.figure(figsize=figsize)
    chrom_offsets = {chrom: max_distance * (chrom - 1) for chrom in range(1, 23)}

    _rz = scatter_rasterized_for_path(output_filename) if output_filename else False

    # Display Manhattan plot points for each chromosome
    for i, (chrom, chrom_data) in enumerate(df.groupby('#CHROM')):
        chrom_data['ABS_POS'] = chrom_data['POS'] + chrom_offsets[chrom]
        plt.scatter(
            chrom_data['ABS_POS'],
            -np.log10(chrom_data['P']),
            color=colors[int(chrom+1) % len(colors)],
            rasterized=_rz,
        )

    # X-axis settings
    plt.xlim(0, 22 * max_distance)
    chrom_labels = [str(c) for c in range(1, 23)]
    chrom_positions = [chrom_offsets[c] + max_distance / 2 for c in range(1, 23)]
    plt.xticks(chrom_positions, chrom_labels)

    # Significance thresholds
    plt.axhline(y=-np.log10(bonferroni_threshold), color='r', linestyle='--', label='Bonferroni')

    # Labels and title
    if title:
        plt.title(title, fontsize=fontsize.get('title', 20))
    plt.xlabel('Chromosomes', fontsize=fontsize.get('xlabel', 15))
    plt.ylabel('-log10(p-value)', fontsize=fontsize.get('ylabel', 15))
    plt.legend(fontsize=fontsize.get('legend', 15))

    # Save the plot
    plt.tight_layout()
    if save:
        skw = default_savefig_kwargs(output_filename)
        plt.savefig(output_filename, **skw)
    plt.show()