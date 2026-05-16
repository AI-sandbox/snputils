import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt

from ._figure_export import default_savefig_kwargs, scatter_rasterized_for_path


_DEFAULT_COLORS = ["steelblue", "navy"]


def manhattan_plot(
    data: Union[str, pd.DataFrame],
    colors: Optional[list] = None,
    significance_threshold: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    fontsize: Optional[Dict[str, float]] = None,
    save: Optional[bool] = None,
    output_filename: Optional[str] = None,
):
    """Generate a Manhattan plot from association study results.

    Accepts either a file path or an in-memory :class:`pandas.DataFrame`.
    The input must contain columns ``#CHROM``, ``POS``, and ``P`` (p-values).

    Args:
        data:
            Path to a tab-separated results file or an in-memory
            :class:`~pandas.DataFrame` with columns ``#CHROM``, ``POS``, and ``P``.
            PLINK2-style output files are supported directly.
        colors:
            List of colors to apply per chromosome.  The chromosome number modulo
            ``len(colors)`` is used to select the color.  Defaults to
            ``["steelblue", "navy"]``.
        significance_threshold:
            Nominal significance threshold used to derive the Bonferroni-corrected
            threshold (``significance_threshold / n_variants``).  Default is 0.05.
        figsize:
            Optional ``(width, height)`` tuple passed to :func:`matplotlib.pyplot.figure`.
        title:
            Plot title.  If ``None`` no title is shown.
        fontsize:
            Mapping with optional keys ``'title'``, ``'xlabel'``, ``'ylabel'``, and
            ``'legend'`` controlling per-element font sizes.  Missing keys fall back to
            sensible defaults (20 for title, 15 for everything else).
        save:
            If ``True``, saves the figure to ``output_filename``.
        output_filename:
            Destination path for the saved figure (``.pdf``, ``.svg``, ``.png``, …).
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data, sep='\t')

    _colors = colors if colors is not None else _DEFAULT_COLORS
    _fs = fontsize or {}

    # Scale absolute genomic positions across chromosomes
    max_distance = 0
    for _, chrom_data in df.groupby('#CHROM'):
        chrom_max_pos = chrom_data['POS'].max()
        if chrom_max_pos > max_distance:
            max_distance = chrom_max_pos

    df['ABS_POS'] = df['POS'] + max_distance * df['#CHROM']

    bonferroni_threshold = significance_threshold / len(df)

    plt.figure(figsize=figsize)
    chrom_offsets = {chrom: max_distance * (chrom - 1) for chrom in range(1, 23)}

    _rz = scatter_rasterized_for_path(output_filename) if output_filename else False

    for chrom, chrom_data in df.groupby('#CHROM'):
        chrom_data = chrom_data.copy()
        chrom_data['ABS_POS'] = chrom_data['POS'] + chrom_offsets[chrom]
        plt.scatter(
            chrom_data['ABS_POS'],
            -np.log10(chrom_data['P']),
            color=_colors[int(chrom + 1) % len(_colors)],
            rasterized=_rz,
        )

    plt.xlim(0, 22 * max_distance)
    chrom_labels = [str(c) for c in range(1, 23)]
    chrom_positions = [chrom_offsets[c] + max_distance / 2 for c in range(1, 23)]
    plt.xticks(chrom_positions, chrom_labels)

    plt.axhline(y=-np.log10(bonferroni_threshold), color='r', linestyle='--', label='Bonferroni')

    if title:
        plt.title(title, fontsize=_fs.get('title', 20))
    plt.xlabel('Chromosomes', fontsize=_fs.get('xlabel', 15))
    plt.ylabel('-log10(p-value)', fontsize=_fs.get('ylabel', 15))
    plt.legend(fontsize=_fs.get('legend', 15))

    plt.tight_layout()
    if save:
        skw = default_savefig_kwargs(output_filename)
        plt.savefig(output_filename, **skw)
    plt.show()
