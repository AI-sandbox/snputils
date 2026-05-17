import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt

from ._figure_export import (
    default_savefig_kwargs,
    scatter_rasterized_for_path,
    style_association_axes,
)


_DEFAULT_COLORS = ["black", "grey"]
_LOG10_P_LABEL = r"$-\log_{10}(p)$"
_DEFAULT_FIGSIZE = (12.0, 6.0)  # width : height = 2 : 1
_X_PADDING_FRAC = 0.02  # fraction of data span added on each side


def manhattan_plot(
    data: Union[str, pd.DataFrame],
    colors: Optional[list] = None,
    significance_threshold: float = 0.05,
    point_size: Optional[float] = 7.0,
    line_width: float = 1.0,
    line_color: str = "r",
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
            ``["black", "grey"]``.
        significance_threshold:
            Nominal significance threshold used to derive the Bonferroni-corrected
            threshold (``significance_threshold / n_variants``).  Default is 0.05.
        point_size:
            Marker area for scatter points (matplotlib ``s``).  Default is 7.0.
        line_width:
            Width of the Bonferroni reference line.  Default is 1.0.
        line_color:
            Color of the Bonferroni reference line.  Default is ``"r"``.
        figsize:
            Optional ``(width, height)`` tuple passed to :func:`matplotlib.pyplot.figure`.
            Defaults to ``(12, 6)`` (2:1 aspect ratio).
        title:
            Plot title.  Default is ``None`` (no title).
        fontsize:
            Mapping with optional keys ``'title'``, ``'xlabel'``, and ``'ylabel'``
            controlling font sizes.  Missing keys fall back to sensible defaults
            (20 for title, 15 for axis labels).
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

    if figsize is None:
        figsize = _DEFAULT_FIGSIZE

    plt.figure(figsize=figsize)
    chrom_offsets = {chrom: max_distance * (chrom - 1) for chrom in range(1, 23)}

    _rz = scatter_rasterized_for_path(output_filename) if output_filename else False

    scatter_kw = {"rasterized": _rz}
    if point_size is not None:
        scatter_kw["s"] = point_size

    x_lo = np.inf
    x_hi = -np.inf
    for chrom, chrom_data in df.groupby('#CHROM'):
        chrom_data = chrom_data.copy()
        abs_pos = chrom_data['POS'].to_numpy(dtype=np.float64) + chrom_offsets[chrom]
        x_lo = min(x_lo, float(abs_pos.min()))
        x_hi = max(x_hi, float(abs_pos.max()))
        plt.scatter(
            abs_pos,
            -np.log10(chrom_data['P']),
            color=_colors[int(chrom + 1) % len(_colors)],
            **scatter_kw,
        )

    x_span = x_hi - x_lo
    x_pad = x_span * _X_PADDING_FRAC if x_span > 0 else max_distance * _X_PADDING_FRAC
    plt.xlim(x_lo - x_pad, x_hi + x_pad)
    chrom_labels = [str(c) for c in range(1, 23)]
    chrom_positions = [chrom_offsets[c] + max_distance / 2 for c in range(1, 23)]
    plt.xticks(chrom_positions, chrom_labels)

    plt.axhline(
        y=-np.log10(bonferroni_threshold),
        color=line_color,
        linestyle='--',
        linewidth=line_width,
    )

    if title:
        plt.title(title, fontsize=_fs.get('title', 20))
    plt.xlabel('Chromosome', fontsize=_fs.get('xlabel', 15))
    plt.ylabel(_LOG10_P_LABEL, fontsize=_fs.get('ylabel', 15))
    style_association_axes(hide_bottom=True, y_floor=0)

    plt.tight_layout()
    if save:
        skw = default_savefig_kwargs(output_filename)
        plt.savefig(output_filename, **skw)
    if output_filename is None:
        plt.show()
