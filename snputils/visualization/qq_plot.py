import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt

from ._figure_export import (
    default_savefig_kwargs,
    scatter_rasterized_for_path,
    style_association_axes,
)

_LOG10_P_LABEL = r"$-\log_{10}(p)$"


def qq_plot(
    data: Union[str, pd.DataFrame],
    color: str = "black",
    significance_threshold: float = 0.05,
    point_size: float = 7.0,
    line_width: float = 1.0,
    expected_line_color: str = "red",
    threshold_line_color: str = "orange",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    fontsize: Optional[Dict[str, float]] = None,
    save: Optional[bool] = None,
    output_filename: Optional[str] = None,
):
    """Generate a quantile-quantile (QQ) plot of association study p-values.

    Plots observed ``-log10(p)`` against the expected ``-log10(p)`` under the
    null hypothesis of no association (uniform distribution), together with the
    identity reference line and a Bonferroni significance threshold.

    Accepts either a file path or an in-memory :class:`pandas.DataFrame`.
    The input must contain a column ``P`` with p-values.

    Args:
        data:
            Path to a tab-separated results file or an in-memory
            :class:`~pandas.DataFrame` with a column ``P``.
            PLINK2-style output files are supported directly.
        color:
            Color for the scatter points.  Defaults to ``"black"``.
        significance_threshold:
            Nominal significance threshold used to derive the Bonferroni-corrected
            threshold (``significance_threshold / n_variants``).  Default is 0.05.
        point_size:
            Marker area for scatter points (matplotlib ``s``).  Default is 7.0.
        line_width:
            Width of the expected-null and Bonferroni reference lines.  Default is 1.0.
        expected_line_color:
            Color of the identity (expected under null) reference line.  Default is ``"red"``.
        threshold_line_color:
            Color of the Bonferroni threshold line.  Default is ``"orange"``.
        figsize:
            Optional ``(width, height)`` tuple passed to :func:`matplotlib.pyplot.figure`.
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

    _fs = fontsize or {}

    p_values = df['P'].dropna().values
    n = len(p_values)

    observed = np.sort(-np.log10(p_values))[::-1]
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    bonferroni_threshold = -np.log10(significance_threshold / n)

    _rz = scatter_rasterized_for_path(output_filename) if output_filename else False

    plt.figure(figsize=figsize)
    plt.scatter(expected, observed, color=color, s=point_size, rasterized=_rz)

    # Identity reference line (expected under null)
    max_val = max(expected.max(), observed.max())
    plt.plot(
        [0, max_val],
        [0, max_val],
        color=expected_line_color,
        linestyle='--',
        linewidth=line_width,
    )

    # Bonferroni threshold
    plt.axhline(
        y=bonferroni_threshold,
        color=threshold_line_color,
        linestyle=':',
        linewidth=line_width,
    )

    if title:
        plt.title(title, fontsize=_fs.get('title', 20))
    plt.xlabel(f'Expected {_LOG10_P_LABEL}', fontsize=_fs.get('xlabel', 15))
    plt.ylabel(f'Observed {_LOG10_P_LABEL}', fontsize=_fs.get('ylabel', 15))
    style_association_axes(y_floor=0, x_floor=0)

    plt.tight_layout()
    if save:
        skw = default_savefig_kwargs(output_filename)
        plt.savefig(output_filename, **skw)
    if output_filename is None:
        plt.show()
