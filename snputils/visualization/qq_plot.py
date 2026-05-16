import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt

from ._figure_export import default_savefig_kwargs, scatter_rasterized_for_path


def qq_plot(
    data: Union[str, pd.DataFrame],
    color: str = "steelblue",
    significance_threshold: float = 0.05,
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
            Color for the scatter points.  Defaults to ``"steelblue"``.
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

    _fs = fontsize or {}

    p_values = df['P'].dropna().values
    n = len(p_values)

    observed = np.sort(-np.log10(p_values))[::-1]
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    bonferroni_threshold = -np.log10(significance_threshold / n)

    _rz = scatter_rasterized_for_path(output_filename) if output_filename else False

    plt.figure(figsize=figsize)
    plt.scatter(expected, observed, color=color, s=8, rasterized=_rz, label='p-values')

    # Identity reference line (expected under null)
    max_val = max(expected.max(), observed.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=1, label='Expected')

    # Bonferroni threshold
    plt.axhline(y=bonferroni_threshold, color='orange', linestyle=':', linewidth=1, label='Bonferroni')

    if title:
        plt.title(title, fontsize=_fs.get('title', 20))
    plt.xlabel('Expected -log\u2081\u2080(p)', fontsize=_fs.get('xlabel', 15))
    plt.ylabel('Observed -log\u2081\u2080(p)', fontsize=_fs.get('ylabel', 15))
    plt.legend(fontsize=_fs.get('legend', 15))

    plt.tight_layout()
    if save:
        skw = default_savefig_kwargs(output_filename)
        plt.savefig(output_filename, **skw)
    plt.show()
