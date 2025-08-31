import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def reorder_admixture(Q_mat):
    """
    Reorder Q_mat rows so that rows are grouped by each sample's dominant ancestry,
    and columns are sorted by descending average ancestry proportion.
    """
    n_samples, K = Q_mat.shape
    
    # Reorder columns by descending average proportion
    col_means = Q_mat.mean(axis=0)
    col_order = np.argsort(col_means)[::-1]   # largest first
    Q_cols_sorted = Q_mat[:, col_order]
    
    # Group samples by whichever column is argmax
    row_groups = []
    boundary_list = [0]
    argmax_all = np.argmax(Q_cols_sorted, axis=1)
    for k in range(K):
        rows_k = np.where(argmax_all == k)[0]
        # Sort these rows by their proportion in col k
        rows_k_sorted = rows_k[np.argsort(Q_cols_sorted[rows_k, k])[::-1]]
        row_groups.append(rows_k_sorted)
        boundary_list.append(boundary_list[-1] + len(rows_k_sorted))
    
    # Combine them into one final row order
    row_order = np.concatenate(row_groups) if row_groups else np.arange(n_samples)
    Q_mat_sorted = Q_cols_sorted[row_order, :]
    
    return Q_mat_sorted, row_order, boundary_list, col_order

def plot_admixture(ax, Q_mat_sorted, boundary_list, col_order=None, colors=None, show_boundaries=True, show_axes_labels=True, show_ticks=True, set_limits=True, minimal=False):
    """
    Plot a structure-style bar chart of Q_mat_sorted in the given Axes ax.
    If colors is not None, it should be a list or array of length K.
    If col_order is not None, colors are reordered according to col_order.

    Optional controls:
    - show_boundaries (bool): draw vertical lines at group boundaries. Default True.
    - show_axes_labels (bool): set X/Y axis labels. Default True.
    - show_ticks (bool): show axis ticks. Default True.
    - set_limits (bool): set xlim and ylim to [0, n_samples-1] and [0,1]. Default True.
    - minimal (bool): if True, overrides to disable boundaries, labels, ticks, limits and hides spines.
    """
    n_samples, K = Q_mat_sorted.shape

    # Minimal overrides
    if minimal:
        show_boundaries = False
        show_axes_labels = False
        show_ticks = False
        set_limits = False

    # If we have a specific color list and a col_order, reorder the colors to match the columns
    if (colors is not None) and (col_order is not None):
        colors = [colors[idx] for idx in col_order]

    # cumulative sum across columns for stacked fill
    Q_cum = np.cumsum(Q_mat_sorted, axis=1)
    # Use step='post' with padded x/y so the last bar renders fully and no thin band appears
    x_edges = np.arange(n_samples + 1)
    Q_pad = np.vstack([Q_cum, Q_cum[-1]])

    # fill-between for a stacked bar effect
    for j in range(K):
        c = colors[j] if (colors is not None) else None
        lower = Q_pad[:, j - 1] if j > 0 else np.zeros(n_samples + 1)
        upper = Q_pad[:, j]
        ax.fill_between(
            x_edges,
            lower,
            upper,
            step="post",
            color=c,
            linewidth=0,
            edgecolor='none',
        )

    # Vertical lines for group boundaries
    if show_boundaries:
        for boundary in boundary_list:
            ax.axvline(boundary, color='black', ls='--', lw=1.0)

    if set_limits:
        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)

    if show_axes_labels:
        ax.set_xlabel("Samples")
        ax.set_ylabel("Ancestry Proportion")

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if minimal:
        # Hide all spines for a clean, minimal appearance
        for spine in ax.spines.values():
            spine.set_visible(False)
