import numpy as np
from typing import Any, Optional, Tuple, Dict, List, cast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.visualization.constants import CHROM_SIZES


def _custom_cmap(colors: Dict, padding: float = 1.05):
    """
    Create a custom colormap from a dictionary.

    Args:
        colors: Dictionary with levels as keys and color names as values.
        padding: Offset applied to levels. Defaults to 1.05.

    Returns:
        cmap: The custom colormap.
    """
    labels = sorted(float(key) for key in colors if key is not None)
    clrs = [colors[key] for key in labels]
    
    # Adjust levels to match the number of colors minus one
    levels = sorted([labels[0] - 1] + [x * padding for x in labels])

    cmap, _ = mcolors.from_levels_and_colors(levels, clrs)
    return cmap


def _normalize_chromosome_label(label: str) -> str:
    normalized = str(label)
    if normalized.lower().startswith("chr"):
        normalized = normalized[3:]
    return normalized


def _infer_chromosome_size_build(
    chromosome_labels: np.ndarray,
    physical_pos: Optional[np.ndarray],
) -> str:
    if physical_pos is None or len(physical_pos) != len(chromosome_labels):
        return "hg38"

    observed_max = {}
    for chrom, stop in zip(chromosome_labels, physical_pos[:, 1]):
        chrom_str = _normalize_chromosome_label(str(chrom))
        observed_max[chrom_str] = max(observed_max.get(chrom_str, 0), int(stop))

    best_build = "hg38"
    best_score = float("inf")
    for build, chrom_sizes in CHROM_SIZES.items():
        shared = [chrom for chrom in observed_max if chrom in chrom_sizes]
        if not shared:
            continue
        rel_errors = [
            abs(observed_max[chrom] - chrom_sizes[chrom]) / chrom_sizes[chrom]
            for chrom in shared
            if chrom_sizes[chrom] > 0
        ]
        if not rel_errors:
            continue
        score = float(np.mean(rel_errors))
        if score < best_score:
            best_score = score
            best_build = build

    return best_build


def plot_lai(
    laiobj: LocalAncestryObject, 
    colors: Dict,
    sort: Optional[bool]=True,
    figsize: Optional[Tuple[float, float]]=None,
    legend: Optional[bool]=False,
    legend_kwargs: Optional[Dict[str, Any]]=None,
    title: Optional[str]=None,
    fontsize: Optional[Dict[str, float]]=None,
    scale: int=2,
):
    """
    Plot LAI (Local Ancestry Inference) data with customizable options. Each row 
    represents the ancestry of a sample at the window level, distinguishing 
    between maternal and paternal strands. Whitespace is used to separate 
    individual samples.

    Args:
        laiobj: A LocalAncestryObject containing LAI data.
        colors: A dictionary with ancestry-color mapping.
        sort: If True, sort samples based on the most frequent ancestry. 
            Samples are displayed with the most predominant ancestry first, followed by the 
            second most predominant, and so on. Defaults to True.
        figsize: Figure size. If is None, the figure is displayed with a default size of 
            (25, 25). Defaults to None.
        legend: If True, display a legend. If ``sort==True``, ancestries in the 
            legend are sorted based on their total frequency in descending order. Defaults to False.
        legend_kwargs: Optional keyword arguments passed through to ``Axes.legend``.
            Defaults keep the legend centered below the x-axis label.
        title: Title for the plot. If None, no title is displayed. Defaults to None.
        fontsize: Font sizes for various plot elements. If None, default font sizes are used. 
            Defaults to None.
        scale: Number of times to duplicate rows for enhanced vertical visibility. Defaults to 2.
    """
    # The `lai` field is a 2D array containing the window-wise ancestry 
    # information for each individual. Consecutive rows of the transformed form 
    # correspond to the maternal and paternal ancestries of the same individual
    lai_T = laiobj.lai.T
    
    # Obtain number of samples and windows
    n_samples = int(lai_T.shape[0]/2)
    n_windows = lai_T.shape[1]
    sample_ids = laiobj.samples if laiobj.samples is not None else None
    
    if fontsize is None:
        fontsize = {
            'xticks' : 20, 
            'yticks' : 9, 
            'xlabel' : 20, 
            'ylabel' : 20,
            'legend' : 20
        }
    
    if sort:
        # Reshape `lai_T` to a 3D array where the third dimension represents 
        # pairs of consecutive rows, corresponding to maternal and paternal ancestries
        # Dimension: n_samples x 2 x n_windows
        maternal_paternal_pairs = lai_T.reshape(n_samples, 2, n_windows)
        
        # Reshape `maternal_paternal_pairs` to a 2D array where each row contains 
        # concatenated maternal and paternal ancestries
        # Dimension: n_samples x (2·n_windows)
        if maternal_paternal_pairs.ndim != 3:
            raise ValueError("maternal_paternal_pairs must be a 3D array, got array with ndim=" + str(maternal_paternal_pairs.ndim))
        num_samples, num_maternal_paternal, num_windows = cast(Tuple[int, int, int], maternal_paternal_pairs.shape)
        
        flat_ancestry_pairs = maternal_paternal_pairs.reshape(
            num_samples, num_maternal_paternal * num_windows
        )

        # Determine the most frequent ancestry for each sample
        most_freq_ancestry_sample = np.apply_along_axis(
            lambda row: np.bincount(row).argmax(), axis=1, arr=flat_ancestry_pairs
        )
        
        # Obtain unique ancestry values and total counts
        ancestry_values, ancestry_counts = np.unique(lai_T, return_counts=True)
        # Sort ancestry values by total counts in decreasing order
        sorted_ancestry_values = ancestry_values[np.argsort(ancestry_counts)[::-1]]        
        
        # For each ancestry, obtain the samples where that ancestry is predominant
        # and store the indexes sorted in decreasing order based on the ancestry count
        all_sorted_row_idxs = []
        for ancestry in sorted_ancestry_values:
            ancestry_filter = np.where(most_freq_ancestry_sample == ancestry)[0]
            ancestry_counts_sample = np.sum(
                flat_ancestry_pairs[ancestry_filter, :]==ancestry, axis=1
            )
            sorted_row_idxs_1 = np.argsort(ancestry_counts_sample)[::-1]
            sorted_row_idxs = ancestry_filter[sorted_row_idxs_1]
            all_sorted_row_idxs += list(sorted_row_idxs)
        
        # Sort sample IDs based on most frequent ancestry
        if laiobj.samples is not None:
            sample_ids = [laiobj.samples[idx] for idx in all_sorted_row_idxs]
        else:
            sample_ids = None
        
        # Sort `lai_T` based on most frequent ancestry
        num_samples, num_maternal_paternal, num_windows = cast(Tuple[int, int, int], maternal_paternal_pairs.shape)
        lai_T = maternal_paternal_pairs[all_sorted_row_idxs, :].reshape(
            -1, num_windows
        )
    
    # Check if ancestry_map keys are strings of integers (reversed form)
    if all(isinstance(key, str) and key.isdigit() for key in laiobj.ancestry_map.keys()):
        # Reverse the dictionary to match integer-to-ancestry format
        ancestry_map_reverse = {int(key): value for key, value in laiobj.ancestry_map.items()}
    else:
        # The ancestry_map is already in the correct integer-to-ancestry format
        ancestry_map_reverse = laiobj.ancestry_map
    
    # Dictionary with integer-to-color mapping
    colors_map = {key : colors[value] for key, value in ancestry_map_reverse.items()}
    
    # Total number of ancestries
    n_ancestries = len(ancestry_map_reverse.keys())
    
    # Insert whitespace between samples
    lai_T_with_whitespace = np.insert(lai_T, np.arange(2, n_samples*2, 2), n_ancestries, axis=0)
    
    # Repeat rows to increase height of samples in plot
    lai_T_repeat = np.repeat(lai_T_with_whitespace, scale, axis=0)
    
    colors_map[n_ancestries] = 'white'
    
    # Configure custom map from matrix values to colors
    cmap = _custom_cmap(colors_map)

    plot_matrix = lai_T_repeat
    chromosome_tick_positions: Optional[List[float]] = None
    chromosome_tick_labels: Optional[List[str]] = None
    chromosome_boundaries: List[float] = []
    use_chromosome_axis = False

    # If chromosome metadata is present, visually separate chromosome blocks
    # along the x-axis instead of displaying one concatenated strip.
    x_edges: Optional[np.ndarray] = None
    if laiobj.chromosomes is not None and len(laiobj.chromosomes) == n_windows:
        chromosome_labels = np.asarray(
            [_normalize_chromosome_label(ch) for ch in np.asarray(laiobj.chromosomes, dtype=str)],
            dtype=object,
        )
        boundaries = np.where(chromosome_labels[1:] != chromosome_labels[:-1])[0] + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [n_windows]))
        labels = [chromosome_labels[start] for start in starts]

        inferred_build = _infer_chromosome_size_build(chromosome_labels, laiobj.physical_pos)
        chrom_sizes = CHROM_SIZES[inferred_build]

        chromosome_tick_positions = []
        chromosome_tick_labels = []
        x_edge_values: List[float] = [0.0]
        cursor = 0.0
        for idx, (start, stop, label) in enumerate(zip(starts, stops, labels)):
            width = int(stop - start)
            chrom_key = str(label)
            chrom_len = float(chrom_sizes.get(chrom_key, width))
            per_window_width = chrom_len / max(width, 1)
            block_start = cursor
            for _ in range(width):
                cursor += per_window_width
                x_edge_values.append(cursor)
            chromosome_tick_positions.append((block_start + cursor) / 2.0)
            label_text = str(label)
            chromosome_tick_labels.append(
                label_text if label_text.lower().startswith("chr") else f"chr{label_text}"
            )
            if idx < len(starts) - 1:
                chromosome_boundaries.append(cursor)

        x_edges = np.asarray(x_edge_values, dtype=float)
        use_chromosome_axis = True
    
    # Plot LAI output
    if figsize is None:
        plt.figure(figsize=(25, 25))
    else:
        plt.figure(figsize=figsize)
    if use_chromosome_axis and x_edges is not None:
        y_edges = np.arange(plot_matrix.shape[0] + 1, dtype=float)
        ax = plt.gca()
        ax.pcolormesh(
            x_edges,
            y_edges,
            plot_matrix,
            cmap=cmap,
            shading="flat",
        )
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.invert_yaxis()
    else:
        plt.imshow(plot_matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    
    # Display sample IDs in y-axis
    yticks_positions = np.arange(scale, lai_T_repeat.shape[0]+1, scale*(2+1))
    plt.yticks(yticks_positions, sample_ids, fontsize=fontsize['yticks'])
    
    ax = plt.gca()
    if use_chromosome_axis and chromosome_tick_positions is not None and chromosome_tick_labels is not None:
        xtick_fontsize = min(fontsize['xticks'], 12) if len(chromosome_tick_labels) > 12 else fontsize['xticks']
        ax.set_xticks(chromosome_tick_positions)
        ax.set_xticklabels(chromosome_tick_labels, fontsize=xtick_fontsize)
        ax.set_xlabel('Chromosome', fontsize=fontsize['xlabel'], labelpad=8)
        for boundary in chromosome_boundaries:
            ax.axvline(boundary, color="#D0D0D0", linewidth=1.2, zorder=3)
    else:
        ax.tick_params(axis='x', labelsize=fontsize['xticks'])
        ax.set_xlabel('Window', fontsize=fontsize['xlabel'], labelpad=8)

    ax.set_ylabel('Sample', fontsize=fontsize['ylabel'])
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', length=8, width=2)
    ax.tick_params(axis='x', which='major', pad=4)
    ax.tick_params(axis='y', which='major', pad=4)
    
    if legend:
        if sort:
            # Sort legend based on global ancestry frequency in decresaing order 
            sorted_colors = [colors_map[x] for x in sorted_ancestry_values]
            ancestries = [ancestry_map_reverse[x] for x in sorted_ancestry_values]
            legend_patches = [patches.Patch(color=color, label=label) 
                              for color, label in zip(sorted_colors, ancestries)]
        else:
            # Add patches for each color to represent the legend squares
            legend_patches = [patches.Patch(color=color, label=label) 
                              for color, label in zip(colors.values(), colors.keys())]

        resolved_legend_kwargs = {
            'loc': 'lower center',
            'bbox_to_anchor': (0.5, -0.25),
            'borderaxespad': 0.0,
            'ncol': n_ancestries,
            'fontsize': fontsize['legend'],
        }
        if legend_kwargs is not None:
            resolved_legend_kwargs.update(legend_kwargs)

        ax.legend(handles=legend_patches, **resolved_legend_kwargs)
    
   
    if title:
        plt.title(title)
