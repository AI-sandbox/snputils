import numpy as np
from typing import Optional, Tuple, Dict, cast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from snputils.ancestry.genobj.window import WindowLevelAncestryObject


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


def plot_lai(
    laiobj: WindowLevelAncestryObject, 
    colors: Dict,
    sort: Optional[bool]=True,
    figsize: Optional[Tuple[float, float]]=None,
    legend: Optional[bool]=False,
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
        laiobj: A WindowLevelAncestryObject containing LAI data.
        colors: A dictionary with ancestry-color mapping.
        sort: If True, sort samples based on the most frequent ancestry. 
            Samples are displayed with the most predominant ancestry first, followed by the 
            second most predominant, and so on. Defaults to True.
        figsize: Figure size. If is None, the figure is displayed with a default size of 
            (25, 25). Defaults to None.
        legend: If True, display a legend. If ``sort==True``, ancestries in the 
            legend are sorted based on their total frequency in descending order. Defaults to False.
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
    
    # Plot LAI output
    if figsize is None:
        plt.figure(figsize=(25, 25))
    else:
        plt.figure(figsize=figsize)
    plt.imshow(lai_T_repeat, cmap=cmap)
    
    # Display sample IDs in y-axis
    yticks_positions = np.arange(scale, lai_T_repeat.shape[0]+1, scale*(2+1))
    plt.xticks(fontsize=fontsize['xticks'])
    plt.yticks(yticks_positions, sample_ids, fontsize=fontsize['yticks'])
    
    ax = plt.gca()
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
    
        ax.legend(
            handles=legend_patches, 
            loc='lower center', 
            borderaxespad=-5,
            ncol=n_ancestries,
            fontsize=fontsize['legend']
        )
    
    if title:
        plt.title(title)
