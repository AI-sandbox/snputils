import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional
from adjustText import adjust_text


def scatter(
    dimredobj: np.ndarray,
    labels_file: str,
    abbreviation_inside_dots: bool = True,
    arrows_for_titles: bool = False,
    dots: bool = True,
    legend: bool = True,
    color_palette=None,
    show: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a scatter plot with centroids for each group, with options for labeling and display styles.
    
    Args:
        dimredobj (np.ndarray): 
            Reduced dimensionality data; expected to have `(n_haplotypes, 2)` shape.
        labels_file (str): 
            Path to a TSV file with columns 'indID' and 'label', providing labels for coloring and annotating points.
        abbreviation_inside_dots (bool): 
            If True, displays abbreviated labels (first 3 characters) inside the centroid markers.
        arrows_for_titles (bool): 
            If True, adds arrows pointing to centroids with group labels displayed near the centroids.
        legend (bool): 
            If True, includes a legend indicating each group label.
        color_palette (optional): 
            Color map or list of colors to use for unique labels. Defaults to 'tab10' if None.
        show (bool, optional):
            Whether to display the plot. Defaults to False.
        save_path (str, optional):
            Path to save the plot image. If None, the plot is not saved.
        
    Returns:
        None
    """
    # Load labels from TSV
    labels_df = pd.read_csv(labels_file, sep='\t')

    # Ensure 'indID' is treated as a string
    labels_df['indID'] = labels_df['indID'].astype(str)

    # Filter labels based on the indIDs in dimredobj
    sample_ids = dimredobj.samples_
    filtered_labels_df = labels_df[labels_df['indID'].isin(sample_ids)]

    # Define unique colors for each group label, either from color_palette or defaulting to 'tab10'
    unique_labels = filtered_labels_df['label'].unique()
    colors = color_palette if color_palette else cm.get_cmap('tab10', len(unique_labels))

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate the overall center of the plot (used for positioning arrows)
    plot_center = dimredobj.X_new_.mean(axis=0)

    # Dictionary to hold centroid positions for each label
    centroids = {}

    # Plot data points and centroids by label
    for i, label in enumerate(unique_labels):
        # Get sample IDs corresponding to the current label
        sample_ids_for_label = filtered_labels_df[filtered_labels_df['label'] == label]['indID']
        
        # Filter points based on sample IDs
        points = dimredobj.X_new_[np.isin(dimredobj.samples_, sample_ids_for_label)]

        if dots:
            # Plot individual points for the current group
            ax.scatter(points[:, 0], points[:, 1], s=30, color=colors(i), alpha=0.6, label=label)
        else:
            # TODO: solve bug
            for point in points:
                print(point[0], point[1])
                ax.text(point[0], point[1], label[:2].upper(), ha='center', va='center', color=colors(i), fontsize=8, weight='bold')

        # Calculate and mark the centroid for the current group
        centroid = points.mean(axis=0)
        centroids[label] = centroid  # Store centroid for later use
        
        # Plot the centroid as a larger dot
        ax.scatter(*centroid, color=colors(i), s=300)

        # Optionally add an abbreviation inside the centroid dot
        if abbreviation_inside_dots:
            ax.text(centroid[0], centroid[1], label[:2].upper(), ha='center', va='center', color='white', fontsize=8, weight='bold')

    # Adding arrows and labels with `adjust_text` for no overlap
    texts = []
    for label, centroid in centroids.items():
        # Determine the direction of the arrow based on centroid position
        offset_x = 0.07 if centroid[0] >= plot_center[0] else -0.07
        offset_y = 0.07 if centroid[1] >= plot_center[1] else -0.07
        
        if arrows_for_titles:
            ax.annotate('', xy=centroid, 
                        xytext=(centroid[0] + offset_x, centroid[1] + offset_y),
                        arrowprops=dict(facecolor=colors(unique_labels.tolist().index(label)), 
                                        shrink=0.05, width=1.5, headwidth=8))
        
            # Label text for centroid
            texts.append(ax.text(centroid[0] + offset_x, centroid[1] + offset_y, label, 
                                color=colors(unique_labels.tolist().index(label)), 
                                fontsize=12, weight='bold'))

        # Adjust text to prevent overlap
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.6))

    # Configure additional plot settings
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if legend:
        ax.legend(loc='upper right')
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Display plot if show is True
    if show:
        plt.show()
    else:
        plt.close()