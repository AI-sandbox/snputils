import logging
import os
import pathlib
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from snputils.visualization.constants import snputils_palette, CHROM_SIZES

log = logging.getLogger(__name__)


def load_color_map(color_map_path: str) -> Dict[str, str]:
    """
    Loads color mappings from a TSV file into a dictionary.

    Args:
        color_map_path: Path to TSV file with 'CODE', 'NAME', and 'COLOR' columns

    Returns:
        Dictionary mapping ancestry codes/names to colors
    """
    if color_map_path and os.path.exists(color_map_path):
        color_map_df = pd.read_csv(color_map_path, sep='\t')
        color_map = {}
        for _, row in color_map_df.iterrows():
            color_map[row['CODE']] = row['COLOR']
            color_map[row['NAME']] = row['COLOR']
        return color_map
    return {}


def generate_hex_colors(n: int) -> Dict[int, str]:
    """
    Generate HEX colors for integers from 0 to n using the snputils_palette by default,
    falling back to the plasma colormap if more colors are needed.
    
    Args:
        n: The maximum integer value
    
    Returns:
        Dictionary mapping integers to HEX colors
    """
    # First try to use the predefined palette
    if n <= len(snputils_palette):
        return {i: color for i, color in enumerate(snputils_palette[:n + 1])}
    
    # If we need more colors than in the palette, fall back to plasma colormap
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 1, n + 1))
    color_dict = {i: mcolors.to_hex((r, g, b), keep_alpha=False) 
                  for i, (r, g, b, _) in enumerate(colors)}
    
    return color_dict


def get_bed_data(msp_df: pd.DataFrame, sample: str, pop_order: Optional[Dict] = None) -> Dict:
    """
    From: https://github.com/AI-sandbox/gnomix/blob/main/src/postprocess.py
    Transforms MSP DataFrame data into a BED-like format for a specific sample.

    Args:
        msp_df: DataFrame containing MSP file data
        sample: The specific sample column to process
        pop_order: Optional mapping from population numeric to desired labels

    Returns:
        Dictionary with BED-format data for the sample
    """
    ancestry_label = lambda pop_numeric: pop_numeric if pop_order is None else pop_order[pop_numeric]

    chm, spos, sgpos = [[val] for val in msp_df[["#chm", "spos", "sgpos"]].iloc[0]]
    epos, egpos = [], []
    anc = msp_df[sample].iloc[0]
    ancestry_labels = [ancestry_label(anc)]

    for i, row_anc in enumerate(msp_df[sample].iloc[1:]):
        row = i + 1
        if row_anc != anc:
            anc = row_anc
            epos.append(msp_df["epos"].iloc[row - 1])
            egpos.append(msp_df["egpos"].iloc[row - 1])
            chm.append(msp_df["#chm"].iloc[row])
            ancestry_labels.append(ancestry_label(row_anc))
            spos.append(msp_df["spos"].iloc[row])
            sgpos.append(msp_df["sgpos"].iloc[row])

    epos.append(msp_df["epos"].iloc[-1])
    egpos.append(msp_df["egpos"].iloc[-1])

    bed_data = {
        "#chr": np.array(chm).astype(int),
        "start": np.array(spos).astype(int),
        "stop": np.array(epos).astype(int),
        "feature": np.zeros(np.array(chm).shape[0], dtype=int),
        "size": np.ones(np.array(chm).shape[0], dtype=int),
        "ancestry": ancestry_labels
    }
    return bed_data




def msp_to_bed(msp_file, root, pop_order=None):
    """
    From: https://github.com/AI-sandbox/gnomix/blob/main/src/postprocess.py

    Args:
        msp_file
        root
        pop_order

    Returns:
        None
    """
    with open(msp_file) as f:
        _ = f.readline()
        second_line = f.readline()

    header = second_line.split("\t")
    msp_df = pd.read_csv(msp_file, sep="\t", comment="#", names=header)

    samples = header[6:]

    for sample in samples:
        sample_file_name = os.path.join(root, sample.replace(".", "_") + ".bed")
        sample_bed_data = get_bed_data(msp_df, sample, pop_order=pop_order)
        sample_bed_df = pd.DataFrame(sample_bed_data)
        sample_bed_df.to_csv(sample_file_name, sep="\t", index=False)



def fill_missing_segments(bed_df, build):
    # Create a list to collect new segments for both chrCopy 1 and 2
    new_segments = []

    # Iterate through each chromosome in CHROM_SIZES
    for chrom, size in CHROM_SIZES[build].items():
        # For each chromosome, we need to do this for both chrCopy 1 and 2
        for chr_copy in [1, 2]:
            # Filter the dataframe for the current chromosome and chrCopy
            chrom_df = bed_df[(bed_df['#chr'].astype(str) == str(chrom)) & (bed_df['chrCopy'] == chr_copy)]

            # If there are entries for this chromosome and chrCopy
            if not chrom_df.empty:
                # Get the start and end positions of the segments
                first_start_pos = chrom_df['start'].min()
                last_end_pos = chrom_df['stop'].max()

                # If the first segment does not start at 0, add a new segment at the beginning
                if first_start_pos > 0:
                    new_segments.append({
                        '#chr': chrom,
                        'start': 0,
                        'stop': first_start_pos - 1,
                        'feature': 0,
                        'size': 1,
                        'color': '#8B8982',
                        'chrCopy': chr_copy
                    })

                # If the last segment does not reach the end position, add a new segment at the end
                if last_end_pos < size:
                    new_segments.append({
                        '#chr': chrom,
                        'start': last_end_pos + 1,
                        'stop': size,
                        'feature': 0,
                        'size': 1,
                        'color': '#8B8982',
                        'chrCopy': chr_copy
                    })

    # Create a DataFrame from the new segments
    new_segments_df = pd.DataFrame(new_segments)

    # Combine the original DataFrame with the new segments
    filled_bed_df = pd.concat([bed_df, new_segments_df], ignore_index=True)

    # Sort by chromosome, chrCopy, and start position
    filled_bed_df.sort_values(by=['#chr', 'chrCopy', 'start'], inplace=True)

    # Return the new dataframe with missing segments filled
    return filled_bed_df


def sanitize_name(text: str) -> str:
    """Convert text to a safe format by removing special characters and spaces.
    
    Args:
        text: Input string to sanitize
    
    Returns:
        Sanitized string with only alphanumeric chars and underscores
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Keep only alphanumeric chars and underscores
    text = ''.join(c for c in text if c.isalnum() or c == '_')
    return text


def msp_files_to_bed(
        msp_files: List[str],
        root: str,
        sample_from: int = 0,
        max_sample_count: int = 100,
        num_labels: int = 8,
        build: str = 'hg37',
        color_map: Optional[str] = None,
        fill_empty: bool = False,
) -> List[str]:
    """
    Converts a list of 'msp' files to 'bed' files for genomic data analysis.

    Parameters:
    - msp_files (List[str]): A list of paths to 'msp' files.
    - root (str): The directory where the resulting 'bed' files will be saved.
    - sample_from (int, default 0): Starting index for samples to be processed.
    - max_sample_count (int, default -1): Number of max samples to process.
    - num_labels (int, default 8): The number of labels/colors to use in the color mapping.
    - build (str, default 'hg37'): The build version to be used in processing.
    - color_map (Optional[str], default None): Path to a color map file, if a specific color scheme is desired.
    - fill_empty (bool, default False): Whether to fill empty genomic segments in the 'bed' file.

    Returns:
    - List[str]: A list of paths to the generated '
    bed' files.

    """

    all_files = []

    # Read the first msp_file to get the sample names (assuming all msp_files have the same samples)
    with open(msp_files[0]) as f:
        _ = f.readline()
        second_line = f.readline()
    header = second_line.split("\t")
    samples = header[6:]
    paired_samples = [samples[i:i + 2] for i in range(0, len(samples), 2)]

    # Check if conditions to process all samples are met
    if max_sample_count > 0:
        sample_count = min(max_sample_count, len(paired_samples))
    else:
        sample_count = len(paired_samples)

    for sample in paired_samples[sample_from:sample_from + sample_count]:
        aggregated_dfs = []  # To store data from all msp_files for the current sample
        for msp_file in msp_files:
            msp_df = pd.read_csv(msp_file, sep="\t", comment="#", names=header)

            # Extract the original_mapping from the msp file header
            with open(msp_file, "r") as msp_file_r:
                header_line = msp_file_r.readline().strip().replace("#Subpopulation order/codes: ", "")
                original_mapping = {label.split('=')[0]: int(label.split('=')[1]) for label in header_line.split("\t")}

            # unique_labels = np.unique(msp_df.iloc[:,6:].values.flatten())
            # color_dict = generate_hex_colors(len(unique_labels))
            if isinstance(color_map, str):
                _color_dict = load_color_map(color_map)
                color_dict = {original_mapping[k]: _color_dict[k] for k in original_mapping.keys()}
            elif isinstance(color_map, dict):
                color_dict = color_map
            else:
                color_dict = generate_hex_colors(num_labels)
            dfs = []
            for haploid in sample:
                parts = haploid.split('.')
                last_value = parts[-1].strip()
                sample_bed_data = get_bed_data(msp_df, haploid, pop_order=None)
                sample_bed_df = pd.DataFrame(sample_bed_data)
                sample_bed_df['color'] = sample_bed_df['ancestry'].replace(color_dict)
                sample_bed_df['chrCopy'] = int(last_value) + 1
                sample_bed_df.drop(columns=['ancestry'], inplace=True)
                dfs.append(sample_bed_df)
            sample_df_from_current_msp = pd.concat(dfs, ignore_index=True)
            aggregated_dfs.append(sample_df_from_current_msp)

        # Combine the data from all msp_files for the current sample
        final_sample_df = pd.concat(aggregated_dfs, ignore_index=True)

        if fill_empty:
            final_sample_df = fill_missing_segments(final_sample_df, build)

        # Save the aggregated data with the sample name as the filename
        # sample_name = sample[0].split('.')[0]  # Extracting the sample name from the haploid name
        
        # Get sample name (removing .0 or .1 suffix)
        sample_name = sample[0].rsplit('.', 1)[0]
        sample_name = sanitize_name(sample_name)
        sample_bed_file = os.path.join(root, f"{sample_name}.bed")
        final_sample_df.to_csv(sample_bed_file, sep="\t", index=False)
        all_files.append(sample_bed_file)

    # Return the paths to all the generated .bed files
    return all_files
