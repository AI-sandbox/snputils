# Library Importation
import allel
import numpy as np
import pandas as pd
import sys
import time
import warnings
import logging
import datetime
from os import path
warnings.simplefilter("ignore", category=RuntimeWarning)


def process_vit(vit_file):
    """                                                                                       
    Process the Viterbi file to extract ancestry information.

    This function reads a Viterbi file containing ancestry information for individuals 
    at different genomic positions. It processes the file to construct an ancestry 
    matrix, where each row represents a genomic position and each column corresponds 
    to an individual.

    Args:
        vit_file (str): 
            Path to the Viterbi file. The file should be tab-separated, where the first 
            column represents genomic positions, and the remaining columns contain 
            ancestry assignments for individuals.

    Returns:
        np.ndarray of shape (n_snps, n_samples): 
            An ancestry matrix where `n_snps` represents the number of genomic positions 
            and `n_samples` represents the number of individuals.
    """
    start_time = time.time()
    vit_matrix = []
    with open(vit_file) as file:
        for x in file:
            x_split = x.replace('\n', '').split('\t')
            vit_matrix.append(np.array(x_split[1:-1]))
    ancestry_matrix = np.stack(vit_matrix, axis=0).T
    logging.info("VIT Processing Time: --- %s seconds ---" % (time.time() - start_time))
    
    return ancestry_matrix


def process_fbk(fbk_file, n_ancestries, prob_thresh):    
    """                                                                                       
    Process the FBK file to extract ancestry information.

    This function reads an FBK file containing ancestry probability values for 
    individuals across genomic positions. It processes these probabilities and 
    assigns ancestries based on a specified probability threshold.

    Args:
        fbk_file (str): 
            Path to the FBK file. The file should be space-separated, where each 
            row represents a genomic position, and columns contain probability values.
        n_ancestries (int): 
            Number of distinct ancestries in the dataset.
        prob_thresh (float): 
            Probability threshold for assigning ancestry to an individual at a 
            specific position.

    Returns:
        np.ndarray of shape (n_snps, n_samples): 
            An ancestry matrix where `n_snps` represents the number of genomic 
            positions and `n_samples` represents the number of individuals.
    """
    start_time = time.time()
    # Load FBK file into a DataFrame
    df_fbk = pd.read_csv(fbk_file, sep=" ", header=None)
    
    # Extract ancestry probability values, excluding the last column
    fbk_matrix = df_fbk.values[:, :-1]
    
    # Initialize ancestry matrix with zeros
    ancestry_matrix = np.zeros((fbk_matrix.shape[0], int(fbk_matrix.shape[1] / n_ancestries)), dtype=np.int8)
    
    # Assign ancestry based on probability threshold
    for i in range(n_ancestries):
        ancestry = i+1
        ancestry_matrix += (fbk_matrix[:, i::n_ancestries] > prob_thresh) * 1 * ancestry
    
    # Convert ancestry values to string format
    ancestry_matrix = ancestry_matrix.astype(str)
    logging.info("FBK Processing Time: --- %s seconds ---" % (time.time() - start_time))
    
    return ancestry_matrix


def process_tsv_fb(tsv_file, n_ancestries, prob_thresh, variants_pos, calldata_gt, variants_id):
    """                                                                                       
    Process the TSV/FB file to extract ancestry information.

    This function reads a TSV file containing ancestry probabilities, aligns 
    positions with the given genome data, and assigns ancestry labels based on 
    a probability threshold.

    Args:
        tsv_file (str): 
            Path to the TSV file. The file should be tab-separated and must contain 
            a 'physical_position' column along with ancestry probability values.
        n_ancestries (int): 
            Number of distinct ancestries in the dataset.
        prob_thresh (float): 
            Probability threshold for assigning ancestry to an individual at a 
            specific position.
        variants_pos (list of int): 
            A list containing the chromosomal positions for each SNP.
        calldata_gt (np.ndarray of shape (n_snps, n_samples)): 
            An array containing genotype data for each sample.
        variants_id (list of str): 
            A list containing unique identifiers (IDs) for each SNP.

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_samples): 
                An ancestry matrix indicating the ancestry assignment for each 
                individual at each genomic position.
            - np.ndarray of shape (n_snps, n_samples): 
                The updated genotype matrix after aligning with TSV positions.
            - list of str: 
                The updated list of SNP identifiers. 
    """
    start_time = time.time()
    # Load TSV file, skipping the first row
    df_tsv = pd.read_csv(tsv_file, sep="\t", skiprows=1)

    # Extract physical positions and remove unnecessary columns
    tsv_positions = df_tsv['physical_position'].tolist()
    df_tsv.drop(columns = ['physical_position', 'chromosome', 'genetic_position', 'genetic_marker_index'], inplace=True)

    # Convert DataFrame to NumPy array
    tsv_matrix = df_tsv.values

    # Find the range of positions that match between TSV and provided positions
    i_start = variants_pos.index(tsv_positions[0])
    if tsv_positions[-1] in variants_pos:
        i_end = variants_pos.index(tsv_positions[-1]) + 1
    else:
        i_end = len(variants_pos)

    # Update genome data to match the TSV file range
    calldata_gt = calldata_gt[i_start:i_end, :]
    variants_pos = variants_pos[i_start:i_end]
    variants_id = variants_id[i_start:i_end]

    # Initialize probability matrix with the same shape as filtered positions
    prob_matrix = np.zeros((len(variants_pos), tsv_matrix.shape[1]), dtype=np.float16)

    # Align TSV probabilities with genomic positions
    i_tsv = -1
    next_pos_tsv = tsv_positions[i_tsv+1]
    for i in range(len(variants_pos)):
        pos = variants_pos[i]
        if pos >= next_pos_tsv and i_tsv + 1 < tsv_matrix.shape[0]:
            i_tsv += 1
            probs = tsv_matrix[i_tsv, :]
            if i_tsv + 1 < tsv_matrix.shape[0]:
                next_pos_tsv = tsv_positions[i_tsv+1]
        prob_matrix[i, :] = probs

    # Replace TSV matrix with aligned probability matrix
    tsv_matrix = prob_matrix

    # Initialize ancestry matrix
    ancestry_matrix = np.zeros((tsv_matrix.shape[0], int(tsv_matrix.shape[1] / n_ancestries)), dtype=np.int8)

    # Assign ancestry based on probability threshold
    for i in range(n_ancestries):
        ancestry = i+1
        ancestry_matrix += (tsv_matrix[:, i::n_ancestries] > prob_thresh) * 1 * ancestry

    # Adjust ancestry values to start at 0
    ancestry_matrix -= 1

    # Convert ancestry matrix to string format
    ancestry_matrix = ancestry_matrix.astype(str)
    logging.info("TSV Processing Time: --- %s seconds ---" % (time.time() - start_time))
    
    return ancestry_matrix, calldata_gt, variants_id


def process_laiobj(laiobj, snpobj):
    """                                                                                       
    Obtain a SNP-level ancestry matrix by matching genomic positions and chromosomes in the SNPObject 
    with ancestry segments in the LocalAncestryObject.

    Args:
        laiobj (LocalAncestryObject): 
            A LocalAncestryObject instance.
        variants_pos (list of int): 
            A list containing the chromosomal positions for each SNP.
        variants_chrom (list of int): 
            A list containing the chromosome for each SNP.
        calldata_gt (np.ndarray of shape (n_snps, n_samples)): 
            An array containing genotype data for each sample.
        variants_id (list of str): 
            A list containing unique identifiers (IDs) for each SNP.

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_samples): 
                An ancestry matrix where `n_snps` represents the number of genomic 
                positions and `n_samples` represents the number of individuals. 
                Ancestry values are assigned based on LAI data.                                                                                          
    """
    start_time = time.time()
    ancestry_matrix = laiobj.convert_to_snp_level(snpobj, lai_format='2D').calldata_lai
    logging.info("TSV Processing Time: --- %s seconds ---" % (time.time() - start_time))
    return ancestry_matrix


def process_beagle(beagle_file, rs_ID_dict, rsid_or_chrompos):
    """                                                                                       
    Process a Beagle file to extract genotype and variant information.

    This function processes a Beagle genotype file to extract individual IDs, 
    reformat variant identifiers, and encode genetic information into a structured 
    genotype matrix.

    Args:
        beagle_file (str): 
            Path to the Beagle file containing genotype data.
        rs_ID_dict (dict): 
            Dictionary mapping variant identifiers to reference alleles.
            If an identifier is not found, it will be added to the dictionary.
        rsid_or_chrompos (int): 
            Specifies the format of variant identifiers:
            - `1`: rsID format (e.g., "rs12345").
            - `2`: Uses Chromosome_Position format (e.g., "1.12345" for chromosome 1, position 12345).

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_samples * ploidy): 
                A genotype matrix where `n_snps` represents the number of genomic 
                positions and `n_samples * ploidy` represents the flattened genotype calls 
                for diploid individuals.
            - np.ndarray of shape (n_samples,): 
                Array of individual IDs corresponding to the genotype matrix.
            - list of int or float: 
                List of variant identifiers, formatted based on `rsid_or_chrompos` selection.
            - dict: 
                Updated dictionary mapping variant identifiers to reference alleles.
    """
    start_time = time.time()
    variants_id = []
    lis_beagle = []
    with open(beagle_file) as file:
        # Read header line and extract individual IDs
        x = file.readline()
        x_split = x.replace('\n', '').split('\t')
        ind_IDs = x_split[2:]
        ind_IDs = np.array(ind_IDs)
        for x in file:
            x_split = x.replace('\n', '').split('\t')
            if rsid_or_chrompos == 1:
                variants_id.append(int(x_split[1][2:]))
            elif rsid_or_chrompos == 2:
                rs_ID_split = x_split[1].split('_')
                variants_id.append(np.float64(rs_ID_split[0] + '.' + rs_ID_split[1][::-1]))
            else:
                sys.exit("Illegal value for rsid_or_chrompos. Choose 1 for rsID format or 2 for Chromosome_position format.")
            lis_beagle.append(x_split[2:])
    
    # Initialize genotype matrix
    calldata_gt = np.zeros((len(lis_beagle),len(lis_beagle[0])), dtype=np.float16)
    
    # Process reference allele encoding
    processed_IDs = rs_ID_dict.keys()
    for i in range(len(lis_beagle)):
        # Check how we usually encode:
        if (variants_id[i] in processed_IDs):
            ref = rs_ID_dict[variants_id[i]]
        else:
            ref = lis_beagle[i][0]
            rs_ID_dict[variants_id[i]] = ref

        for j in range(1, len(lis_beagle[i])):
            calldata_gt[i, j] = (lis_beagle[i][j] != ref)*1

    logging.info("Beagle Processing Time: --- %s seconds ---" % (time.time() - start_time))

    return calldata_gt, ind_IDs, variants_id, rs_ID_dict


def process_snpobj(snpobj, rsid_or_chrompos):
    """                                                                                       
    Process genotype data from a SNPObject:
    - Reshape the 3D genotype array (n_snps, n_samples, 2) to 2D (n_snps, n_samples × 2). 
    - Replace missing values (-1) with NaN.
    - Generate variant identifiers using either the rsID or chromosome_position format, based on a provided flag.
    - Create diploid sample identifiers by appending "_A" and "_B" to each sample.

    This function processes genetic variant data from a SNPObject, restructuring genotype information, 
    formatting variant identifiers, and ensuring consistency in allele encoding.

    Args:
        snpobj (SNPObjects): 
            A SNPObject instance.
        rsid_or_chrompos (int): 
            Specifies the format of variant identifiers:
            - `1`: rsID format (e.g., "rs12345").
            - `2`: Uses Chromosome_Position format (e.g., "1.12345" for chromosome 1, position 12345).

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_haplotypes): 
                A genotype matrix where `n_snps` represents the number of genomic 
                positions and `n_haplotypes` represents the flattened genotype calls 
                for diploid individuals.
            - np.ndarray of shape (n_haplotypes,): 
                Array of individual IDs corresponding to the genotype matrix.
            - list of int or float: 
                List of variant identifiers, formatted based on `rsid_or_chrompos` selection.
    """
    start_time = time.time()

    # Extract genotype data and reshape to 2D (n_snps, n_samples * ploidy)
    calldata_gt = snpobj['calldata_gt']
    n_snps, n_samples, ploidy = calldata_gt.shape
    calldata_gt = calldata_gt.reshape(n_snps, n_samples * ploidy).astype(np.float16)

    # Replace missing genotype values (-1) with NaN
    np.place(calldata_gt, calldata_gt < 0, np.nan)

    # Extract variant identifiers based on rsID format selection
    if rsid_or_chrompos == 1:
        IDs = snpobj['variants_id']
        variants_id = [int(x[2:]) for x in IDs]
    elif rsid_or_chrompos == 2:
        variants_id = []
        for i in range(len(snpobj['variants_chrom'])):
            variants_id.append(np.float64(snpobj['variants_chrom'][i] + '.' + str(snpobj['variants_pos'][i])[::-1]))
    else:
        sys.exit("Illegal value for rsid_or_chrompos. Choose 1 for rsID format or 2 for Chromosome_position format.")

    # Extract individual sample IDs
    samples = snpobj['samples']

    # Generate individual IDs for diploid samples
    ind_IDs = np.array([f"{sample}_{suffix}" for sample in samples for suffix in ["A", "B"]])
    
    logging.info("SNPObject Processing Time: --- %s seconds ---" % (time.time() - start_time))
    return calldata_gt, ind_IDs, variants_id


def average_parent_snps(masked_ancestry_matrix, force_nan_incomplete_strands=False):
    """                                                                                       
    Average haplotypes to obtain genotype data for individuals. 

    This function combines pairs of haplotypes by computing their mean.
    If `force_nan_incomplete_strands=True`, the result is set to NaN if either haplotype in a pair is NaN.
    Otherwise, it computes the mean ignoring NaN values.

    Args:
        masked_ancestry_matrix (np.ndarray of shape (n_snps, n_haplotypes)): 
            The masked matrix for an ancestry, where n_snps represents the number of SNPs and 
            n_haplotypes represents the number of haplotypes.
        force_nan_incomplete_strands (bool): 
            If `True`, sets the result to NaN if either haplotype in a pair is NaN. 
            Otherwise, computes the mean while ignoring NaNs (e.g., 0|NaN -> 0, 1|NaN -> 1).

    Returns:
        np.ndarray of shape (n_snps, n_samples):  
            A new matrix where each pair of haplotypes has been averaged, resulting in genotype 
            data for individuals instead of haplotypes.
    """
    rows, cols = masked_ancestry_matrix.shape

    # Reshape the matrix to group pairs of adjacent columns (each individual has 2 haplotypes)
    reshaped_matrix = masked_ancestry_matrix.reshape(rows, cols // 2, 2)

    if force_nan_incomplete_strands:
        # Identify pairs where at least one strand is NaN
        nan_mask = np.any(np.isnan(reshaped_matrix), axis=2)

        # Compute mean while ignoring NaNs
        avg_matrix = np.nanmean(reshaped_matrix, axis=2, dtype=np.float16)

        # Apply NaN mask: Set to NaN if either haplotype was NaN
        avg_matrix[nan_mask] = np.nan
    else:
        # Compute mean, allowing np.nanmean to handle missing values
        avg_matrix = np.nanmean(reshaped_matrix, axis=2, dtype=np.float16)

    return avg_matrix


def mask_calldata_gt(ancestry_matrix, calldata_gt, ancestry, average_strands=False, force_nan_incomplete_strands=False):
    """
    Mask the genotype matrix by retaining only the entries that match a given ancestry.

    This function processes a genotype matrix by masking SNP-haplotype values based on a 
    specified ancestry. Genotype values corresponding to other ancestries are replaced with NaN. 
    Optionally, the function averages the SNP values across haplotypes for each individual.

    Args:
        ancestry_matrix (np.ndarray of shape (n_snps, n_haplotypes)): 
            Matrix indicating the ancestry assignments, where `n_snps` represents the number of SNPs and 
            `n_haplotypes` represents the number of haplotypes.
        calldata_gt (np.ndarray of shape (n_snps, n_haplotypes)): 
            Genetic matrix encoding the genotype information for haplotypes.
        ancestry (str): 
            Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at `0`.
        average_strands (bool, default=False): 
            Whether to average haplotypes for each individual. Default is `False`.
        force_nan_incomplete_strands (bool): 
            If True, sets the result to NaN if either haplotype in a pair is NaN.
            If False, computes the mean while ignoring NaNs.

    Returns:
        - mask (dict of str to np.ndarray): 
            A dictionary where key represent ancestry identifiers, and values are the corresponding genotype matrices.  
            The matrices contain only genotype data for the specified ancestry, with all other ancestries set to NaN.
    """
    start_time = time.time()

    # Dictionary to store the masked genotype matrix for the target ancestry
    mask = {}

    # Initialize a flat array with NaN values to store masked genotype data
    mask[ancestry] = np.empty(ancestry_matrix.size, dtype=np.float16)
    mask[ancestry][:] = np.nan  # Default to NaN for all positions

    # Identify positions in ancestry_matrix where ancestry matches the specified target
    matching_indices = ancestry_matrix.ravel() == ancestry

    # Retain genotype data only for the matched ancestry positions
    mask[ancestry][matching_indices] = calldata_gt.ravel()[matching_indices]

    # Log the time taken for the masking operation
    logging.info(f"Masking for ancestry {ancestry} --- {time.time() - start_time:.4f} seconds")

    # Reshape the masked array back to its original 2D form
    mask[ancestry] = mask[ancestry].reshape(ancestry_matrix.shape).astype(np.float16)

    # If averaging strands is enabled, compute the average SNP values per individual
    if average_strands:
        start = time.time()
        mask[ancestry] = average_parent_snps(mask[ancestry], force_nan_incomplete_strands)
        logging.info("Combining time --- %s seconds ---" % (time.time() - start))

    return mask


def remove_AB_indIDs(ind_IDs):
    """
    Removes A/B labels from individual (haplotype) identifiers.

    This function assumes that the input list contains alternating identifiers
    (e.g., ['ID1_A', 'ID1_B', 'ID2_A', 'ID2_B', ...]) and extracts only 
    the base identifier (e.g., ['ID1', 'ID2', ...]), effectively halving the list.

    Args:
        ind_IDs (list or array): List of individual identifiers with '_A' and '_B' suffixes.

    Returns:
        np.array: Array of unique individual identifiers without A/B labels.
    """
    new_ind_IDs = []
    for i in range(int(len(ind_IDs)/2)):
        new_ind_IDs.append(ind_IDs[2*i][:-2])
    new_ind_IDs = np.array(new_ind_IDs)
    return new_ind_IDs


def add_AB_indIDs(ind_IDs):
    """
    Expands individual (haplotype) identifiers by adding A/B labels.

    This function duplicates each identifier by appending '_A' and '_B' suffixes,
    assuming each individual has two haplotypes.

    Args:
        ind_IDs (list or array): List of base individual identifiers (e.g., ['ID1', 'ID2', ...]).

    Returns:
        np.array: Array where each identifier is expanded into two haplotypes (e.g., ['ID1_A', 'ID1_B', 'ID2_A', 'ID2_B', ...]).
    """
    new_ind_IDs = []
    for i in range(len(ind_IDs)):
        new_ind_IDs.append(str(ind_IDs[i]) + '_A')
        new_ind_IDs.append(str(ind_IDs[i]) + '_B')
    new_ind_IDs = np.array(new_ind_IDs)
    return new_ind_IDs


def process_calldata_gt(snpobj, laiobj, ancestry, average_strands, force_nan_incomplete_strands, is_masked, rsid_or_chrompos): 
    """                                                                                       
    Process genotype data with optional ancestry-based masking and return the corresponding 
    SNP and individual identifiers.

    This function processes genotype data by restructuring it into a 2D matrix:
    - If `average_strands=False`, the shape is `(n_snps, n_samples × 2)`, preserving haplotype data.
    - If `average_strands=True`, the shape is `(n_snps, n_samples)`, averaging SNP values across haplotypes for each individual.

    If `is_masked=True`, the function filters SNP-haplotype values based on the specified ancestry, 
    replacing all other ancestry values with NaN.

    Args:
        snpobj (SNPObject): 
            A SNPObject instance.
        laiobj (LocalAncestryObject): 
            A LocalAncestryObject instance.
        ancestry (str): 
            Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at `0`.
        average_strands (bool): 
            Whether to average haplotypes for each individual.
        force_nan_incomplete_strands (bool): 
            If `True`, sets the result to NaN if either haplotype in a pair is NaN. 
            Otherwise, computes the mean while ignoring NaNs (e.g., 0|NaN -> 0, 1|NaN -> 1).
        is_masked (bool): 
            If `True`, applies ancestry-specific masking to the genotype matrix, retaining only genotype data 
            corresponding to the specified `ancestry`. If `False`, uses the full, unmasked genotype matrix.
        rsid_or_chrompos (int): 
            Specifies the format of variant identifiers:
            - `1`: rsID format (e.g., "rs12345").
            - `2`: Uses Chromosome_Position format (e.g., "1.12345" for chromosome 1, position 12345).

    Returns:
        tuple:
            - mask (dict of str to np.ndarray): 
                A dictionary where key represents the ancestry identifier, and value is the corresponding genotype matrix.  
                If `is_masked=True`, the matrix contains only genotype data for the specified ancestry, with all other ancestries set to NaN.  
                If `is_masked=False`, the full, unmasked genotype matrix is returned.
            - variants_id (list of str): 
                A list of unique SNP identifiers in the specified format (`rsid_or_chrompos`).  
                The list may be reduced if some SNPs are not present in the `laiobj`.
            - haplotypes (list of str): 
                A list of unique individual sample identifiers.  
                If haplotypes are averaged (`average_strands=True`), duplicate haplotype labels (e.g., "A" and "B") are removed.
    """
    # Obtain the masked genotype matrices, SNP identifiers, and haplotype identifiers
    logging.info("------ Array Processing: ------")
    
    # Extract genotype data, sample identifiers, variant identifiers, and positions from the SNPObject
    calldata_gt, haplotypes, variants_id = process_snpobj(snpobj, rsid_or_chrompos)

    if is_masked:
        # Obtain a SNP-level ancestry matrix by matching genomic positions and chromosomes in the SNPObject 
        # with ancestry segments in the LocalAncestryObject
        ancestry_matrix = process_laiobj(laiobj, snpobj)
        # Mask the genotype matrix by retaining only the entries that match a given ancestry
        mask = mask_calldata_gt(ancestry_matrix, calldata_gt, ancestry, average_strands, force_nan_incomplete_strands)
    else:
        # If averaging strands is enabled, compute the average SNP values per individual
        if average_strands:
            calldata_gt = average_parent_snps(calldata_gt)
        
        # Store the unmasked genotype data in the mask dictionary
        mask = {}
        mask[ancestry] = calldata_gt

        logging.info("No masking")

    if average_strands:
        # Remove duplicate haplotype identifiers (A/B strand labels)
        haplotypes = remove_AB_indIDs(haplotypes)
    
    return mask, variants_id, haplotypes


def process_labels_weights(
        labels_file, 
        mask, 
        variants_id, 
        haplotypes, 
        average_strands, 
        ancestry, 
        min_percent_snps, 
        group_snp_frequencies_only,
        groups_to_remove, 
        is_weighted, 
        save_masks, 
        masks_file,
    ):
    """
    Process individual genomic labels and weights, aligning them with a masked genotype matrix by 
    filtering out low-coverage individuals, reordering data to match the matrix structure, and 
    handling group-based adjustments.

    Steps:
      1. Load label/weight data: Read labels from a `.tsv` file and convert individual identifiers (`indID`) to string.
      2. Align genotype matrix columns: Reorder columns in the genotype matrix (`mask[ancestry]`) to match 
         the sequence of individuals (`indID`) from the labels file.
      3. Assign weights: If `is_weighted=True`, assign weights from the `weight` column of `labels_file`; 
         otherwise, all individuals receive a default weight of `1.0`.
      4. Filter individuals based on SNP coverage (`min_percent_snps`) and remove unwanted groups: 
         Compute the percentage of non-missing SNPs per individual from `mask[ancestry]`. Any individual 
         with SNP coverage below `min_percent_snps` is removed. Additionally, individuals belonging to 
         groups listed in `groups_to_remove` are excluded by setting their weight to `0`.
      5. Remove individuals with zero weight: After filtering, any remaining individuals with a weight of `0` 
         are removed from `mask[ancestry]`, `haplotypes`, `labels`, and `weights`.
      6. Identify individuals assigned to the same `combination` group, average their genotype data, and create a new merged individual. 
         The new entry is appended to `mask[ancestry]`, and its weight is assigned using `combination_weight`.
      7. Update and optionally save the processed mask, haplotypes, labels, and weights.

    Args:
        labels_file (str, optional): 
            Path to a `.tsv` file with metadata on individuals, including population labels, optional weights, and groupings. 
                The `indID` column must contain unique individual identifiers matching those in `laiobj` and `snpobj` for proper alignment. 
                The `label` column assigns population groups. If `is_weighted=True`, a `weight` column must be provided, assigning a weight to 
                each individual, where those with a weight of zero are removed. Optional columns include `combination` and `combination_weight` 
                to aggregate individuals into combined groups, where SNP frequencies represent their sequences. The `combination` column assigns 
                each individual to a specific group (0 for no combination, 1 for the first group, 2 for the second, etc.). All members of a group 
                must share the same `label` and `combination_weight`. If `combination_weight` column is not provided, the combinations are 
                assigned a default weight of `1`. Individuals excluded via `groups_to_remove` or those falling below `min_percent_snps` are removed 
                from the analysis.
        mask (dict of str to np.ndarray): 
            A dictionary where key represents the ancestry identifier, and value is the corresponding genotype matrix.  
            If `is_masked=True`, the matrix contains only genotype data for the specified ancestry, with all other ancestries set to NaN.  
            If `is_masked=False`, the full, unmasked genotype matrix is returned.
        variants_id (list of str): 
            A list of unique SNP identifiers in the specified format (`rsid_or_chrompos`).  
            The list may be reduced if some SNPs are not present in the `laiobj`.
        haplotypes (list of str): 
            A list of unique individual sample identifiers.  
            If haplotypes are averaged (`average_strands=True`), duplicate haplotype labels (e.g., "A" and "B") are removed.
        average_strands (bool): 
            Whether to average haplotypes for each individual.
        ancestry (str): 
            Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at `0`.
        min_percent_snps (float): 
            Minimum percentage of SNPs that must be known for an individual to be included in the analysis.
            All individuals with fewer percent of unmasked SNPs than this threshold will be excluded.
        group_snp_frequencies_only (bool):
            If True, mdPCA is performed exclusively on group-level SNP frequencies, ignoring individual-level data. This applies when `is_weighted` is 
            set to True and a `combination` column is provided in the `labels_file`,  meaning individuals are aggregated into groups based on their assigned 
            labels. If False, mdPCA is performed on individual-level SNP data alone or on both individual-level and group-level SNP frequencies when 
            `is_weighted` is True and a `combination` column is provided.
        groups_to_remove (list of str): 
            List with groups to exclude from analysis. Example: ['group1', 'group2'].
        is_weighted (bool): 
            If `True`, assigns individual weights from the `weight` column in `labels_file`. Otherwise, all individuals have equal weight of `1`.
        save_masks (bool): 
            True if the masked matrices are to be saved in a `.npz` file, or False otherwise.s
        masks_file (str or pathlib.Path): 
            Path to the `.npz` file used for saving/loading masked matrices.

    Returns:
        tuple:  
            - mask (dict of str to np.ndarray):  
                A dictionary where key represents the ancestry identifier, and value is the corresponding genotype matrix.  
                If `is_masked=True`, the matrix contains only genotype data for the specified ancestry, with all other ancestries set to NaN.  
                If `is_masked=False`, the full, unmasked genotype matrix is returned.
            - haplotypes (list of str):  
                A list of unique individual sample identifiers after filtering.  
                If haplotypes are averaged (`average_strands=True`), duplicate haplotype labels (e.g., "A" and "B") are removed.
            - label_list (np.ndarray of str):  
                A NumPy array containing the labels assigned to each individual after processing.  
                The labels may be repeated or adjusted depending on strand averaging and filtering criteria.
            - weight_list (np.ndarray of float):  
                A NumPy array containing the weight assigned to each individual after processing.  
                If `is_weighted=True`, weights are assigned based on the input file; otherwise, all weights default to 1.  
                Individuals with zero weight are excluded from the final output.
    """
    # Load the labels file
    labels_df = pd.read_csv(labels_file, sep='\t')
    labels_df['indID'] = labels_df['indID'].astype(str)
    
    if average_strands:
        # Keep IDs as-is and match them directly
        labels = np.array(labels_df['label'][labels_df['indID'].isin(haplotypes)])
        label_ind_IDs = np.array(labels_df['indID'][labels_df['indID'].isin(haplotypes)])
    else:
        # Remove "A"/"B" suffixes for matching, then duplicate labels to cover both strands
        temp_ind_IDs = remove_AB_indIDs(haplotypes)
        labels = np.array(labels_df['label'][labels_df['indID'].isin(temp_ind_IDs)])
        labels = np.repeat(labels, 2)
        label_ind_IDs = np.array(labels_df['indID'][labels_df['indID'].isin(temp_ind_IDs)])
        label_ind_IDs = add_AB_indIDs(label_ind_IDs)
        
    # Align masked matrix columns with the order of label_ind_IDs
    keep_indices = [haplotypes.tolist().index(x) for x in label_ind_IDs]
    mask[ancestry] = mask[ancestry][:, keep_indices]
    haplotypes = haplotypes[keep_indices]

    # Initialize or retrieve weights and combinations
    if not is_weighted:
        # If weighting is not enabled, assign a default weight of 1 to all individuals
        weights = np.ones(len(labels))
        # Default combination values to zero (no merging of individuals)
        combinations = np.zeros(len(labels))
        combination_weights = np.zeros(len(labels))
    else:
        if average_strands:
            weights = np.array(labels_df['weight'][labels_df['indID'].isin(haplotypes)])
            if 'combination' in labels_df.columns:
                combinations = np.array(labels_df['combination'][labels_df['indID'].isin(haplotypes)])
            else:
                combinations = np.zeros(len(weights))
            if 'combination_weight' in labels_df.columns:
                combination_weights = np.array(labels_df['combination_weight'][labels_df['indID'].isin(haplotypes)])
            else:
                combination_weights = np.ones(len(weights))
        else:
            temp_ind_IDs = remove_AB_indIDs(haplotypes)
            # Retrieve once, then duplicate for A/B strands
            weights = np.array(labels_df['weight'][labels_df['indID'].isin(temp_ind_IDs)])
            weights = np.repeat(weights, 2)
            if 'combination' in labels_df.columns:
                combinations = np.array(labels_df['combination'][labels_df['indID'].isin(temp_ind_IDs)])
                combinations = np.repeat(combinations, 2)
            else:
                combinations = np.zeros(len(weights))
            if 'combination_weight' in labels_df.columns:
                combination_weights = np.array(labels_df['combination_weight'][labels_df['indID'].isin(temp_ind_IDs)])
                combination_weights = np.repeat(combination_weights, 2)
            else:
                combination_weights = np.ones(len(weights))

    # Remove specified groups by setting their weight to zero
    if groups_to_remove:
        for i in range(len(labels)):
            if labels[i] in groups_to_remove:
                weights[i] = 0
    
    # Exclude individuals (columns) whose SNP coverage is below min_percent_snps
    percent_snps = 100 * (1 - np.mean(np.isnan(mask[ancestry]), axis=0))
    keep_indices = np.argwhere(percent_snps >= min_percent_snps).flatten()
    mask[ancestry] = mask[ancestry][:,keep_indices]
    haplotypes = haplotypes[keep_indices]
    labels = labels[keep_indices]
    weights = weights[keep_indices]
    combinations = combinations[keep_indices]
    combination_weights = combination_weights[keep_indices]

    # Exclude individuals with zero weight
    keep_indices = np.argwhere(weights > 0).flatten()
    mask[ancestry] = mask[ancestry][:, keep_indices]
    haplotypes = haplotypes[keep_indices]
    labels = labels[keep_indices]
    weights = weights[keep_indices]
    combinations = combinations[keep_indices]
    combination_weights = combination_weights[keep_indices]

    # Identify and combine individuals that share the same "combination" > 0
    pos_combinations = sorted(set(combinations[combinations > 0]))
    for combination in pos_combinations:
        # Find indices of all individuals with this combination
        combined_indices = np.argwhere(combinations == combination)

        # Compute the mean column across these individuals
        snp_frequencies = np.nanmean(mask[ancestry][:, combined_indices], axis=1)

        # Append the SNP frequencies column directly to the original mask
        mask[ancestry] = np.append(mask[ancestry], snp_frequencies, axis=1)

        # Append corresponding haplotype, label, and weight in-place
        haplotypes = np.append(haplotypes, 'combined_ind_' + str(combination))
        labels = np.append(labels, labels[combined_indices[0][0]])
        weights = np.append(weights, combination_weights[combined_indices[0][0]])

    # Remove original individual sequences if exclude_individuals_after_grouping is True
    if pos_combinations and group_snp_frequencies_only:
        keep_indices = np.where(np.char.find(haplotypes, 'combined_ind_') >= 0)[0]
        mask[ancestry] = mask[ancestry][:, keep_indices]
        haplotypes = haplotypes[keep_indices]
        labels = labels[keep_indices]
        weights = weights[keep_indices]
    
    # Optionally save the updated mask and accompanying arrays
    if save_masks:
        np.savez_compressed(masks_file, mask=mask, variants_id=variants_id, haplotypes=haplotypes,
                 labels=labels, weights=weights, protocol=4)
    
    return mask, haplotypes, labels, weights


def center_masked_matrix(masked_matrix):
    masked_matrix -= np.nanmean(masked_matrix, axis=0)
    return masked_matrix


def logger_config(verbose=True):
    logging_config = {"version": 1, "disable_existing_loggers": False}
    fmt = '[%(levelname)s] %(asctime)s: %(message)s'
    logging_config["formatters"] = {"basic": {"format": fmt, "datefmt": "%Y-%m-%d %H:%M:%S"}}
    now = datetime.datetime.now()

    logging_config["handlers"] = {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG" if verbose else "INFO",
                "formatter": "basic",
                "stream": "ext://sys.stdout"
            },
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG" if verbose else "INFO", 
                "formatter": "basic",
                "maxBytes": 10485760,
                "backupCount": 20,
                "filename": f"log{now.year}_{now.month}_{now.day}__{now.hour}_{now.minute}.txt", # choose a better name or name as param?
                "encoding": "utf8"
                }
            }
    logging_config["root"] = {
                "level": "DEBUG",
                "handlers": ["console", "info_file_handler"]
            }
    return logging_config
