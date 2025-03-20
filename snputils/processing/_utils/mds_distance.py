# Library Importation
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import plotly
import plotly_express as px 
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._mask import _get_mask
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import is_scalar_nan
import time
import warnings
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
warnings.simplefilter("ignore", category=RuntimeWarning)
from sklearn.impute import SimpleImputer


def demean(S, w):
    w_sum = np.matmul(w, S)
    w_rowsum = w_sum.reshape((1, S.shape[0]))
    w_colsum = w_sum.reshape((S.shape[0], 1))
    w_totalsum = np.dot(w_sum, w)
    S -= (w_rowsum + w_colsum) -  w_totalsum
    return S

def weighted_cmdscale(D, weights, num_dims):
    n = len(D)
    H = np.eye(n) - np.ones((n, n))/n
    S = -H.dot(D**2).dot(H)/2
    weights_normalized = weights / weights.sum()
    S = demean(S, weights_normalized)
    W = np.diag(weights)
    WSW = np.matmul(np.matmul(np.sqrt(W), S), np.sqrt(W))
    svd = TruncatedSVD(num_dims, algorithm="arpack")
    svd.fit(WSW)
    dist_transformed = np.matmul(svd.components_, np.linalg.inv(np.sqrt(W))).T
    return dist_transformed

def nan_manhattan_distances(X, Y=None):    
    missing_values=np.nan
    force_all_finite = 'allow-nan' if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 force_all_finite=force_all_finite)
    missing_X = _get_mask(X, missing_values)
    missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)
    
    if X is Y:
        X_is_Y = True
    else:
        X_is_Y = False

    X[missing_X] = 0
    Y[missing_Y] = 0

    distances = manhattan_distances(X, Y)

    X = np.abs(X)
    Y = np.abs(Y)
    Y = Y.T
    distances -= np.dot(X, missing_Y.T)
    distances -= np.dot(missing_X, Y)

    np.clip(distances, 0, None, out=distances)

    if X_is_Y:
        np.fill_diagonal(distances, 0.0)

    present_X = ~missing_X
    present_Y = present_X if X_is_Y else ~missing_Y
    X = present_X.astype(np.float32)
    Y = present_Y.T.astype(np.float32)
    present_count = np.dot(X, Y)
    distances[present_count == 0] = np.nan
    np.maximum(1, present_count, out=present_count)
    distances /= present_count

    return distances

def nan_rms_distances(X, Y=None):    
    missing_values=np.nan
    force_all_finite = 'allow-nan' if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 force_all_finite=force_all_finite)
    missing_X = _get_mask(X, missing_values)
    missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)

    if X is Y:
        X_is_Y = True
    else:
        X_is_Y = False
    
    X[missing_X] = 0
    Y[missing_Y] = 0

    distances = euclidean_distances(X, Y, squared=True)

    X = X * X
    Y = Y * Y
    Y = Y.T
    distances -= np.dot(X, missing_Y.T)
    distances -= np.dot(missing_X, Y)

    np.clip(distances, 0, None, out=distances)

    if X_is_Y:
        np.fill_diagonal(distances, 0.0)

    present_X = ~missing_X
    present_Y = present_X if X_is_Y else ~missing_Y
    X = present_X.astype(np.float32)
    Y = present_Y.T.astype(np.float32)
    present_count = np.dot(X, Y)
    distances[present_count == 0] = np.nan
    np.maximum(1, present_count, out=present_count)
    distances /= present_count
    
    np.sqrt(distances, out=distances)

    return distances

def nan_ap_distances(X, Y=None):
    missing_values=np.nan
    force_all_finite = 'allow-nan' if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 force_all_finite=force_all_finite)
    missing_X = _get_mask(X, missing_values)
    missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)
    
    if X is Y:
        X_is_Y = True
    else:
        X_is_Y = False
    
    X[missing_X] = 0
    Y[missing_Y] = 0

    present_X = ~missing_X
    present_Y = present_X if X_is_Y else ~missing_Y
    
    singlerow_X = np.dot(X, present_Y.T)
    singlerow_Y = np.dot(Y, present_X.T)
    Y = Y.T
    distances = singlerow_X - 2 * np.dot(X, Y) + singlerow_Y.T
    
    if X_is_Y:
        np.fill_diagonal(distances, 0.0)
    
    X = present_X.astype(np.float32)
    Y = present_Y.T.astype(np.float32)
    present_count = np.dot(X, Y)    
    distances[present_count == 0] = np.nan
    np.maximum(1, present_count, out=present_count)
    distances /= present_count
    
    return distances

def distance_mat(first, second=None, dist_func = 'AP'):
    """                                                                                       
    Distance Matrix Computation Function between two different arrays.
    
    Parameters                                                                                
    ----------                                                                                
    first       : (m, n1) array
                  First genome matrix for a chosen ancestry. Indicates which genetic encoding
                  is present for individual n at position m, or NaN if position is masked.
                  
    second      : (m, n2) array
                  Second genome matrix for a chosen ancestry. Indicates which genetic encoding
                  is present for individual n at position m, or NaN if position is masked.
                  
    dist_func   : string
                  Distance function utilized in computation. Defaulted to be 'AP'.
                  Options to choose from are: 'Manhattan', 'RMS', 'AP'
                                                                                
    Returns                                                                                   
    -------                                                                                   
    distance   : (n1, n2) array                                                                       
                 Computed distance between two individuals i of first and j of second provided at entry (i, j).
    """
    start = time.time()
    first = first.T
    first = first.astype(np.float32)
    
    if second is not None:
        second = second.T
        second = second.astype(np.float32)
        if second.shape[0] == 0:
            logging.info("Distance Matrix building: --- %s seconds ---" % (time.time() - start))
            return np.array([[]] * first.shape[0])
        if first.shape[0] == 0:
            logging.info("Distance Matrix building: --- %s seconds ---" % (time.time() - start))
            return np.array([[]] * second.shape[0]).T
        
    if first.shape[0] == 0:
        logging.info("Distance Matrix building: --- %s seconds ---" % (time.time() - start))
        return np.array([[]]).T.dot(np.array([[]]))
    
    if (dist_func == 'Manhattan'):
        distance = nan_manhattan_distances(first, second)   
        
    elif (dist_func == 'RMS'):
        distance = nan_rms_distances(first, second)
        
    elif (dist_func == 'AP'):
        distance = nan_ap_distances(first, second)
            
    else:
        logging.info("Distance Function Undefined -- Defaulted to AP Distance")
        distance = nan_ap_distances(first, second)
    logging.info("Distance Matrix building: --- %s seconds ---" % (time.time() - start))
    
    return distance


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))  

def binary_intersection(rs_ID_list):
    """                                                                                       
    Intersecting Positions Calculation between two arrays:
    
    Parameters                                                                                
    ----------                                                                                
    rs_ID_list  : (num_arrays, ) list 
        List of rs IDs for each of the processed arrays.
                                                                                
    Returns                                                                                   
    -------                                                                                   
    binary   : (num_arrays, num_arrays) list                                                                      
        Entry (i, j) contains an array of intersecting rs IDs between array i,
        and array j.
    """
    binary = []
    for i in range(0,len(rs_ID_list)):
        binary.append([1]*(i+1))
        first = list(rs_ID_list[i])
        for j in range(i+1, len(rs_ID_list)):
            second = list(rs_ID_list[j])
            intersect = intersection(first, second)
            binary[i].append(intersect)
            
    for i in range(len(binary)):
        logging.info("Array %s", i+1)
        for j in range(i+1,len(binary[i])):
            logging.info("Intersect with %s : %s", j+1, len(binary[i][j]))
            
    return binary

def build_overlap(overlap, array1_ID, array2_ID, array1, array2):
    '''
    Build Overlap Function.
    
    Parameters                                                                                
    ----------                                                                                
    overlap  : List of rs IDs in common between the two arrays.
    array1_ID: List of all rs IDs in the first array.
    array2_ID: List of all rs IDs in the second array.
    array1   : Masked genetic matrix for the first array.
    array2   : Masked genetic matrix for the second array.

    Returns                                                                                   
    -------                                                                                   
    overlap_mat   : array representing the combined genetic matrix of 
                    individuals in both arrays across their common snps.
    '''
    index_1 = np.where(np.in1d(array1_ID, overlap))[0]
    index_2 = np.where(np.in1d(array2_ID, overlap))[0]

    return [array1[index_1],array2[index_2]]


def overlap_blocks(ancestry, ref_col, ref_row, num_arrays, rs_ID_list, binary, masks):
    """                                                                                       
    Binary Intersection Array Computation:
    
    Parameters                                                                                
    ---------- 
    ancestry    : integer
                  Ancestry of interest to perform distance matrix computation on.
    ref_col     : integer
                  Column of reference for conversion calculation.
    ref_row     : integer
                  Row of reference for conversion calculation.
    num_arrays  : integer
                  Number of arrays in dataset.
    rs_ID_list  : (num_arrays, ) list 
                  List of rs IDs for each of the processed arrays.
    binary      : (num_arrays, num_arrays) list                                                                      
                  Entry (i, j) contains an array of intersecting rs IDs between array i,
                  and array j.
    masks       : (num_arrays, ) list                                                                          
                  List of masked matrices for each ancestries at each given array.
                                                                   
    Returns                                                                                   
    -------                                                                                   
    overlap   : (num_arrays, num_arrays) list
                Sets of overlapping matrices between two arrays across common snps
    """    
    overlap = []
    for i in range(num_arrays):
        overlap.append([1]*i)
        for j in range(i, num_arrays):
            if (j == 0):
                ind = np.where(np.in1d(rs_ID_list[j], np.array(binary[ref_row][ref_col])))[0]
                overlap[i].append(masks[j][ancestry][ind, :])
            elif (j == i):
                ind = np.where(np.in1d(rs_ID_list[j], binary[ref_row][j]))[0]
                overlap[i].append(masks[j][ancestry][ind, :])
            else:
                overlap[i].append(build_overlap(np.array(binary[i][j]), rs_ID_list[i], rs_ID_list[j], masks[i][ancestry], masks[j][ancestry]))
    return overlap


def convert(overlap_1, overlap_2, array, rs_IDs, distance_type, plot_reg, index):
    '''
    Build Overlap Function.
    
    Parameters                                                                                
    ----------                                                                                
    overlap_1      : First set of overlapping positions between two arrays.
    overlap_2      : Second set of overlapping positions between two arrays.
    array          : Combined matrix of two arrays individuals across common snps.
    rs_IDs         : List of all positions present in array. 
    distance_type  : Distance function utilized in computation. Defaulted to be 'AP'.
                     Options to choose from are: 'Manhattan', 'Euclidean', 'AP'

    Returns                                                                                   
    -------                                                                                   
    [reg.coef_, reg.intercept_]   : conversion matrix for distance matrix.
    
    '''
    ind1 = np.where(np.in1d(rs_IDs, overlap_1))[0]
    ind2 = np.where(np.in1d(rs_IDs, overlap_2))[0]   
    
    logging.info("--- Number of overlapping positions: --- %s", len(ind1))
    dist1 = distance_mat(first=array[ind1], dist_func=distance_type)
    dist2 = distance_mat(first=array[ind2], dist_func=distance_type)

    dist1 = np.reshape(dist1, (dist1.size, 1)).flatten()
    dist2 = np.reshape(dist2, (dist2.size, 1)).flatten()
    
    remove1 = np.argwhere(np.isnan(dist1)).flatten()
    remove2 = np.argwhere(np.isnan(dist2)).flatten()
    remove = np.array(list(set(remove1) | set(remove2)), dtype = np.int32)
    
    dist1 = np.delete(dist1, remove)
    dist2 = np.delete(dist2, remove)
    
    dist1 = np.reshape(dist1, (dist1.size, 1))
    dist2 = np.reshape(dist2, (dist2.size, 1))

    reg = LinearRegression().fit(dist1, dist2)
    
    if (plot_reg):
        plt.scatter(dist1, dist2, marker='x')
        plt.plot(dist1, dist1 * reg.coef_ + reg.intercept_)
        plt.savefig('conv_plot_' + index + '.jpg')
        plt.clf()

    return [reg.coef_, reg.intercept_, reg.score(dist1,dist2)]

def conversion_metrics(ancestry, ref_col, ref_row, num_arrays, rs_ID_list, binary, masks, distance_type, plot_reg=False):
    """                                                                                       
    Conversion Matrix Computation Function:
    
    Parameters                                                                                
    ---------- 
    ancestry       : integer
                    Ancetry of interest to perform distance matrix computation on.
    ref_col        : integer
                    Column of reference for conversion calculation.
    ref_row        : integer
                     Row of reference for conversion calculation.
    num_arrays     : integer
                     Number of arrays in dataset.
    rs_ID_list     : (num_arrays, ) list 
                     List of rs IDs for each of the processed arrays.
    binary         : (num_arrays, num_arrays) list                                                                      
                    Entry (i, j) contains an array of intersecting rs IDs between array i,
                    and array j.
    masks          : (num_arrays, ) list                                                                          
                     List of masked matrices for each ancestries at each given array.
    distance_type  : string
                     Distance function utilized in computation. Defaulted to be 'AP'.
                     Options to choose from are: 'Manhattan', 'Euclidean', 'AP'
                                                                   
    Returns                                                                                   
    -------                                                                                   
    conversion  : (num_arrays, num_arrays) list
                  Conversion coefficients for each of the arrays in our dataset.
    intercept   : (num_arrays, num_arrays) list
                  Conversion intercepts for each of the arrays in our dataset.
    """
    conversion = np.ones((num_arrays,num_arrays))
    intercept = np.zeros((num_arrays,num_arrays))
    score = np.zeros((num_arrays,num_arrays))
    for j in range(ref_col+1, num_arrays):
        for i in range(j+1):
            index = str(10*j+i)
            if (i == ref_col):
                conv_list = convert(binary[i][j], binary[ref_row][ref_col], masks[i][ancestry], rs_ID_list[i], distance_type, plot_reg, index)
                conversion[i][j] = conv_list[0]
                intercept[i][j] = conv_list[1]
                score[i][j] = conv_list[2]
            elif (i == ref_row):
                conv_list = convert(binary[i][j], binary[ref_row][ref_col], masks[i][ancestry], rs_ID_list[i], distance_type, plot_reg, index)
                conversion[i][j] = conv_list[0]
                intercept[i][j] = conv_list[1]                
                score[i][j] = conv_list[2]
            elif (i != j):
                conv_list = convert(binary[i][j], binary[ref_row][j], masks[j][ancestry], rs_ID_list[j], distance_type, plot_reg, index)
                conversion[i][j] = conv_list[0]
                intercept[i][j] = conv_list[1]
                score[i][j] = conv_list[2]
            else:
                conversion[i][j] = conversion[ref_row][j]
                intercept[i][j] = intercept[ref_row][j]
                score[i][j] = conv_list[2]

    # Symmetric:
    for i in range(conversion.shape[0]):
        for j in range(conversion.shape[1]):
            conversion[j][i] = conversion[i][j]
            intercept[j][i] = intercept[i][j]
            score[j][i] = score[i][j]
    
    np.savetxt('conversion.csv', conversion, delimiter=',')
    np.savetxt('intercept.csv', intercept, delimiter=',')
    np.savetxt('score.csv', score, delimiter=',')

    return conversion, intercept

def distance_overlap(ref_col, ref_row, num_arrays, overlap, conversion, intercept, distance_type):
    """                                                                                       
    Binary Intersection Distance Matrix Computation:
    
    Parameters                                                                                
    ---------- 
    ref_col     : integer
                  Column of reference for conversion calculation.
    ref_row     : integer
                  Row of reference for conversion calculation.
    num_arrays  : integer
                  Number of arrays in dataset.
    overlap     : (num_arrays, num_arrays) list
                  Sets of overlapping matrices between two arrays across common snps
    conversion  : (num_arrays, num_arrays) list
                  Conversion coefficient estimated from linear fit, to account for the 
                  difference in chosen snps between any two arrays, and the reference
                  arrays determined with a row and column.
    intercept   : (num_arrays, num_arrays) list
                  Conversion intercept estimated from linear fit, to account for the 
                  difference in chosen snps between any two arrays, and the reference
                  arrays determined with a row and column.
    distance_type : string
                  Distance function utilized in computation. Defaulted to be 'AP'.
                  Options to choose from are: 'Manhattan', 'Euclidean', 'AP'
                                                                                
    Returns                                                                                   
    -------                                                                                   
    dist_lis   : (num_arrays, num_arrays) list                                                                      
        Entry (i, j) contains a distance matrix array of intersecting rs IDs between array i,
        and array j.
    """
    distance_list = []
    for i in range(len(overlap)):
        distance_row = []
        for j in range(len(overlap[0])):
            if (j > i):
                distance_row.append(distance_mat(first=overlap[i][j][0], second=overlap[i][j][1], dist_func=distance_type))
            elif (i == j):
                distance_row.append(distance_mat(first=overlap[i][j], dist_func=distance_type))
            else:
                distance_row.append(1)
        distance_list.append(distance_row)

    # Make distance_list symmetric
    for i in range(len(distance_list)):
        for j in range(len(distance_list[0])):
            if (j < i):
                distance_list[i][j] = distance_list[j][i]
                
#     dist_lis = [[distance_list[0][0],distance_list[0][1]],[distance_list[1][0],distance_list[1][1]], [1, 1], [1, 1, 1]]
    dist_lis = [[distance_list[0][0],distance_list[0][1]],[distance_list[1][0],distance_list[1][1]]]
    for i in range(2,num_arrays):
        dist_lis.append(i*[1])
        
    for j in range(ref_col+1, num_arrays):
        for i in range(j+1):
            if (i == ref_col) or (i == ref_row):
                dist_lis[i].append(conversion[i][j]*distance_list[i][j] + intercept[i][j])
            elif (i != j):
                dist_lis[i].append((distance_list[i][j]*conversion[i][j] + intercept[i][j])*conversion[ref_row][j]+intercept[ref_row][j])
            else:
                dist_lis[i].append(conversion[ref_row][j]*distance_list[i][j] + intercept[ref_row][j])

    # Make distance_list symmetric
    for i in range(len(dist_lis)):
        for j in range(len(dist_lis[0])):
            if (j < i):
                dist_lis[i][j] = dist_lis[j][i].T  
                
    return dist_lis

def combine_dist_mat(distance_list):
    """                                                                                       
    Combine Distance Matrix Function:
    
    Parameters                                                                                
    ---------- 
    distance_list: (num_arrays, num_arrays) list 
                   List of computed distances on individuals of two intersecting arrays 
                   over the same set of snp's.
                                                                                
    Returns                                                                                   
    -------                                                                                   
    dist_mat     : (m, m)                                                                     
                   Combined and converted distance matrix of all individuals in the dataset.
    """
    # Construct Full Distance Matrix over selected individuals of overlap
    dist_mat = distance_list[0][0]
    for i in range(1,len(distance_list[0])):
        dist_mat = np.hstack((dist_mat, distance_list[0][i]))#*conversion[0][i]))

    for i in range(1,len(distance_list)):
        dist_row = distance_list[i][0]
        for j in range(1,len(distance_list[i])):
            if (i == j): 
                dist_row = np.hstack((dist_row, distance_list[i][j]))
            elif (i < j):
                dist_row = np.hstack((dist_row, distance_list[i][j]))
            else: 
                dist_row = np.hstack((dist_row, distance_list[i][j]))
        dist_mat = np.vstack((dist_mat, dist_row))
    return dist_mat

def mean_impute(d):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(d)
    d = imp.transform(d)
    return d

def additive_impute(d):
    num_missing = np.isnan(d).sum()
    n = d.shape[0]
    while True:
        old_num_missing = num_missing
        for i in range(n):
            for j in range(n):
                if np.isnan(d[i,j]):
                    MinMax = np.nanmax(d)
                    fill_d = False
                    for k in range(n):
                        for l in range(n):
                            if (np.isnan([d[i,k], d[j,k], d[i,l], d[j,l], d[k,l]]) == False).all():
                                fill_d = True
                                Max = max((d[i,k] + d[j,l]), (d[i,l] + d[j,k])) - d[k,l]
                                if Max < MinMax:
                                    MinMax = Max
                    if fill_d:
                        d[i,j] = MinMax
                        num_missing -= 1
        if num_missing == old_num_missing:
            break
    return d

def remove_individuals(dist_mat, groups, weights, ind_IDs):
    keep_inds = np.argwhere(np.isnan(dist_mat).sum(axis=0) != dist_mat.shape[0]).flatten()
    dist_mat = dist_mat[keep_inds, :][:, keep_inds]
    groups = groups[keep_inds]
    weights = weights[keep_inds]
    ind_IDs = ind_IDs[keep_inds]
    return dist_mat, groups, weights, ind_IDs

def mds_transform(distance_list, groups, weights, ind_ID_list, num_dims, num_arrays=1):
    """                                                                                       
    Multi-Dimensional Scaling (MDS) Function:
    
    Parameters                                                                                
    ---------- 
    distance_list: (num_arrays, num_arrays) list 
                   List of computed distances on all individuals in the dataset. 
    groups       : (n, ) array
                   Corresponding groups for each individual in the dataset.
                   Order is consistent to order of individuals in the distance matrix.
    ind_ID_list  : 
                   List of individual IDs for each of the processed arrays.
    num_arrays   : Total number of arrays in dataset.
                                                                                
    Returns                                                                                   
    -------                                                                                   
    None
    """
    ind_IDs = ind_ID_list
    for i in range(1,num_arrays):
        ind_IDs = np.hstack((ind_IDs, ind_ID_list[i]))
    
    dist_mat = combine_dist_mat(distance_list)
    dist_mat, groups, weights, ind_IDs = remove_individuals(dist_mat, groups, weights, ind_IDs)
    np.savetxt('distance_unimputed.csv', dist_mat, delimiter=',')
    
    dist_mat = additive_impute(dist_mat)
    np.savetxt('distance_imputed.csv', dist_mat, delimiter=',')
    dist_transformed = weighted_cmdscale(dist_mat, weights, num_dims)
    
    categories = pd.factorize(pd.Series(groups))
    
    output_df = pd.DataFrame()
    output_df['indID'] = ind_IDs
    output_df['label'] = groups
    for i in range(num_dims):
        output_df['MDS' + str(i+1)] = dist_transformed[:,i]
    output_df.to_csv('output.tsv', sep='\t', index=False)

    return dist_transformed[:, :num_dims]