import sys

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, theme, position_dodge, scale_fill_grey
from plotnine import geom_histogram, geom_violin, geom_boxplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib import artificial_data



################################################################################
# ----------------------------- Matrix utilities ----------------------------- #
################################################################################


def zeropad(arr, pad_width):
    """
    Pad array with zeros.
    pad_width : sequence of tuple[int, int], like in np.pad
        Pad width on both sides for each dimension in `arr`.
    """
    # Allocate grown array
    new_shape = tuple(s + sum(p) for s, p in zip(arr.shape, pad_width))
    padded = np.zeros(new_shape, dtype=arr.dtype)

    # Copy old array into correct space
    old_area = tuple(
        slice(None if left == 0 else left, None if right == 0 else -right)
        for left, right in pad_width
    )
    padded[old_area] = arr

    return padded




def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape.
    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.
    """
    operation = operation.lower()
    if not operation in ['sum', 'mean', 'max']: raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

# MaxPooling a 3d matrix along its first axis (the [0] axis, for "region length")
def squish(m, factor=10):
    # Calculate new shape based on desired reduction
    new_shape = tuple([int(m.shape[0]/factor)]) + (m.shape[1:])
    return bin_ndarray(m, new_shape=new_shape, operation='max')

def stretch(m, factor=10):
    # Kronecker product to up-sample.
    # The only differences with the original will be due to rounding (if the
    # peaks lengths were not divisible by the squishing factor).
    if len(m.shape) != 4 : raise ValueError('stretch() needs a 4D matrix (region_length, nb_datasets, nb_tf, channels=1)')
    return np.kron(m, np.ones((factor,1,1,1)))



def plot_3d_matrix(mat, figsize=(10,6),
                    no_zeros = True, rb_scale=True,
                    alpha = 0.5, guideline = True,
                    standard_scale = True,
                    tf_colors_override = None):
    """
    Plot any 3d numpy matrix. Useful to visualize data.

    Arguments :
        - no_zeroes (default True) : Zeroes are not plotted. Suited for sparse matrices. Independant of rb_scale.
        - rb_scale (default True) : if True, low values are in red and high values in blue; if False, those are resp. white and black.

        - alpha (default 0.5) : Controls the transparency of plotted points
        - guideline (default True) : draws a line across nonzero elements in the X axis to help see perspective
        - standard_scale (default True) : will assume the values are between 0 and 1.


    Example :

    >>> m = np.zeros((3200,6,4))
    >>> m[1000:2000,0,0] = m[1000:2000,1,0] = m[1000:2000,2,0] = 1
    >>> m[1200:2100,2,1] = m[1200:2100,3,1] = 1
    >>> m[1500:2300,2,2] = m[1500:2300,4,2] = 1
    >>> m[0:800,1,3] = 1
    >>> plot_3d_matrix(m, guideline = False, tf_colors_override = [(0,0,0.6),(0.5,0,0.5),(0,0.8,0),(1,0.6,0)])

    """

    ## Check : object must be a non-uniform 3d matrix
    if len(mat.shape) != 3:
        print('ERROR - Trying to draw a non-3D matrix.')
        return

    if np.min(mat) == np.max(mat):
        print('ERROR - All values in the matrix are the same (min and max are equal). Plotting is useless.')
        print('This matrix is just a repetition of ['+str(np.max(mat))+'] across a shape of '+str(mat.shape)+'.')
        return

    if (np.min(mat) < 0 or np.max(mat) > 1) and standard_scale:
        print('ERROR - You cannot use the standard scale with values below [0;1].')
        print('Defaulting to a relative scale between the min and max value.')
        standard_scale = False

    # Create the x, y, and z coordinate arrays. Use numpy's broadcasting.
    xu = np.arange(mat.shape[0])[:, None, None]
    yu = np.arange(mat.shape[1])[None, :, None]
    zu = np.arange(mat.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(xu, yu, zu)

    # Cast arrays as floats so we can use NaNs
    mat = mat.astype(float)
    x, y, z = x.astype(float), y.astype(float), z.astype(float)


    # If required, remove elements from `coordinates` if they are equal to zero
    # TODO It works, but is slow. Try to optimize this.
    if no_zeros :
        for xi in xu.ravel():
            for yi in yu.ravel():
                for zi in zu.ravel():
                    if np.equal(mat[xi,yi,zi],0) :
                        x[xi,yi,zi] = np.nan
                        y[xi,yi,zi] = np.nan
                        z[xi,yi,zi] = np.nan

    xr = x.ravel()[~np.isnan(x.ravel())]
    yr = y.ravel()[~np.isnan(y.ravel())]
    zr = z.ravel()[~np.isnan(z.ravel())]


    ## Turn the volumetric data into an RGB array.
    # NOTE The color of each point could be made conditional by applying operations to `c`, if needed.
    c_raw_list = [mat[int(xr[i]),int(yr[i]),int(zr[i])] for i in range(len(xr))] # xr, yz and zr have the same nb of elements
    c_raw = np.array(c_raw_list)
    c = np.tile(c_raw[:,None], [1, 3]) # Fake RGB tiling


    ## Color scale
    if not standard_scale: c = (c-np.min(mat))/(np.max(mat)-np.min(mat))

    # Plot values as white (low) -> black (high)
    c = 1 - c # High values should be black, not white, and should be between 0 and 1

    # If required, plot values as red (low) -> blue (high)
    if rb_scale :
        c[:,2] = 1 - c[:,2]
        c[:,1] = 0



    # DEBUG - TF color override for figures 
    if tf_colors_override is not None:
    # For display, use one color per TF, in the order given by a vector
    # tf_colors_override = [(255,255,0),(255,255,255),...]
        for i in range(len(xr)):
            c[i,:] = np.array(tf_colors_override[int(zr[i])])
        c = np.array(c)


    ## Do the plotting in a single call.
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.scatter(xr, yr, zr,
               c=c, s=20,
               alpha=alpha) # See-through to see far elements.

    # To better see perspective, for each y,z pair with nonzero elements, draw a line across it
    if guideline :
        for Y in range(mat.shape[1]):
            for Z in range(mat.shape[2]):
                if sum(mat[:,Y,Z]) != 0:
                    ax.plot([0 ,mat.shape[0]],[Y,Y],[Z,Z])

    # To understand the scale, print min, max and mean values.
    scale = 'Min = '+"{:.2E}".format(np.min(mat))+' - Mean = '+"{:.2E}".format(np.mean(mat))+' - Max = '+"{:.2E}".format(np.max(mat))
    fig.text(x=0.22, y=0.06, s=scale, fontweight='bold')

    # Set lims to the original matrix size
    ax.set_xlim(0,mat.shape[0])
    ax.set_ylim(0,mat.shape[1])
    ax.set_zlim(0,mat.shape[2])

    return fig # and then in main code do this_fig.savefig()













################################################################################
# ----------------------------- File processing ----------------------------- #
################################################################################


def produce_result_bed(origin_data_file, anomaly_matrix,
                        ordered_datasets, ordered_tfs,
                        crm_start, crm_length,
                        debug_print = False):
    """
    Takes the data file of original peak likes (as a list of strings), and the corresponding anomaly matrix.
    This also requires the list of datasets and TFs, to know which coordinate in the matrix corresponds to which.

    Also requires crm_start (BED position) and crm_length for coordinate correction.
    """


    # Convert origin_data_file from a list of strings to a list of lists
    origin_lines = [l.rstrip().split('\t') for l in origin_data_file]

    # For datasets and names : since they are given in the same order as the matrix,
    # prepare a conversion key form coordinate to dataset_id/tf_id
    ordered_datasets = {ordered_datasets[i]:i for i in np.arange(len(ordered_datasets))}
    ordered_tfs = {ordered_tfs[i]:i for i in np.arange(len(ordered_tfs))}

    result_bed = []
    # For each peak in the origin data file...
    for peak in origin_lines:

        # Collect its coordinates
        pad_to = anomaly_matrix.shape[0] # Fix coordinates due to padding
        xmin_original = peak[5]
        xmin = (pad_to - crm_length) + (int(xmin_original) - crm_start)
        xmax_original = peak[6]
        xmax = (pad_to - crm_length) + (int(xmax_original) - crm_start)


        # NOTE Sometimes peaks were inside the CRM for less than 10 base pairs, and those will have been squished out by the, well, squishing.
        # In that case, the anomaly matrix may be an improper slice.
        # To fix this, we use nanmean and we set xmin to be at least 0, and xmax to be maximum the matrix length.
        # xmax must also be at least one higher than xmin.
        xmin = max(0,xmin)
        xmax = min(pad_to, xmax)
        if xmax == xmin : xmax = xmin+1


        tf = ordered_tfs[peak[2]]
        dataset = ordered_datasets[peak[3]]


        # Collect its anomaly vector
        anomaly_vector = anomaly_matrix[xmin:xmax,dataset,tf]

























        ## Maximum score along the peak
        anomaly_score_raw = np.nanmax(anomaly_vector) * 1000
        anomaly_score = int(np.around(anomaly_score_raw))
        # We multiply by 1000 to match the typical BED score format

        if debug_print : print('Peak : ('+str(xmin)+' -> '+str(xmax)+','+str(tf)+','+str(dataset)+') -- Anomaly = '+str(anomaly_score))


        # Produce the result lines in ReMap format :
        # chr    start   end    dataset.tf.cell_line    score   strand
        result = (str(peak[4])+'\t'+str(xmin_original)+'\t'+str(xmax_original)+'\t'+
            peak[3]+'.'+peak[2]+'.'+peak[0]+'\t'+ #peak[the dataset]+'.'+peak[the tf]+'.'+peak[the cell line]+'\t'+
            str(anomaly_score)+'\t'+'.')
        # TODO Add gregariousness : how many peaks in the same row, column, or entire CRM.
        # Also keep the MACS score and the peak center ?

        result_bed.append(result)


    return result_bed






def print_merge_doublons(bedfilepath, ignore_first_line_header = True, outputpath = None):
    """
    Peaks can sometimes be divided into two different 3200-long matrices,
    meaning they get two scores, one for each rebuilt matrix.
    This function will open a file in pandas, and merge them, by giving them
    an anomaly score that is the mean of the two previous doublons

    Since the coordinates of the peaks are based on the original data file, not
    the matrix itself, merging is easier
    """

    # Default output path
    if outputpath is None : outputpath = bedfilepath + "_merged_doublons.bed"

    # If there was a header in the original file, we must ignore it, but also add it back to the output file
    if ignore_first_line_header:

        skiprows = 1

        with open(bedfilepath) as inf : header = inf.readline()
        with open(outputpath, 'a') as of : of.write(header)

    else : skiprows = None

    bed = pd.read_csv(bedfilepath, header = None, sep = '\t', skiprows = skiprows)

    # Take rounded mean for the same peaks that have two lines
    def rounded_mean(x): return np.round(np.mean(x))
    def keep_only_first(series): return series.iloc[0]
    mergedbed = bed.groupby(by=[0,1,2,3]).agg({4: rounded_mean, 5: keep_only_first})


    with open(outputpath, 'a') as of : # Open in append mode
        mergedbed.to_csv(of, header=False, index=True, sep = '\t')









def normalize_result_file_with_coefs_dict(result_file_path, scaling_factor_dict, cl_name, outfilepath = None):
    """
    Given a dictionary of the form : {(dataset, tf ): k} where dataset and tf are strings, will apply the
    k coefficient to each score of each peak of the corresponding combi
    """

    # Write a normalized file
    if outfilepath is None : outfilepath = result_file_path + "_normalized_corr_group.bed" # Default file path
    normalized_rf = open(outfilepath,'w')
    # Header
    normalized_rf.write('track name ='+cl_name+'_corr-group-normalized description="'+cl_name+' peaks with anomaly score - normalized by correlation group" useScore=1'+'\n')


    rf = open(result_file_path,'r')

    for line in rf:
        if (line[0:10] != 'track name'):
            line = line.split('\t') ; line[3] = line[3].split('.')
            dataset = line[3][0]; tf = line[3][1]
            score = float(line[4])

            # Apply scaling factor
            new_score = score * scaling_factor_dict[(dataset, tf)]

            # Round as integer
            new_score = int(np.around(new_score))

            # Rejoin line and write it
            line[4] = str(new_score)
            line[3] = '.'.join(line[3])
            line = '\t'.join(line)
            normalized_rf.write(line)


    normalized_rf.close()
    rf.close()













def normalize_result_file_score_by_tf(result_file_path, cl_name, outfilepath = None):

    scores_tf=dict()
    scores_datasets=dict()

    # Open the result file
    rf = open(result_file_path,'r')

    # Get minimum and maximum
    for line in rf:

        # If line is not a header...
        if (line[0:10] != 'track name'):

            line = line.split('\t') # Split by tab
            line[3] = line[3].split('.') # Further split the dataset.tf.cell_line line

            # Get the dataset
            dataset = line[3][0]

            # Get the TF
            tf = line[3][1]
            # Get the score
            score = float(line[4])

            # Set the observed minimum or observed maximum
            try:
                scores_tf[tf] += [score]
                scores_datasets[dataset] += [score]
            except: # Key is not present
                scores_tf[tf] = [score]
                scores_datasets[dataset] = [score]

    rf.close()




    # Process the collected scores

    # FOR THE TFS
    scores_df_tf = pd.DataFrame(columns = ["tf","count","mean","std","min","25pc","50pc","75pc","max"])
    for k,v in scores_tf.items():
        description = list(pd.Series(v).describe())
        scores_df_tf.loc[len(scores_df_tf)] = [k] + description
    scores_df_tf =scores_df_tf.set_index('tf')
    scores_df_tf.to_csv(result_file_path+'_scores_tf.tsv',sep='\t')


    # FOR THE DATASETS
    scores_df_datasets = pd.DataFrame(columns = ["dataset","count","mean","std","min","25pc","50pc","75pc","max"])
    for k,v in scores_datasets.items():
        description = list(pd.Series(v).describe())
        scores_df_datasets.loc[len(scores_df_datasets)] = [k] + description
    scores_df_datasets = scores_df_datasets.set_index('dataset')
    scores_df_datasets.to_csv(result_file_path+'_scores_datasets.tsv',sep='\t')


    # Write a normalized file
    # We normalize by TF only, but I keep the scores by dataset for information
    if outfilepath is None : outfilepath = result_file_path + "_normalized_by_tf.bed" # Default file path
    normalized_rf = open(outfilepath,'w')
    # Header
    normalized_rf.write('track name ='+cl_name+'_tf-normalized description="'+cl_name+' peaks with anomaly score - normalized by TF" useScore=1'+'\n')


    ## Re-open original result file RE-OPEN ORIGINAL RESULT FILE
    rf = open(result_file_path,'r')

    for line in rf:
        if (line[0:10] != 'track name'):
            line = line.split('\t') ; line[3] = line[3].split('.')
            tf = line[3][1] ; score = float(line[4])

            # Apply a "normalization" (does not mean the scores are normally distributed)
            new_score = (score - scores_df_tf.at[tf, 'mean']) / scores_df_tf.at[tf, 'std']

            # Center at 500.
            # I take 0.5 * new_score to reduce the dispersion a bit.
            new_score = 500 * (1 + 0.5 * new_score)


            if np.isnan(new_score) : new_score = 0 # Hotfix

            # FINALLY clip at 1000 to match BED format
            new_score = int(np.around(np.clip(new_score,0,1000)))

            # Rejoin line and write it
            line[4] = str(new_score)
            line[3] = '.'.join(line[3])
            line = '\t'.join(line)
            normalized_rf.write(line)
    normalized_rf.close()
    rf.close()

    return scores_df_tf, scores_df_datasets
