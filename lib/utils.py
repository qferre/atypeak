"""
# Put here functions for :
# Anomaly score
# Evaluations
# Diagnosis
"""

import sys

import numpy as np
import pandas as pd

#import pybedtools # Not sure it's needed ?
from plotnine import (ggplot, aes, geom_histogram, geom_violin,
                        scale_fill_grey, geom_boxplot, position_dodge, theme)

#from lib import utils # Circular import !
from lib import artificial_data

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.measure
import pandas as pd

#
# def add_noise(mat, frequencies, return_pos):
#     """
#     Adds peak noise (trace along X axis), frequencies i
#
#     mat : the 3D matrix to which we add the noise
#     frequencies : a 2d matrix of the axes Y and Z of the `mat` matrix. When picing a (Y,Z) for each noise peak, odds will be taken from this frequencies object.
#     return_pos : if True, return both the matrix with noise and the positions of the noisy tfs.
#
#     param : add ways to control peak generation
#     """
#
#     noise_nb = ss.poisson(1).rvs()
#     for _ in range(noise_nb): # Add some random peaks
#
#
#         r_center = int(ss.uniform(0,mat.shape[0]).rvs())
#         r_length = int(ss.lognorm(scale=mat.shape[0]/15,s=0.25).rvs())
#         #r_intensity = ss.norm(500,120).rvs() + ss.norm(10,10).rvs()
#         r_intensity = 1
#
#         # Pick TF and dataset from 2D weights
#
#
#         p = np.array([[0.9, 0.1, 0.3], [0.4, 0.5, 0.1], [0.5, 0.8, 0.5]])
#         frequencies = p
#
#
#
#
#         linear_idx = np.random.choice(frequencies.size, p=frequencies.ravel()/float(frequencies.sum()))
#         r_dataset, r_tf = np.unravel_index(linear_idx, frequencies.shape)
#
#         r_dataset, r_tf
#
#
#     # TODO ! THE WEIGHTS WILL BE DONE THROUGH, LIKE,  1000 RANDOM CRMS WITHOUT NOISE, LIKE THE USUAL
#
#
#
#
#     # Now turn this list of noise peaks into a matrix, and sum it to the original matrix
#
#     ar.list_of_peaks_to_matrix(noise_peaks)
#
#
#
#
#     # Maybe clip the new, summed matrix at its maximum of before or something. (to prevent one peak atop the other making it exceed previous values)
#
#     # Great ! now use this on the generator for data, and see if it improves matters
#
#



################################################################################
# ----------------------------- Matrix utilities ----------------------------- #
################################################################################


def zeropad(arr, pad_width):
    """Pad array with zeros.
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

# ### MaxPooling and Extending a 3d matrix along its first axis (the [0] axis)
# # Rq : those matrices are '4D' due to the channel dimension
# def squish(m, factor=10):
#     # 'Squishing' is done by MaxPooling across the 'region length' dimension.
#     if len(m.shape) != 4 : raise ValueError('squish() needs a 4D matrix (region_length, nb_datasets, nb_tf, channels=1)')
#     return skimage.measure.block_reduce(m, (factor,1,1,1), np.max)


# New version, slighlty faster, does not require skimage
def squish(m, factor=10, squishing_a_batch = False):
    # Calculate new shape based on desired reduction
    if not squishing_a_batch:
        new_shape = tuple([int(m.shape[0]/factor)]) + (m.shape[1:])
    else:
        # Same as squish, but will squish the SECOND dimension instead of the first
        # to work on 5D arrays (batch_size, region_length, nb_datasets, nb_tfs, 1)
        new_shape = tuple([m.shape[0]]) + tuple([int(m.shape[1]/factor)]) + (m.shape[2:])
    return bin_ndarray(m, new_shape=new_shape, operation='max')




def stretch(m, factor=10):
    # Kronecker product to up-sample.
    # The only differences with the original will be due to rounding (if the
    # peaks lengths were not divisible by the squishing factor).
    if len(m.shape) != 4 : raise ValueError('stretch() needs a 4D matrix (region_length, nb_datasets, nb_tf, channels=1)')
    return np.kron(m, np.ones((factor,1,1,1)))




"""
# TODO REMOVE SKIMAGE DEPENDENCY IN ENV IF I REPLACE IT !

l = k = 0

import time

m = np.arange(0,1280000,1).reshape((3200,20,20,1))

factor = 10

s = time.time()
for i in range(32):
    new_shape= tuple([int(m.shape[0]/factor)]) + (m.shape[1:])
    return bin_ndarray(m, new_shape=new_shape, operation='max')
e = time.time()
print(e-s)

s = time.time()
for i in range(32):
    l =skimage.measure.block_reduce(m, (factor,1,1,1), np.max)
e = time.time()
print(e-s)

np.sum(k == l)
"""




















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
    # NOTE : the color of each point could be made conditional by applying operations to `c`, if needed.
    c_raw_list = [mat[int(xr[i]),int(yr[i]),int(zr[i])] for i in range(len(xr))] # xr, yz and zr have the same nb of elements
    c_raw = np.array(c_raw_list)
    c = np.tile(c_raw[:,None], [1, 3]) # Fake RGB tiling







    # Scale
    if not standard_scale: c = (c-np.min(mat))/(np.max(mat)-np.min(mat))

    # Plot values as white (low) -> black (high)
    c = 1 - c # High values should be black, not white, and should be between 0 and 1

    # If required, plot values as red (low) -> blue (high)
    if rb_scale :
        c[:,2] = 1 - c[:,2]
        c[:,1] = 0








    #DEBUG : TF COLOR OVERRIDE FOR PAPER FIG
    if tf_colors_override is not None:
    #For display, use one color per TF, in the order given by a vector
    #tf_colors_override = [(255,255,0),(255,255,255),...]

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






"""

import numpy as np
m = np.zeros((100,10,3))
m[0:100,2,2] = 1
m[0:100,2,1] = 1

plot_3d_matrix(m,
    tf_colors_override = [(1,0,0),(0,1,0),(0,0,1)])

"""


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
        chrom = peak[4]

        pad_to = anomaly_matrix.shape[0] # Fix coordinates due to padding
        xmin_original = peak[5]
        xmin = (pad_to - crm_length) + (int(xmin_original) - crm_start)
        xmax_original = peak[6]
        xmax = (pad_to - crm_length) + (int(xmax_original) - crm_start)


        # REMARK Sometimes peaks were inside the CRM for less than 10 base pairs, and those will have been squished out by the, well, squishing.
        # In that case, the anomaly matrix may be an improper slice.
        # To fix this, we use nanmean and we set xmin to be at least 0, and xmax to be maximum the matrix length.
        # xmax must also be at least one higher than xmin.
        xmin = max(0,xmin)
        xmax = min(pad_to, xmax)
        if xmax == xmin : xmax = xmin+1


        cell_line = peak[0]
        tf = ordered_tfs[peak[2]]
        dataset = ordered_datasets[peak[3]]


        # Collect its anomaly vector
        anomaly_vector = anomaly_matrix[xmin:xmax,dataset,tf]
        # Maximum score along the peak, and double it given that in originl trial most scores were under 400
        anomaly_score = int(np.clip(np.nanmax(anomaly_vector) * 2000, 0, 1000))


        # TODO : INSTEAD OF MAXIMUM SCORE, MEAN WOULD BE BETTER ACCORDING TO ARTIFICIAL DATA. It sadly makes much more many peaks seen as noise. Try to find a compromise.


        if debug_print : print('Peak : ('+str(xmin)+' -> '+str(xmax)+','+str(tf)+','+str(dataset)+') -- Anomaly = '+str(anomaly_score))



        # TODO Add gregariousness (in the average CRM, the guide, how many non-zero elements are there in the same row ? the same column ?)
        # TODO Add other info : say how many peaks there were originally in this CRM

        # Produce the result lines in remap format :
        # chr    start   end    dataset.tf.cell_line    score   strand
        result = (str(peak[4])+'\t'+str(xmin_original)+'\t'+str(xmax_original)+'\t'+
            peak[3]+'.'+peak[2]+'.'+peak[0]+'\t'+ #peak[the dataset]+'.'+peak[the tf]+'.'+peak[the cell line]+'\t'+
            str(anomaly_score)+'\t'+'.')

        # TODO instead keep the MACS score and the peak center and add anomaly score + gregariousness in other columns
        # TODO scores in BED files must be between 0 and 1000. Be careful about that when normalizing.
        result_bed.append(result)


    return result_bed















#
# tmp = 'chr11	62545614	62545830	GSE59657.MYB.JURKAT	277.78711915016174'
# tmp = tmp.split('\t') # Split by tab
# tmp[3] = tmp[3].split('.') # Further split the dataset.tf.cell_line line
# tmp[3][1]
#
# tmp[3] = '.'.join(tmp[3])
# '\t'.join(tmp)

def normalize_result_file_score_by_tf(result_file_path, cl_name):

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
    # WE NORMALIZE BY TF ONLY ! I keep the scores by dataset only for information !
    normalized_rf = open(result_file_path+'_normalized_by_tf.bed','w')
    # Header
    normalized_rf.write('track name ='+cl_name+'_tf-normalized description="'+cl_name+' peaks with anomaly score - normalized by TF" useScore=1'+'\n')


    # RE-OPEN ORIGINAL RESULT FILE
    rf = open(result_file_path,'r')

    for line in rf:
        if (line[0:10] != 'track name'):
            line = line.split('\t') ; line[3] = line[3].split('.')
            tf = line[3][1] ; score = float(line[4])

            #new_score = (score - min_obs[tf]) / (max_obs[tf] - min_obs[tf]) * 1000
            # NO HARDCODE MIN OBS AT ZERO BECAUSE IF A TF IS ALWYS GOOD IT WOULD BE BAD THEN
            # FINALLY WRITE DEPENDING ON MEDIAN

            """
            The scores are not normally distributed, but I'm gonna use this anyways.
            """
            new_score = (score - scores_df_tf.at[tf, 'mean']) / scores_df_tf.at[tf, 'std']


            # Center at 500.
            # I take 0.5 * new_score to reduce the dispersion a bit.
            new_score = 500 * (1 + 0.5 * new_score)
            if np.isnan(new_score) : new_score = 0 # Hotfix
            new_score = int(np.clip(new_score,0,1000))

            #new_score = score / max_obs[tf] * 1000
            # print('----------')
            # print("TF : ",tf)
            # print("Old score : ",score)
            # print("Max score for this TF : ",max_obs[tf])
            # print("New score : ",new_score)



            # Rejoin line and write it
            line[4] = str(new_score)
            line[3] = '.'.join(line[3])
            line = '\t'.join(line)
            normalized_rf.write(line)
    normalized_rf.close()
    rf.close()





    return scores_df_tf, scores_df_datasets
