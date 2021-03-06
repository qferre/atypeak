import scipy.stats as ss
import numpy as np

import random

import lib.utils as utils

"""
Functions to generate artificial data for model evaluation.

In broad strokes, the data generated will place a stack of peaks (plus some noise as well as a constant watermark at 0,0)
This stack will be placed in "reliable_datasets" and by picking either one of two TF correlation groups.

The goal is to show that the model will usually rebuilt the correlation group as a whole, but nothing from the other group.
"""


def generator_fake(batch_size = 10, region_length = 160, #reliable_datasets=np.arange(16),
                    nb_datasets = 16, nb_tfs = 10, squish_factor = 10, ones_only = False,
                    watermark_prob = 0.75, tfgroup_split = 2/3, overlapping_groups = False, 
                    this_many_groups_of_4_tfs = None,
                    split_tfs_into_this_many_groups = None, crumb = None):
    """
    Generator object that calls the make_a_fake_matrix() function.

    Most arguments are self explanatory.
        - region_length, nb_datasets and nb_tfs give the matrix dimensions.
        - squish_factor indicate by how much the X axis will be scaled down
        - ones_only : if true, all scores will be 1000.

    To be passed to Keras' fit_generator()
    """
    while True:

        batch_features = list()
        
        for i in range(batch_size):
            X = make_a_fake_matrix(region_length, nb_datasets, nb_tfs,
                ones_only=ones_only, watermark_prob=watermark_prob,
                tfgroup_split=tfgroup_split, overlapping_groups=overlapping_groups,
                this_many_groups_of_4_tfs=this_many_groups_of_4_tfs,
                split_tfs_into_this_many_groups=split_tfs_into_this_many_groups)

            if crumb != None :
                # To counter sparsity, add crumbs if requested
                X = utils.add_crumbing(X, crumb)

            Xi = X[..., np.newaxis] # Add meaningless 'channel' dimension
            batch_features.append(Xi)

        result = np.array(batch_features)

        target = list()
        for i in range(len(result)) : target.append(result[i]) # The target is the data itself
        target = np.array(target)

        # Squishing along the X axis to reduce computing cost
        result = [utils.squish(m, squish_factor) for m in result]
        target = [utils.squish(m, squish_factor) for m in target]
        result = np.array(result)
        target = np.array(target)

        yield (result,target)




def list_of_peaks_to_matrix(peaks, region_length, nb_datasets, nb_tfs, ones_only = False):
    """
    Utility function designed for make_a_fake_matrix(). It will convert a list of peak objects, which
    are tuples of (dataset, tf, center, length, intensity), into a 3D tensor.
    """
    # TODO This is more or less the same code than for real data, but real data
    # does not use this function. Recode so that it does and share the same code.

    mat = np.zeros((region_length,nb_datasets,nb_tfs))

    # Now write the peaks to the matrix
    for peak in peaks:
        dataset, tf, center, length, intensity = peak # Read peak

        # Truncate if the generated peak falls outside of the region
        begin = max(0, center-length)
        end = min(region_length, center+length)

        ## Write to matrix
        # Intensity is clipped at 0-1000 and divided by 1000
        # This is done because input peaks are assumed to be in BED format, whose intensity caps at 1000
        intensity_clipped = np.clip(intensity, 0, 1000) / 1000

        # Write peak
        mat[begin:end,dataset,tf] = mat[begin:end,dataset,tf] + intensity_clipped

        # Override : if ones_only, we overwrite the current value with 1
        if ones_only : mat[begin:end,dataset,tf] = 1

    return mat


def make_a_fake_matrix(region_length, nb_datasets, nb_tfs,
                        reliable_datasets = None, signal = True, noise = True,
                        ones_only = False, watermark_prob = 1,
                        tfgroup_split = 2/3, overlapping_groups = False,
                        return_separately = False,
                        this_many_groups_of_4_tfs = None,
                        split_tfs_into_this_many_groups = None):
    """
    region_length,nb_datasets,nb_tfs : matrix size
    reliable_datasets : which datasets are reliable
    signal, noise : which kinds of peaks to genrate
    ones_only : with score = 1000, or random

    tfgroup_split : odds of using one tf group or the other
    overlapping_groups: whether the groups should be {A, B} or {A, AB}
    this_many_groups_of_4_tfs: insteaf oa making 2 groups of TFs, will make this many groups of 4 TFs. Override previous two.
    split_tfs_into_this_many_groups : instead of making 2 groups of TFs, will make this many groups. Overrives previous three

    return_separately : if True, will stop and return a separate list of lists of peaks (resp. stack, noise, watermark)
    To be processed in evaluation - this will NOT return a tensor.
    """

    # TODO Do not hardcode the parameters of Poisson and other laws

    peaks_stack = list()
    peaks_watermark = list()
    peaks_noise = list()

    ###### DATASETS STEP
    # Choose datasets that will correlate

    # Using only "reliable datasets" this time (ie the first half, for example)

    if reliable_datasets == None :
        reliable_datasets = range(int(nb_datasets/2),nb_datasets) # Default reliable datasets are the last 50%

    k = ss.poisson(1).rvs() + 1
    datasets = np.random.choice(reliable_datasets,k)
    # TODO Implement weighted choices here 

    # Pick common center for the stack
    common_center = int(ss.uniform(0,region_length).rvs())

    # In some cases you may not want to write true peaks and produce only noise, you should set signal to false in this case
    if signal :

        # We want to pick from either the first half of tfs or the second half.
        # The idea is that, when rebuilding, if the peak is from a top half TFs, the phantom should
        # come only from a top half tf since they are the ones correlating.
        tf_first_half = range(int(nb_tfs/2))
        tf_second_half = range(int(nb_tfs/2),nb_tfs)




  







        # Alternative 2 : If desired, pick not from first half OR second half
        # but instead pick from first+second OR First only
        if overlapping_groups:
            tf_first_half = range(nb_tfs)
            tf_second_half = range(int(nb_tfs/2))
        else:
            tf_first_half = range(int(nb_tfs/2))
            tf_second_half = range(int(nb_tfs/2),nb_tfs)


        # Choosing probability is a parameter (tfgroup_split)
        cointoss = random.uniform(0,1)
        cointoss = cointoss < tfgroup_split

        if cointoss : tfs_to_choose_from = tf_first_half
        else : tfs_to_choose_from = tf_second_half




        # Override 1 : instead of splitting in 2, we can split in groups of 4 Tfs
        # and keep how many we want.
        if this_many_groups_of_4_tfs != None:
            tfs = list(range(nb_tfs))
            tf_groups = [tfs[i:i+4] for i in range(0, len(tfs), 4)]

            # Now pick uniformly randomly one group among them, using only the first N ones as ordered
            tfs_to_choose_from = random.choice(tf_groups[0:this_many_groups_of_4_tfs])
        


        # Override 2 : split the tfs into N groups and pick uniformly randomly 
        # one group among them. It overrides the previous overrides.
        if split_tfs_into_this_many_groups != None:
            tfs = list(range(nb_tfs))
            tf_groups = np.array_split(tfs, split_tfs_into_this_many_groups)
            tfs_to_choose_from = random.choice(tf_groups)





        ### WRITING PEAKS 

        # Pick number of peaks to write
        N = ss.poisson(1).rvs() + 1
        for _ in range(N):
            # We have N peaks correlating, now split them among TFs

            ## TF steps
            kp = ss.poisson(1).rvs() +1
            tfs = np.random.choice(tfs_to_choose_from,kp, replace=True)

            for tf in tfs:
                center_for_this_tf = int(common_center + ss.uniform(-200,200).rvs())
                length = int(ss.lognorm(scale=250,s=0.25).rvs())
                intensity =  ss.norm(500,120).rvs() + ss.norm(10,10).rvs() # intensity = random + noise
                # NOTE Intensity is dicarded by default for now due to ones_only

                # From analysis, peaks FOR THE SAME TF across different datasets should be very similar.
                for dataset in datasets :
                    peaks_stack.append((dataset, tf, center_for_this_tf, length, intensity))



    ## NOISE STEP
    # TODO In some cases you may not want noise (pre-training ?)
    if noise :

        # Drop some peaks from the stack at random
        p = 0.25
        for peak in peaks_stack[:] :
            if ss.uniform.rvs() < p : peaks_stack.remove(peak)

        # Add some random peaks
        F = ss.poisson(1).rvs()
        for _ in range(F):
            # Place a completely random peak
            r_dataset = np.random.choice(range(nb_datasets))
            r_tf = np.random.choice(range(nb_tfs))
            r_center = int(ss.uniform(0,region_length).rvs())
            r_length = int(ss.lognorm(scale=250,s=0.25).rvs())
            r_intensity = ss.norm(500,120).rvs() + ss.norm(10,10).rvs()

            # Small detail : disallow placing noise in the watermark at 0,0
            if (r_dataset == 0) and (r_tf == 0) : r_dataset = 1 ; r_tf = 1

            peaks_noise.append((r_dataset, r_tf, r_center, r_length, r_intensity))


    # CONTROL BAR : have one of the datasets/TF pair have a peak on all the length
    # of the region. It should get in most cases its own group or a slight correlation.
    # To prevent learning traps, don't make it always appear.
    if ss.uniform.rvs() < watermark_prob:
        peaks_watermark.append((0,0,int(region_length/2),region_length-2,1000))

    # If we want to return separately (useful in evaluation, when computing scores per category of peak)
    if return_separately:
        result = [peaks_stack, peaks_noise, peaks_watermark]
        return result

    # In any other cases, treat them in bulk.
    peaks = peaks_stack + peaks_noise + peaks_watermark

    # Convert list of peaks to tensor
    result = list_of_peaks_to_matrix(peaks, region_length, nb_datasets, nb_tfs, ones_only = ones_only)

    # Clip result !
    result = np.clip(result,0,1)

    return result
    # NOTE Those matrices are quite denser (less sparse) than typical real ones, may need to revise down number of parameters.
    # The fact that they are denser and of smaller dimensions is also why I use no crumbs for now.
