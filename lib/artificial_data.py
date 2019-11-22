import scipy.sparse as sp
import scipy.stats as ss
import numpy as np

import random



from lib import utils
from lib import convpeakdenoise as cr

"""
Functions to generate artificial data for model evaluation.

In broad strokes, the data generated will place a stack of peaks (plus some noise as well as a constant watermark at 0,0)

This stack will be places in "reliable_datasets" and by picking either one of two TF correlation groups.

The goal is to show that the model will usually rebuilt the correlation group as a whole, but nothing from the other group.
"""













def generator_fake(batch_size = 10,region_length = 160, #reliable_datasets=np.arange(16)
                    nb_datasets = 16, nb_tfs = 10, squish_factor = 10, ones_only = False,
                    watermark_prob = 0.75, tfgroup_split = 2/3, overlapping_groups = False, crumb = None):
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
        batch_status = list()
        for i in range(batch_size):
            #X = make_a_fake_matrix(region_length,nb_datasets,nb_tfs,reliable_datasets)
            X = make_a_fake_matrix(region_length, nb_datasets, nb_tfs, ones_only=ones_only, watermark_prob=watermark_prob, tfgroup_split=tfgroup_split, overlapping_groups=overlapping_groups)

            if crumb != None :
                # To counter sparsity, add crumbs (see function documentation)
                X = cr.look_here_stupid(X, crumb)

            Xi = X[..., np.newaxis] # Add meaningless 'channel' dimension
            batch_features.append(Xi)
            batch_status.append('data')



        """
        # Sometimes, give a matrix of only noise and tell it to rebuild it as "full zero"
        # if rand < p :
        #     noisy_shape = dense_matrix.shape
        #     noisy = make_a_fake_matrix(region_length,nb_datasets,nb_tfs,signal = False)
        #     batch_features.append('noisy')
        #     batch_status.append('noise')
        """


        result = np.array(batch_features)

        target = list()
        for i in range(len(result)) :
            if batch_status[i] == 'data' : target.append(result[i]) # The target is the data
            #if batch_status[i] == 'noisy' : target.append(np.zeros(result[i].shape)) # This was just noise. The target should be an empty matrix

        target = np.array(target)




        # Squishing along the X axis to reduce computing cost

        result = [utils.squish(m, squish_factor) for m in result]
        target = [utils.squish(m, squish_factor) for m in target]
        result = np.array(result)
        target = np.array(target)

        """
        result = utils.squish(np.array(result), squish_factor, squishing_a_batch = True)
        target = utils.squish(np.array(target), squish_factor, squishing_a_batch = True)
        """

        yield (result,target)















def list_of_peaks_to_matrix(peaks,region_length,nb_datasets,nb_tfs, crumb=False, ones_only = False):
    """
    Utility function designed for make_a_fake_matrix(). It will convert a list of peak objects, which
    are tuples of (dataset, tf, center, length, intensity), into a 3D tensor.
    """

    mat = np.zeros((region_length,nb_datasets,nb_tfs))
    # TODO make this a sparse matrix ? is that really necessary ? If not remove scipy.sparse import

    # Now write the peaks to the matrix
    for peak in peaks:
        dataset, tf, center, length, intensity = peak # Read peak

        # Truncate if the generated peak falls outside of the region
        begin = max(0, center-length)
        end = min(region_length, center+length)

        ## Write to matrix
        # WARNING intensity must be between 0 and 1 !
        # Let's clip at 0-1000 and divide by 1000
        intensity_clipped = np.clip(intensity,0,1000) / 1000



        # Write peak
        mat[begin:end,dataset,tf] = mat[begin:end,dataset,tf] + intensity_clipped

        # Add crumbs
        if crumb :
            mat[begin:end,:,tf] = mat[begin:end,:,tf] + 0.1*intensity_clipped
            mat[begin:end,dataset,:] = mat[begin:end,dataset,:] + 0.1*intensity_clipped


        # Override : if ones_only, we overwrite the current value with 1
        if ones_only : mat[begin:end,dataset,tf] = 1



    """
    # Debug : transpose the TF and dataset axis. I used this to test a theory
    # about whether there is a precision biais towards one or tht other
    # TODO MAKE THIS A PARAMETER LIKE THE REST !!!!
    if debug_transpose :
        mat = np.transpose(mat,(0,2,1))
    """



    return mat




















def make_a_fake_matrix(region_length,nb_datasets,nb_tfs, reliable_datasets = None,
                        signal = True, noise=True,
                        ones_only = False, watermark_prob = 1, tfgroup_split = 2/3, overlapping_groups = False,
                        return_separately = False):
    """

    region_length,nb_datasets,nb_tfs : matrix size
    reliable_datasets : which datasets are reliable
    signal, noise : which kinds of peaks to genrate
    ones_only : with score = 1000, or random

    return_separately : if True, will stop and return a separate list of lists of peaks (resp. stack, noise, watermark)
    to be processed in evaluation - this will NOT return a tensor.

    """

    # TODO Do not hardcode the parameters of Poisson and other laws

    peaks_stack = list()
    peaks_watermark = list()
    peaks_noise = list()

    ###### DATASETS STEP
    # Choose datasets that will correlate

    # TODO : remove the +1 for k and treat the k=0 case by skipping all of this until noise
    # same for N

    # Using only "reliable datasets" this time (ie the first half, for example)

    if reliable_datasets == None :
        reliable_datasets = range(int(nb_datasets/2),nb_datasets) # Default reliable datasets are the last 50%

    k = ss.poisson(1).rvs() + 1
    datasets = np.random.choice(reliable_datasets,k)
    # TODO : not all datasets correlate ! assign different probas ? Or simply draw from reliable_datasets
    # datasets = np.random.choice(reliable_datasets,k)

    common_center = int(ss.uniform(0,region_length).rvs())

    # In some cases you may not want to write true peaks and produce only noise, you should set signal to false in this case
    if signal :


        # Checkup : we want to pick from either the first half of tfs or the second half.
        # The idea is that, when rebuilding, if the peak is from a top half TFs, the phantom should
        # come only from a top half tf since they are the ones correlating.
        tf_first_half = range(int(nb_tfs/2))
        tf_second_half = range(int(nb_tfs/2),nb_tfs)

















































        """
        ANOTHER THING TO TEST A THEORY : OVERLAPPING groups
        Pick not from first half OR second half. Instead pick from first+second OR First only.
        Will help test my theory aout corr group size and overlapping groups
        AGAIN TODO MAKE THIS A PARAMETER.
        LEAVE IT OFF BY DEFAULT
        """
        #overlapping_groups = False # SHOULD BE FALSE BY DEFAULT, AND NOW IT IS OKAY
        if overlapping_groups:
            tf_first_half = range(nb_tfs)
            tf_second_half = range(int(nb_tfs/2))
        else:
            tf_first_half = range(int(nb_tfs/2))
            tf_second_half = range(int(nb_tfs/2),nb_tfs)







        """
        # Now, when we pick the correlating tfs, pick them from either the first half of the second half (50% chance)
        # Now 2/3 - 1/3 chance to test a theory
        """
        #tfgroup_split = 2/3
        # OKAY I PUT IT AT 2/3 BY DEFAULT
        #tfgroup_split = 1/2
        """
        # TODO JUST MAKE THIS A PARAMETER !!!!!!!!!
        # COPY ALL CASES WHERE I ADDED WATEMARK_PROB !!!!!!
        """









        cointoss = random.uniform(0,1)
        cointoss = cointoss < tfgroup_split

        if cointoss : tfs_to_choose_from = tf_first_half
        else : tfs_to_choose_from = tf_second_half


















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
                intensity =  ss.norm(500,120).rvs() + ss.norm(10,10).rvs() # intenity = random + noise
                # NOTE intensity is dicarded by default for now due to ones_only

                # From analysis, peaks FOR THE SAME TF across different datasets should be very similar.
                for dataset in datasets :
                    """
                    # TODO Maybe instead we could pick here the 1+ datasets which have this TF.
                    # This way the TFs would not be replicated across always the same datasets.
                    """
                    peaks_stack.append((dataset, tf, center_for_this_tf, length, intensity))



    """
    # In some cases you may not want noise (ie. for pre-training)
    # TODO Split stack and nosie into their own functions
    """


    ## NOISE STEP
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
    # of the region. The idea is that it will be very frequent but not particularly
    # correlated with anything. Given how MSE works, it will still be rebuilt, and
    # always (likely due to having its own filters).

    # To prevent it form being too trivial, and from falling into a learning trap of
    # rebuilding the watermak only when too many dimensions, don't make it always
    # appear.
    if ss.uniform.rvs() < watermark_prob:
        peaks_watermark.append((0,0,int(region_length/2),region_length-2,1000))





    # If we want to return separately (useful in evaluation, when computing scores per category of peak)
    if return_separately:
        result = [peaks_stack, peaks_noise, peaks_watermark]
        return result

    # In any other cases, treat them in bulk.
    peaks = peaks_stack + peaks_noise + peaks_watermark





    # TODO NOTE : For now artificial data has no crumbs. Must allow it to be changed here. Less useful in artificial because the dimensions and
    # sparsity are usually smaller in proof-of-concetps.

    # Convert list of peaks to tensor
    result = list_of_peaks_to_matrix(peaks,region_length,nb_datasets,nb_tfs, crumb = False, ones_only = ones_only)

    # Clip result !
    result = np.clip(result,0,1)

    return result
    # TODO : make it possible to return (matrix) or (matrix, matrix_without_noise)

    # TODO Those matrices are too dense need to revise down number of parameters
