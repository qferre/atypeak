"""
Functions to compute scores, produce result, and evaluate the model.

These functions are used both by the model (notably anomaly score calculation) and the diagnostic functions.
"""

import os, sys, time, functools, itertools

import pybedtools
import numpy as np
import pandas as pd
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt

from plotnine import ggplot, aes, geom_histogram, scale_fill_grey, geom_violin, geom_boxplot, position_dodge, theme, xlab, ylab

import lib.artificial_data as ad
import lib.utils as utils

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


"""
Careful not to import Keras or Tensorflow here, otherwise the subprocess in file production will not work.
"""


################################################################################
# ----------------- Compute anomaly score and produce results ---------------- #
################################################################################


def anomaly(before, prediction):
    """
    Anomaly score between a true (`before`) and predicted (`prediction`) CRMs.
    Both argument should be 3d NumPy arrays.

    WARNING : May be a misnomer, as a score of 1 means "good", not "anomalous !"

    Note : this is all designed to work with scores, but for now the peaks all have scores of 1 when present.
    """

    # TODO Return the un-crumbed matrix for use as a mask ? Not needed if I don't visualize it, since I query the values based on the original peaks coordinates
    # Here is a temporary hotfix that removes all values below 0.1
    # It will stop working when we have values besides real peak values below 0.1 (or higher than 1) !!
    mask = np.clip(np.around(before, decimals = 1),0,1)
    mask[ mask !=0 ] = 1 # This is 0 where there is no peak, and 1 where there is one


    ## Anomaly score computation
    anomaly = 1 - ((before+1E-100) - prediction)/(before+1E-100)
    # WARNING this considers higher plotting as an anomaly ! We want mostly
    # Divide by before, otherwise when peaks with low scores are discarded. Not useful now since before is always 1, but future-proofing.

    anomaly = np.clip(anomaly, 1E-5, np.inf) # Do not clip at 0 or it will not be plotted

    return anomaly * mask
















def produce_result_for_matrix(m,
    model, get_matrix_method, parameters, datasets_clean, cl_tfs):
    """
    Wrapper to produce a result for one matrix. Used in the main loop.
    """

    # Collect original matrix and lines
    current_matrix, origin_data_file, crm_start = get_matrix_method(m, return_data_lines = True)

    # Pad the matrix (much like we do in model_atypeak.generator_unsparse)
    crm_length = current_matrix.shape[0]
    current_matrix_padded = np.pad(current_matrix, pad_width = ((parameters["pad_to"] - crm_length,0),(0,0),(0,0)), mode='constant', constant_values=0)

    squished_matrix = utils.squish(current_matrix_padded[:,:,:,np.newaxis])

    ## Compute result and anomaly 
    prediction = model.predict(squished_matrix[np.newaxis,:,:,:,:])
    result_matrix = utils.stretch(prediction[0,:,:,:,:])
    # This uses directly the anomaly function defined in this file
    anomaly_matrix = anomaly(current_matrix_padded, result_matrix[:,:,:,0])

    # Now write to the result BED the current peaks with their newfound anomaly score
    result = utils.produce_result_bed(origin_data_file, anomaly_matrix,
                                        datasets_clean, cl_tfs,
                                        crm_start, crm_length,
                                        debug_print = False)
    
    # WARNING : This is 1 - anomaly, so the score is actually high for good peaks !! TODO Emphasize it more
    return result




class MonitoredProcessPoolExecutor(ProcessPoolExecutor):
    """
    A class to encapsulate a ProcessPoolExecutor but monitoring the current number of active workers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running_workers = 0

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        self._running_workers += 1
        future.add_done_callback(self._worker_is_done)
        return future

    def _worker_is_done(self, future):
        self._running_workers -= 1

    def get_pool_usage(self):
        return self._running_workers




def result_file_worker(minibatch, save_model_path,
    get_matrix_method, parameters, datasets_clean, cl_tfs,
    model_prepare_function,result_queue):


    # Please don't send me console outputs
    import os, sys
    f = open(os.devnull, 'w')
    from contextlib import redirect_stdout, redirect_stderr

    with redirect_stdout(f):
        with redirect_stderr(f):

            # Create keras session here for the worker
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # No TF logs unless errors

            # NOTE KMP_AFFINITY can lock all processes to the same core.
            # We must disable it to use more than one CPU.
            os.environ["KMP_AFFINITY"] = "none"

            import keras.backend as K
            import tensorflow as tf

            model = model_prepare_function(parameters, len(datasets_clean), len(cl_tfs))
            model.load_weights(save_model_path+".h5") # Do not forget to add the ".h5" !! 

    my_produce_result_function_for_matrix = functools.partial(produce_result_for_matrix, model=model,
        get_matrix_method=get_matrix_method,
        parameters=parameters, datasets_clean=datasets_clean, cl_tfs=cl_tfs)

    
    for m in minibatch:
        my_result = my_produce_result_function_for_matrix(m)
        result_queue.put(my_result)
    


def produce_result_file(all_matrices, output_path, #model,
    get_matrix_method, parameters, model_prepare_function,
    datasets_clean, cl_tfs,
    add_track_header = True,
    nb_processes = 1,
    save_model_path = None):
    """
    Will take all CRM whose ID is in all_matrices, rebuild them with the
    provided model, then compute the anomaly score and write the result in a BED
    file at output_path.
    """

    # Open result file in append mode
    rf = open(output_path,'w')
    if add_track_header : rf.write('track name ='+parameters['cell_line']+' description="'+parameters['cell_line']+' peaks with anomaly score" useScore=1'+'\n')


    ## Main loop iterating over all matrices
    result = list()

    print('Beginning processing...')
  
    if nb_processes < 2:

        print("Mono threaded")

        # Create keras session here for the worker
        from keras.models import load_model
        import keras.backend as K
        import tensorflow as tf
        import keras

        import lib.model_atypeak as cp

        model = model_prepare_function(parameters, len(datasets_clean), len(cl_tfs))
        model.load_weights(save_model_path+".h5") # Do not forget to add the ".h5" !! 

        my_produce_result_function_for_matrix = functools.partial(produce_result_for_matrix,
            model=model, get_matrix_method=get_matrix_method, parameters=parameters, datasets_clean=datasets_clean, cl_tfs=cl_tfs)

        result = list()

        cnt = 1
        for m in all_matrices:

            result += my_produce_result_function_for_matrix(m)

            # Progress display
            sys.stdout.write("\r" +"Processed CRM nb. : " + str(cnt)+" / "+str(len(all_matrices)))
            sys.stdout.flush()
            cnt += 1
    

        # Write the final file
        for line in result: rf.write(line+'\n')
        rf.close()
        print(" -- Done.")


    if nb_processes >= 2:
        
        print("Multi threaded")

        mana = multiprocessing.Manager()
        result_queue = mana.Queue()
        pool = MonitoredProcessPoolExecutor(nb_processes)

        # Split into as many batches as there are child processes 
        minibatches = np.array_split(np.array(all_matrices), nb_processes)

        args = {
            "save_model_path":save_model_path,
            "get_matrix_method":get_matrix_method,
            "parameters":parameters, 
            "datasets_clean":datasets_clean,
            "cl_tfs":cl_tfs,
            "model_prepare_function":model_prepare_function,
            "result_queue":result_queue }

        # Submit to the pool of processes
        for i in range(nb_processes):
            pool.submit(result_file_worker, minibatch = minibatches[i], **args)
            print("Submitted job.")

       
        # Control print while there are still active workers
        while pool.get_pool_usage() > 0:
            #print(pool.get_pool_usage())
            time.sleep(0.05) # Prevent a flurry of messages
            cnt = result_queue.qsize()
            sys.stdout.write("\r" +"Processed CRMs currently : "+str(cnt)+" / "+str(len(all_matrices)))
            sys.stdout.flush()

        # Empty the queue whenever possible
        while True:
            if not result_queue.empty():
                partial_result = result_queue.get()
                for line in partial_result: rf.write(line+'\n')
            else:
                print(" -- Done")
                rf.close()
                break
        del result_queue





################################################################################
# -------------- Corr group and biais correction/normalization --------------- #
################################################################################

def estimate_corr_group_normalization_factors(model, all_datasets, all_tfs, 
    list_of_many_crms, outfilepath,
    crm_length = 3200, squish_factor = 10,
    use_crumbing = True):
    """
    Estimates the corr group for each dataset+tf pair by looking at the added phantoms

    all_datasets and all_tfs are lists of names (eg. ['d01','d02'],['FOS','JUN'])

    list_of_many_crms is a list of matrices given by er.get_some_crms()

    We correct for intra-group biais (more abundant peak within a group get higher scores)
    and inter-group biais (peaks get a perfect score of 1 when the group is completely full, not when it is as full as usual)
    See code comments for details

    Returns a dictionary of scaling factors to be applied compensating for model biais
    """

    coefs_norm = dict()

    # All possible combinations
    all_combis = list(itertools.product(all_datasets, all_tfs))

    # Take the average CRM 2d and repeat it across X axis
    average_crm = np.mean(np.array(list_of_many_crms), axis=0)
    average_crm_2d = np.mean(average_crm, axis = 0)


    # Open output file
    logcombifile = open(outfilepath,'w')
    header="dataset\ttf\tintra_group_weight\tinter_group_weight\toverlapping_group_weight\tfinal_k\n" # TODO Only at beginning OFC
    logcombifile.write(header)

    
    # ------------------ PREPARING SOME PREDICTIONS

    ###### FULL CRM
    # Full CRM where there is a peak everywhere there can be
    full_crm_2d = (average_crm_2d>0).astype(int) 
    full_crm_3d = np.stack([full_crm_2d]*int(crm_length/squish_factor), axis=0).astype('float64')
    #full_crm_3d = np.stack([[0*full_crm_2d]*210 + [full_crm_2d]*10 + [0*full_crm_2d]*100], axis=0).astype('float64')

    if use_crumbing: full_crm_3d = utils.add_crumbing(full_crm_3d)
    #if use_crumbing: full_crm_3d = utils.add_crumbing(full_crm_3d[0,...])

    before_2df = np.mean(full_crm_3d, axis=0)



    BEFORE_VALUE = 1 # Careful about affine effect. TODO Write somewhere.



    ###### BEFORE = THIS SOURCE ONLY

    def pred_for_this_source_only_internal(curr_dataset_id, curr_tf_id):
        # Put a peak only for this tf+dataset combi (ie. this source) and look at the phantoms
        # Here, we consider the correlation group itself
        # This will be used in two steps
        x = np.zeros((crm_length,len(all_datasets),len(all_tfs))) # Create an empty CRM

        # Add a major peak for this particular dataset+tf pair across the entire region
        # Use a value of 10 or 100 to force MSE to show groups ??
        # No use 1 instead to prevent affine pbs
        x[:,curr_dataset_id, curr_tf_id] = BEFORE_VALUE
        if use_crumbing: x = utils.add_crumbing(x) # Add crumbing
        before_2d_p = np.mean(x, axis=0)
            
        xp = utils.squish(x, factor = squish_factor)
        xp2 = xp[np.newaxis, ..., np.newaxis]
        prediction = model.predict(xp2)[0,:,:,:,0]
        prediction_2d = np.mean(prediction, axis=0) # 2D - mean along region axis # VERY IMPORTANT THAT HERE IT IS THE MEAN. TODO SAY SO IN PAPER !!!
        return before_2d_p, prediction_2d


    # WARNING 
    # IT IS CRITICAL THAT average_crm_2d BE NOT CRUMBED FOR LATER CALCULATIONS (ie inferring a full crm from the first weight)
    # NOTE I MUST ADD NOTES where I generate the list_of_many_CRMS to NOT ADD CRUMBING THERE !!!



    # Preparing intra weight mask; harmonize with rest of code !
    # See explanations for this code below. It is done here because we need to have all intra_weights
    # before we calculate the next weight
    intra_weight_mask = np.zeros((len(all_datasets),len(all_tfs)))
    for combi in all_combis:
        curr_dataset, curr_tf = combi
        curr_dataset_id, curr_tf_id = all_datasets.index(curr_dataset), all_tfs.index(curr_tf)
        before_2d_p, prediction_2d = pred_for_this_source_only_internal(curr_dataset_id, curr_tf_id)
        
        # Only nonzero terms that correspond to a real peak : discard negative influences, and crumbing
        #before_sum = np.sum((before_2d_p * full_crm_2d))
        #before_sum = before_2d_p[curr_dataset_id, curr_tf_id] # HOTFIX use this only due to crumbing making sums different if we use the line above
        #prediction_sum = np.sum((prediction_2d * full_crm_2d)[prediction_2d>0])
        # never mind, it didn't help, was not justifiable and could cause problems of its own
        iw = np.sum(before_2d_p)/np.sum(prediction_2d)
        #iw = before_sum/prediction_sum
        

        intra_weight_mask[curr_dataset_id, curr_tf_id] = iw




    for combi in all_combis:

        # Combi information
        curr_dataset = combi[0] # get id in list of the dataset
        curr_tf = combi[1] # get id in list of the tf
        # Convert dataset and tf to the approriate coordinates
        curr_dataset_id = all_datasets.index(curr_dataset)
        curr_tf_id = all_tfs.index(curr_tf)




        # Get the phantoms when trying to predict a CRM with this source only !
        before_2d_p, prediction_2d = pred_for_this_source_only_internal(curr_dataset_id, curr_tf_id)



        # Prepare a mask that contains more or less all peaks that are not in the current correlation group
        value_in_real_full = before_2df[curr_dataset_id,curr_tf_id]
        normalized_pred = prediction_2d/prediction_2d[curr_dataset_id,curr_tf_id]
        # TODO REMEMBER TO WRITE ALSO IN PAPER : the second term represents the value IT WOULD HAVE IN A FULL CRM
        others_mask = normalized_pred * value_in_real_full
        before_others = np.clip(before_2df - others_mask, 0, 2) # Floor at 0 and cap at 2 to prevent biases in others_mask and
        
        

        

        # ------------------ COMPUTING WEIGHTS




        # --- Intra group bias

        # See the ratio between sum of prediction and sum of before
        # If there is an intra group bias in favor of the source, it will be higher (nonwithstanding group size, later on)
        # This has been calculated above, use it
        intra_weight =  intra_weight_mask[curr_dataset_id, curr_tf_id]


        # ---- Inter group bias

        # Groups are not always complete in the same proportion : ABC might have usually 2 out of 3 members, while ABCDEF migt have usually 2 out of 6 members
        # Ie. completeness bias
        # NOTE KEEP THIS
        # Correlation groups reach a rebuilt value of 1 on each peak (hmm after normalizing intra group BE CAREFUL NOTE THIS SOMEWHERE ELSE) when complete, but we
        # need to normalize this by "how complete are they typically"

        # Try to estimate the "request", ie. the group this source belongs to, by looking at
        # the added phantoms, which are the peaks that were expected
        request = prediction_2d # TODO NOTE IN PAPER METHODS AND HERE THIS IS IMPERFECT AND FALSE (by which I mean not rigorous) BUT A GOOD ESTIMATION OF THE REQUEST

        clipped_requested_group = np.clip(request,0,np.inf) # Although it discards negative influences, we need to clip the request to calculate usual completeness

        # Now, we need to determine how complete that group usually is, HOW MUCH THE REQUEST IS USUALLY FULFILLED
        # The idea being that having 1 peak in the group when there is usually only 2 on average is not 
        # the same as having one where there is usually at least 5.

        occupancy = []
        negative_occupancy = []

        mask = clipped_requested_group/np.max(clipped_requested_group) #*16 # so we have full-term-equivalents, full term meaning "the most requested source"
        # TODO I removed the dividing by max, try again with it if there are problems


        for before in list_of_many_crms:
            before_2d = np.max(before, axis = 0) # CRITICAL TO USE MAX TO GET LOCAL GROUPS !? risk is to "superimpose" local CRMs by flattening across. But we can take that risk in an estimation.
            
            
            # Only compute occupancy if the peak (the source) being considered is present (!!) so we see the correlators of the source itself
            # Explain in Bayesian terms
            if before_2d[curr_dataset_id, curr_tf_id] != 0:
                
                occupancy += [np.sum(mask*before_2d*intra_weight_mask)]
                # TODO perhaps I should apply the first weight here ? Or since I apply it for nobody it compensates itself ? Better argument is that I compare the whole group here, not the individuals

                # TODO add here computation of negative occupancy for step 3, again in Bayesian to indeed get their combis and not all combis
                # I could try it with others_mask but I think 
                negative_occupancy += [np.sum(others_mask*before_2d*intra_weight_mask)]


            else: 
                occupancy += [0]
                negative_occupancy += [0]

                

        max_occupancy_possible = np.sum(mask*full_crm_2d*intra_weight_mask)    
        occupancy = np.array(occupancy)
        mean_occupancy = np.mean(occupancy[occupancy.nonzero()]) # Remove all zeroes occupancy (bayesian whatever)
       

        max_negative_occupancy_possible = np.sum(others_mask*full_crm_2d*intra_weight_mask) 
        negative_occupancy = np.array(negative_occupancy)
        mean_negative_occupancy = np.mean(negative_occupancy[negative_occupancy.nonzero()]) 
        
        # NOTE : Due to negative weights mean/max occupancy can be higher than 1 !!?? SAY SO IN METHODS PAPER

        # The final second weight is, 1/usual proportinal occupancy
        inter_weight = max_occupancy_possible/mean_occupancy





        # ----- Overlapping groups

        # Overlapping groups can result in a higher score by combining several influences
        # Such groups are not always seen when looking at phantoms for individual source

        # An estimate is such : 
        # We retake before_other that contains more or less all peaks that are
        # not in the current correlation group

        full_crm_3d_others = np.stack([before_others]*int(crm_length/squish_factor), axis=0).astype('float64')
        #if use_crumbing: full_crm_3d = utils.add_crumbing(full_crm_3d) # DONT READD CRUMBING FOR THIS PART !! IT'S ALREADY ADDED
        predictionf_others = model.predict(full_crm_3d_others[np.newaxis,...,np.newaxis])[0,:,:,:,0]
        prediction_2df_others = np.mean(predictionf_others, axis=0) # 2D - mean along region axis
        
        # Now, when there are all peaks not in the corr group and only those, how much
        # of a phantom is present for this source ?
        others_contibution = prediction_2df_others[curr_dataset_id,curr_tf_id]

        # How much do those others peaks usually contribute to the group ? 
        # So calculate how much in a average scenario, they are expected to provide
        strangers_weight = mean_negative_occupancy / max_negative_occupancy_possible
        overlapping_weight = 1 - ((others_contibution/value_in_real_full) * strangers_weight)


        # TODO This is an ersatz, we need to use Monte Carlo similar to occupancy estimation in te inter-group bias phase : apply intra and inter group corrections, and then look at phantoms. Probably.

        # NOTE General assumption FOR ALL OF THIS MOVE ELSEWHER AND ADD TO PAPER : Assuming negative effects are negligible and that in biology more is always better


        # Finally, combine the weights
        k = intra_weight * inter_weight * overlapping_weight



        # To prevent overamplification of not-learned sources (seen as noise), cap k at 10
        k = min(k,10)

 

        # WIP recording only if it is a real combi, not for crumbing
        if average_crm_2d[curr_dataset_id, curr_tf_id]>0:
            coefs_norm[combi] = k # Record it

        


        # Only write it in the tsv if it is a real source (normalization applied to crumbs does not count)
        # TODO Since averaging over all_before takes time, move this condition directly to the calculation itself
        if average_crm_2d[curr_dataset_id, curr_tf_id]>0:
            
            line=str(combi[0])+'\t'+str(combi[1])+'\t'+str(intra_weight)+'\t'+str(inter_weight)+'\t'+str(overlapping_weight)+'\t'+str(k)+'\n'
            # Maybe add prediction_this and occupancy ?
            logcombifile.write(line)


    # So in the end we have a dictionary of combinations.
    # Normalize by the lowest k as otherwise we give 1000 to the "typical" behavior.
    # Let's say we want to give 750 to typical behavior NOTE IN PAPER IMPORTANT FOR INTERPRETATION
    # We want some margin for improvement : lowest k should bso insteaf of highest k, divide by 75 percent of lowest k
    #maxval = 1/0.75 * coefs_norm[min(coefs_norm, key=coefs_norm.get)]
    #coefs_norm_final = {k: v/maxval for k,v in coefs_norm.items()}
    # To prevent overamplification of not-learned sources (seen as noise), cap k at 10



    logcombifile.close()

    return coefs_norm




def estimate_corr_group_for_combi(dataset_name, tf_name,
                                    all_datasets, all_tfs, model,
                                    crm_length, squish_factor,
                                    before_value = 1, use_crumbing = True):
    """
    For a given source (dataset_name, tf_name) will estimate the corresponding correlation group
    This is done by returning simply what happens when the model is asked to rebuild a CRM containing only a peak for the source being considered

    This is simply a portion of the `estimate_corr_group_normalization_factors()` code. 
    TODO Functionalize it to prevent duplicate code.
    """

    combi = dataset_name, tf_name # Create this tuple to match the format of the other code

    curr_dataset = combi[0]
    curr_tf = combi[1]

    # get id in list of the dataset and tf
    try:
        curr_dataset_id = all_datasets.index(curr_dataset)
        curr_tf_id = all_tfs.index(curr_tf)
    except:
        print('Dataset or TR specified was not found. Skipping correlation group estimation.')
        return -1

    # Create an empty CRM with a peak only for this combi
    x = np.zeros((crm_length, len(all_datasets), len(all_tfs)))
    x[:,curr_dataset_id, curr_tf_id] = before_value

    if use_crumbing: x = utils.add_crumbing(x) # Add crumbing

    # See what the model rebuilds
    xp = utils.squish(x, factor = squish_factor)
    xp2 = xp[np.newaxis, ..., np.newaxis]
    prediction = model.predict(xp2)[0,:,:,:,0]
    prediction_2d = np.mean(prediction, axis=0) #

    # The plot we want
    plt.figure(figsize = (6,5)); resfig = sns.heatmap(np.transpose(prediction_2d),
        annot=True, cmap = 'Greens',
        xticklabels = all_datasets, yticklabels = all_tfs, fmt = '.2f')

    return resfig, prediction_2d


 




################################################################################
# -------------------------------- Q-score ----------------------------------- #
################################################################################


def calculate_q_score(model, list_of_many_befores,
    all_datasets_names, all_tf_names):
    """
    Given a model and generated befores data for the model, will compute the Q-score for this model.

    list_of_many_befores is a result of a command such as `list_of_many_befores = get_some_crms(generator, nb_of_batches_to_generate = 200)`
    where `generator` is a generator of data for the model like train_generator


    If two dimensions (datasets or TF) correlate should result in a higher score for them than when alone and vice-versa.
    The Q-score quantifies this. See details in code comments.
    """

    # Required labels for seaborn heatmaps ; Q labels are : all datasets, then all tfs
    q_labels = all_datasets_names + all_tf_names


    # Predict and compute anomalies using the model
    list_of_many_predictions = [model.predict(X[np.newaxis, ..., np.newaxis]) for X in list_of_many_befores]
    list_of_many_anomaly_matrices = [anomaly(before, predict[0,:,:,:,0]) for before, predict in zip(list_of_many_befores, list_of_many_predictions)]

    # TODO I recompute those in crm_diag_plots(), maybe functionalize it ?
    summed_by_tf = [np.sum(X, axis=1) for X in list_of_many_befores]
    concat_by_tf = pd.DataFrame(np.vstack(summed_by_tf))
    summed_by_dataset = [np.sum(X, axis=2) for X in list_of_many_befores]
    concat_by_dataset = pd.DataFrame(np.vstack(summed_by_dataset))

    # Correlation of TF and dataset
    # General correlation is calculated by concatenating horizontally 
    corr = pd.concat([concat_by_dataset,concat_by_tf],axis=1).corr()
    # NOTE As with usual dimensions, datasets come first, then tfs
    d_offset = len(concat_by_dataset.columns) # Used because in corr, for example the 2nd tf is at the position nb_of dataset + 2



    def q_score_for_pair_from_groupdf(group, AB_corr_coef):
        # The Q-score should take as input dataframes like the ones I usuallly 
        # create for the alone+both plots. (the 'group' variable) and the estimated corr coef for AB
        # So each line is this : [dimension (A or B), value, status (both or alone)]
        # Calculate if the means are different for A alone and A with B

        # NOTE This is directional, we see if the score of A changes, not B (B will be done later by calling this function with the group reversed)

        # Select A alone values and A both
        scores_alone = group.loc[(group['status'] == 'alone') & (group['dim'] == 'A')]['score']
        scores_both  = group.loc[(group['status'] == 'both')  & (group['dim'] == 'A')]['score']

        mean_A_alone = np.mean(scores_alone)
        mean_A_both = np.mean(scores_both)

        # Is there a significant difference ? Do a student t-test
        mean_diff = scipy.stats.ttest_ind(scores_alone, scores_both, equal_var=False)#.pvalue
        is_significant = mean_diff.pvalue < 0.05

        # Now the formula for the q-score
        qscore = (AB_corr_coef - is_significant)**2

        return qscore, mean_A_alone, mean_A_both, mean_diff.pvalue



    # Function to compute an individual q-score
    def get_scores_when_copresent_from_matrices(dim_tuple_A, dim_tuple_B, all_befores, all_anomalies):
        """
        The dim_tuples are in the format (dimension, axis); so (2,1) means 2nd dataset and (4,2) means 4th TF
        """

        group = []

        for m, ma in zip(all_befores, all_anomalies):
            # np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...]
            sliceA = np.take(m, dim_tuple_A[0], axis=dim_tuple_A[1])
            is_A_present = (np.sum(sliceA) != 0)

            sliceB = np.take(m, dim_tuple_B[0], axis=dim_tuple_B[1])
            is_B_present = (np.sum(sliceB) != 0)

            # Record status
            if (is_A_present and is_B_present) : status = 'both'
            elif (is_A_present or is_B_present) : status = 'alone'
            else : status = 'none'


            # Only proceed if there is at least one of the pair, A or B
            if status != 'none':

                ### Now record each nonzero value in the dimensions in anomalies
                # For simplicity, we average along the entire X axis, but do that on
                # the nonzeros of course. In the true diagnostic figures on real CRM
                # we don't do that and correctly consider each peak's value.
                anomaly_xsummed = np.array(np.ma.masked_equal(ma, 0).mean(axis=0))

                # Reduce axis by 1 since we summed along the region_size axis
                sliceA_anomaly = np.take(anomaly_xsummed, dim_tuple_A[0], axis=dim_tuple_A[1]-1)
                sliceB_anomaly = np.take(anomaly_xsummed, dim_tuple_B[0], axis=dim_tuple_B[1]-1)

                # Record all nonzero values for each
                for v in sliceA_anomaly[sliceA_anomaly.nonzero()]: group += [('A',v, status)]
                for v in sliceB_anomaly[sliceB_anomaly.nonzero()]: group += [('B',v, status)]

        # Return the result df
        groupres =  pd.DataFrame(group, columns = ['dim','score','status'])
        return groupres



    # Compute all q-scores :
    all_qscores = []

    # Generate all dim_tuples
    dt_nb = len(concat_by_dataset.columns)
    tf_nb = len(concat_by_tf.columns)

    dim_tuples = [(i, 1) for i in range(dt_nb)] + [(j, 2) for j in range(tf_nb)]

    # Get all pairwise ordered combinations
    all_duos = list(itertools.permutations(dim_tuples, 2))

    for dim_tuple_A, dim_tuple_B in all_duos:


        # compute group
        group = get_scores_when_copresent_from_matrices(dim_tuple_A, dim_tuple_B,
            list_of_many_befores, list_of_many_anomaly_matrices)

        # Finally, we can calculate the Q-score for this particular pair !
        # Get the localisation of the full corr matrix. 1 is dataset and 2 is TF, so we add the d_offset (nb of datasets) in the second case only
        locA = dim_tuple_A[0]+(d_offset*(dim_tuple_A[1]-1))
        locB = dim_tuple_B[0]+(d_offset*(dim_tuple_B[1]-1))
        AB_corr_coef = corr.iloc[locA, locB]

        #print('corr = '+str(AB_corr_coef))
        qscore, mean_A_alone, mean_A_both, mean_diff_pvalue = q_score_for_pair_from_groupdf(group, AB_corr_coef)

        # Make it more diagnostically explicit, with a giant dataframe giving the qscore but also the correlation
        all_qscores += [(dim_tuple_A, dim_tuple_B, AB_corr_coef, qscore, mean_A_alone, mean_A_both, mean_diff_pvalue)]

    all_qscores_df = pd.DataFrame(all_qscores, columns = ['dimA','dimB','corr','qscore', 'mean_A_alone', 'mean_A_both', 'mean_diff_pvalue'])


    # Sum all original 'before' matrices along the X axis to get the mean_frequencies
    mean_freq = np.mean(list_of_many_befores, axis = (0,1))
    # Useful in q-score weight calculation

    # Finally make it into a matrix and see the contributions
    tnb = dt_nb + tf_nb
    res = np.zeros((tnb,tnb))
    posvar = np.zeros((tnb,tnb))
    q_weights = np.zeros((tnb,tnb))

    for _, row in all_qscores_df.iterrows():

        dim_tuple_A = row['dimA'] ; dim_tuple_B = row['dimB']

        # Datasets first then tf
        dim1_raw = dim_tuple_A[0] + d_offset*(dim_tuple_A[1]-1)
        dim2_raw = dim_tuple_B[0] + d_offset*(dim_tuple_B[1]-1)


        # In a sort-of-Bonferroni correction, we fix the p-value threshold at 5% divided by the number of dimensions
        val = row['mean_diff_pvalue'] < (0.05 / tnb)


        # We consider only variation in the same direction as the correlation coefficient.
        # ie. when corr coeff is positive (almost all cases) we discard negative variations that
        # are due to there being no correlation but sometimes noise. However if negative correlation we should keep cases where alone > both

        both_higher_than_alone = row['mean_A_alone'] < row['mean_A_both']

        res[dim1_raw, dim2_raw] = val

        current_corr = corr.iloc[dim1_raw, dim2_raw]
        current_corr_is_pos = np.sign(current_corr) > 0
        posvar[dim1_raw, dim2_raw] = int(both_higher_than_alone == current_corr_is_pos)


        # Take the weights for A and B and sum them and multiply that to the S-score
        A_weight = np.take(mean_freq, dim_tuple_A[0], dim_tuple_A[1]-1)
        B_weight = np.take(mean_freq, dim_tuple_B[0], dim_tuple_B[1]-1)
        qw = np.mean(A_weight) * np.mean(B_weight)
        # We will use this later in calculating the q-score.
        # Multiply and not sum, and take sqrt of product to smooth over differences
        q_weights[dim1_raw, dim2_raw] = np.sqrt(qw)

        # We would like the diagonal of weights to stay full zeros. We don't care about A vs A.
        np.fill_diagonal(q_weights, 0)

    plt.figure(); posvar_x_res_plot = sns.heatmap(posvar * res, xticklabels=q_labels, yticklabels=q_labels).get_figure()

    # To simplify, we binarize by marking as "correlating" pairs with a correlation coefficient above the average
    mean_corr = np.mean(np.mean(corr))
    c = corr > mean_corr

    # But plot the true correlation instead
    plt.figure(); corr_plot = sns.heatmap(corr, xticklabels=q_labels, yticklabels=q_labels).get_figure()

    ## Finally calculate the q-score
    # Goal is to quantify whether there is a difference between the observed 
    # "correlation" in posvar*res and the theoretical ones in c ?
    # Then multiply by the weights
    q_raw = ((posvar*res)-c)**2
    q = q_raw * q_weights
    plt.figure(); qscore_plot = sns.heatmap(q, cmap = 'Reds', xticklabels=q_labels, yticklabels=q_labels).get_figure()


    # Fix for truncated axis tick labels
    qscore_plot.tight_layout()
    corr_plot.tight_layout()
    posvar_x_res_plot.tight_layout()

    return (q, qscore_plot, corr_plot, posvar_x_res_plot)







################################################################################
# --------------------------- Diagnostic plots ------------------------------- #
################################################################################

# ----------------------------- Artificial data ------------------------------ #

def proof_artificial(trained_artificial_model, partial_make_a_fake_matrix,
    region_length,nb_datasets,nb_tfs, reliable_datasets = None, tf_groups = None,
    squish_factor = 10, n_iter = 1000):
    """
    Generates some artificial data with separated peaks.
    You must supply a trained model, and a partial call to artificial_data.make_a_fake_matrix()
    with return_separately enabled (the partial will be used as partial_make_a_fake_matrix() with no additional arguments !)

    The parameters used for matrix generation must also be supplied (region_length,nb_datasets,nb_tfs, reliable_datasets if applicable, tf_groups if applicable)
        where tf_groups is a list of n elements, each element being a list giving a correlation group of tfs
    """

    # Scores
    # Prepare a dataframe which will be appended to everytime
    # Columns are respectively : anomaly score, type of peak (noise, stack...), number of brothers (peaks sharing a TF or a dataset)
    df = pd.DataFrame(columns = ["anomaly_score","rebuilt_value","type","brothers","tf_group","within_stack","in_reliable_datasets","nb_peaks_in_stack_for_this_crm"])

    def get_brothers_nb(peaks, dataset, tf):
        # From the list of peaks, return the number of peaks with tha same dataset or TF. "peaks" must be an order 1 list (not a list of lists)
        try :
            selected = [peak for peak in peaks if ((peak[0] == dataset) | (peak[1] == tf))]
            return len(selected)
        except : return 0

    def was_this_dataset_tf_combi_used_in_stack(dataset, tf, peaks_stack):
        result = False
        for p in peaks_stack:
            if (p[0] == dataset) and (p[1] == tf) : result = True
        return result

    def partial_scores_for_this_CRM(separated_peaks, anomaly_matrix, before, rebuilt, reliable_datasets = None, tf_groups = None):

        partial_df = pd.DataFrame(columns = ["anomaly_score","rebuilt_value","type","brothers","tf_group","within_stack","in_reliable_datasets","nb_peaks_in_stack_for_this_crm"])

        # TODO split more, especially for the noise categories (see below) to make more explicit plots

        # Unpack the seperated peaks
        peaks_stack, peaks_noise, peaks_watermark = separated_peaks

        # ---------------------- Scores depending on category -------------------- #

        # Remember which datasets and which TFs were used in this particular stack.
        # As each artificial CRM uses the same datasets, but different TF correlation groups,
        # this must be called and reinitialized for each CRM
        datasets_used = list()
        tfs_used = list()


        # Workaround : as peaks are additive when created (multiple artificial overlapping peaks sum their intensities), set their intensity to be actually the summed value at the same position in the matrix
        for peak in peaks_stack:      dataset, tf, center, length, intensity = peak; peak = (dataset, tf, center, length,np.mean(before[center-length:center+length,dataset,tf]))
        for peak in peaks_noise:      dataset, tf, center, length, intensity = peak; peak = (dataset, tf, center, length,np.mean(before[center-length:center+length,dataset,tf]))
        for peak in peaks_watermark:  dataset, tf, center, length, intensity = peak; peak = (dataset, tf, center, length,np.mean(before[center-length:center+length,dataset,tf]))

        # Make unique
        peaks_stack = list(set(peaks_stack))
        peaks_noise = list(set(peaks_noise))
        peaks_watermark = list(set(peaks_watermark))

        separated_peaks = [peaks_stack, peaks_noise, peaks_watermark] # Rewrite separated peaks after the modification is done

        stack_left_border=0
        stack_right_border=before.shape[0]

        for stack_peak in peaks_stack :

            # Read peak and take the anomaly score in the rebuilt matrix for this peak.
            dataset, tf, center, length, intensity = stack_peak
            begin = max(0,center-length); end = center+length

            score = np.max(anomaly_matrix[begin:end,dataset,tf])
            value = np.max(rebuilt[begin:end,dataset,tf])

            ## Number of "brothers" : peaks from the same tf or the same dataset
            # Consider all the peaks ?
            brothers_nb = get_brothers_nb(peaks_stack + peaks_noise + peaks_watermark, dataset, tf)

            # Append score
            partial_df.loc[len(partial_df)] = [score,value,"stack",brothers_nb,"current",True, True, len(peaks_stack)]

            # Remember which datasets/tfs were used
            datasets_used = datasets_used + [dataset]
            tfs_used = tfs_used + [tf]

            # Remember the borders
            stack_left_border = min(stack_left_border, begin)
            stack_right_border = max(stack_right_border, end)



        # Reliable datasets and TF correlation groups : defaults
        if reliable_datasets == None : reliable_datasets = list(range(int(nb_datasets/2),nb_datasets))
        if tf_groups == None : tf_groups = [range(int(nb_tfs/2)), range(int(nb_tfs/2),nb_tfs)]


        ## Phantoms - check in the rebuilt matrix

        # First, determine which group was used. Assuming only one was picked.
        current_tf_group = list()
        not_current_tf_group = list()
        for group in tf_groups :
            group = list(group)
            check = all(tf in group for tf in tfs_used)
            if check : current_tf_group += group
            else : not_current_tf_group += group

        # Phantoms for TFs of the same correlation group :
            # take all matrix positions whose tf is IN tfs_used, but whose datasets, begin, end are within stack
            # but the dataset is NOT in datasets_used (otherwise I'm reading a true peak again)
        for tf in current_tf_group:
            for dataset in reliable_datasets:
                if not was_this_dataset_tf_combi_used_in_stack(dataset, tf, peaks_stack):
                    score = np.max(anomaly_matrix[stack_left_border:stack_right_border,dataset,tf])
                    value = np.max(rebuilt[stack_left_border:stack_right_border,dataset,tf])
                    brothers_nb = get_brothers_nb(peaks_stack + peaks_noise + peaks_watermark, dataset, tf)
                    partial_df.loc[len(partial_df)] = [score,value,"phantom",brothers_nb,"current", True, True, len(peaks_stack)]

        # Phantoms NOT in tf_used
        for tf in not_current_tf_group:
            for dataset in reliable_datasets:
                if not was_this_dataset_tf_combi_used_in_stack(dataset, tf, peaks_stack):
                    score = np.max(anomaly_matrix[stack_left_border:stack_right_border,dataset,tf])
                    value = np.max(rebuilt[stack_left_border:stack_right_border,dataset,tf])
                    brothers_nb = get_brothers_nb(peaks_stack + peaks_noise + peaks_watermark, dataset, tf)
                    partial_df.loc[len(partial_df)] = [score,value,"phantom",brothers_nb,"different", True, True, len(peaks_stack)]


        # Same for peaks noise and peaks watermak
        for noise_peak in peaks_noise:
            dataset, tf, center, length, intensity = noise_peak
            begin = max(0,center-length); end = center+length
            brothers_nb = get_brothers_nb(peaks_stack + peaks_noise + peaks_watermark, dataset, tf)
            score = np.max(anomaly_matrix[begin:end,dataset,tf])
            value = np.max(rebuilt[begin:end,dataset,tf])

            is_within_stack = ((begin > stack_left_border) or (end < stack_right_border))

            # If it is a noise from reliable datasets
            if dataset in reliable_datasets :
                if tf in current_tf_group:
                    if is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise",brothers_nb, "current", True, True, len(peaks_stack)]
                    if not is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise",brothers_nb, "current", False, True, len(peaks_stack)]

                else:
                    if is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise",brothers_nb, "different", True, True, len(peaks_stack)]
                    if not is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise",brothers_nb, "different", False, True, len(peaks_stack)]

            # If it is a noise from unreliable datasets
            if dataset not in reliable_datasets :
                if tf in current_tf_group:
                    if is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise_unreliable",brothers_nb, "current", True, False, len(peaks_stack)]
                    if not is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise_unreliable",brothers_nb, "current", False, False, len(peaks_stack)]

                else:
                    if is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise_unreliable",brothers_nb, "different", True, False, len(peaks_stack)]
                    if not is_within_stack: partial_df.loc[len(partial_df)] = [score,value,"noise_unreliable",brothers_nb, "different", False, False, len(peaks_stack)]


        ## Watermark
        for watermark_peak in peaks_watermark:
            dataset, tf, center, length, intensity = watermark_peak
            begin = max(0,center-length); end = center+length
            score = np.max(anomaly_matrix[begin:end,dataset,tf])
            value = np.max(anomaly_matrix[begin:end,dataset,tf])
            brothers_nb = get_brothers_nb([peaks_stack + peaks_noise + peaks_watermark], dataset, tf)
            partial_df.loc[len(partial_df)] = [score, value, "watermark", brothers_nb,"itself", False, False, len(peaks_stack)]


        return partial_df





    # Perform the required iterations
    for _ in range(n_iter):

        # Generate one artificial datum (CRM) with separated peaks using the
        # supplied partial call with no additional arguments.
        separated_peaks = partial_make_a_fake_matrix()


        # Submit MERGED peaks matrix to model
        merged_peaks = [peak for peak_category in separated_peaks for peak in peak_category]
        before_nosquish = ad.list_of_peaks_to_matrix(merged_peaks,region_length,nb_datasets,nb_tfs,
                                                                    ones_only = partial_make_a_fake_matrix.keywords['ones_only'])

        # NOTE make_a_fake_matrix() does not squish the matrix along the X
        # axis. We must do that before submitting to the model, and unsquish
        # before considering the result
        before = utils.squish(before_nosquish[..., np.newaxis], squish_factor)
        rebuilt = trained_artificial_model.predict(before[np.newaxis,...])
        rebuilt = utils.stretch(rebuilt[0,...], squish_factor)

        anomaly_matrix = anomaly(before_nosquish[..., np.newaxis], rebuilt)

        partial_df = partial_scores_for_this_CRM(separated_peaks, anomaly_matrix, before_nosquish, rebuilt[...,0], reliable_datasets, tf_groups)

        df = pd.concat([df,partial_df])


    # Stick all these scores in a dataframe !
    return df, separated_peaks
    # TODO : REMOVE THE RETURN OF separated_peaks IT IS ONLY HERE FOR DEBUG PURPOSES




# ----------------------------- CRM themselves ------------------------------- #


def get_some_crms(train_generator, nb_of_batches_to_generate = 20, try_remove_crumbs = True):
    """
    Will run the supplied train_generator to produce some CRM matrices.
    The number of batches to be ran is also a parameter.

    This function returns all generated CRMs as a list.

    NOTE : if asked, it also tentatively removes curmbs through rounding
    """

    batches = [next(train_generator)[0] for i in range(nb_of_batches_to_generate)]
    many_crms = np.concatenate(batches, axis=0)

    # TODO instead of calling test_generator, call it with get_matrix(crm_id) once that works

    if try_remove_crumbs:
        # Remove crumbs by rounding. Crumbs should never be at 1 unless massive evidnce, so mat-0.5 should never be rounded at 1
        # For cases where we were not crumbed, since peaks are usually at 1, we use mat-0.45 instead, so 1-0.45=0.55 is still rounded to 1
        # TODO As for anomaly, this will break when we have non-binary peak values !!
        list_of_many_crms = [np.clip(np.around(X-0.45), 0,1)[:,:,:,0] for X in many_crms]

    return list_of_many_crms







def crm_diag_plots(list_of_many_crms, datasets, cl_tfs):

    # Average CRM
    plt.figure()
    average_crm = np.mean(np.mean(np.array(list_of_many_crms), axis=0), axis=0)
    average_crm = pd.DataFrame(np.transpose(average_crm), index = cl_tfs, columns = datasets)
    mypalette = sns.cubehelix_palette(256) ; mypalette[0] = [1,1,1]
    average_crm_fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(average_crm, ax=ax, cmap=mypalette, annot = True)

    # For TFs : sum many_crms along the TF axis, then do a correlation matrix.
    plt.figure()
    summed_by_tf = [np.sum(X, axis=1) for X in list_of_many_crms]
    concat_by_tf = pd.DataFrame(np.vstack(summed_by_tf), columns = cl_tfs)
    corr = concat_by_tf.corr()
    tf_corr_fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap='Greens',
                        annot=True, fmt = ".2f")
    plt.figure()
    tf_abundance_fig = concat_by_tf.sum().plot.bar(figsize = (10,6)).get_figure() # And abundance

    # Same for datasets
    plt.figure()
    summed_by_dataset = [np.sum(X, axis=2) for X in list_of_many_crms]
    concat_by_dataset = pd.DataFrame(np.vstack(summed_by_dataset), columns = datasets)
    corr = concat_by_dataset.corr()
    dataset_corr_fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap='Blues',
                        annot=True, fmt = ".2f")
    plt.figure()
    dataset_abundance_fig = concat_by_dataset.sum().plot.bar(figsize = (10,6)).get_figure()

    return (average_crm_fig, tf_corr_fig, tf_abundance_fig, dataset_corr_fig, dataset_abundance_fig)





# ----------------------------- After denoising ------------------------------ #


def group_peaks_by_crm(peaks_file_path, crm_file_path, group_by_CRM = False):
    """
    pandas dataframe of all the peaks, by their CRM. Can be grouped by CRM (pandas mapping) if required
    """

    # Use the result of `bedtools intersect -a file.bed -b crm_selected.bed -wa -wb`
    crm = pybedtools.BedTool(crm_file_path)
    output_bed = pybedtools.BedTool(peaks_file_path)
    intersected = output_bed.intersect(crm, wa=True, wb=True)
    df = intersected.to_dataframe(names = ['chr','start','end','name','anomaly_score','strand','CRM_chr','CRM_start','CRM_end','CRM_name','CRM_score','CRM_strand','CRM_center','CRM_center+1'])

    # Group by CRM
    df['full_crm_coordinate'] = df["CRM_chr"].map(str)+'.'+df["CRM_start"].map(str)+'.'+df["CRM_end"].map(str)

    if group_by_CRM : return df.groupby(['full_crm_coordinate'])
    else : return df



def plot_score_per_crm_density(peaks_file_path, crm_file_path):
    """
    This will plot the reparition of anomaly scores in each CRM, as a function
    of the number of peaks that were originally in it.
    """

    plots = []

    # Get the peaks and group them by CRM
    df = group_peaks_by_crm(peaks_file_path, crm_file_path)
    df_gp = df[['anomaly_score','full_crm_coordinate']].groupby(['full_crm_coordinate'])


    # Distribution of scores
    p = ggplot(df, aes(x='anomaly_score')) + geom_histogram(bins = 100)
    plots += [p]

    ## Average (and max) score per number of peaks in CRM
    res = pd.DataFrame(columns=['nb_peaks', 'average_score', 'max_score'])
    for _, group in df_gp:
        scores = [float(s) for s in group['anomaly_score']]
        npe = len(scores)
        asn = np.mean(scores)
        msn = np.max(scores)

        new_row = pd.DataFrame({'nb_peaks': [npe], 'average_score': [asn], 'max_score': [msn]})
        res = pd.concat([res, new_row], ignore_index = True)

    # Group by number of peaks
    peaks_nb = np.array(res['nb_peaks'])+1E-10
    peaks_nb = peaks_nb.astype(int)

    peak_nb_fact = np.log2(peaks_nb)

    res['nb_peaks_fact'] = peak_nb_fact.astype(int).astype(str)

    # Plots
    p = ggplot(res, aes(x='nb_peaks_fact', y='average_score')) + geom_violin() + xlab("Number of peaks (log2)") + geom_boxplot(width=0.1)
    plots += [p]
    p = ggplot(res, aes(x='nb_peaks_fact', y='max_score')) + xlab("Number of peaks (log2)") + geom_boxplot(width=0.1)
    plots += [p]
    # TODO NOTE removed the geom_violin from the last plot due to an inexplicable bandwith bug


    return plots





def get_scores_whether_copresent(tf_A, tf_B, atypeak_result_file, crm_file_path):
    """
    Used for this : if in literature we know A and B are found together truly, get the scores for A and B when they are together and alone.
    Theoretically, known cofactors should have significantly higher scores when found together.
    """

    result = list()


    # TODO get combinations of N TFs and redo the same kind of graphs, but for n-wise presences
    # Try a Venn diagram ?
    # iterable = [1,2,3,4,5,6]
    # all_combis = list()
    # for l in range(1,len(iterable)+1):
    #     all_combis += list(itertools.combinations(iterable,l))


	# Group all the peaks by CRM...
    df = group_peaks_by_crm(atypeak_result_file, crm_file_path)
    df[['dataset','tf','cell_line']] = df['name'].str.split('.', expand = True)     # Get peak info for all crm
    df_gp = df.groupby(['full_crm_coordinate']) # Now group by CRM

    for _, group in df_gp:

        # Is A present ? Is B present ? Are they both present ?
        present_tfs = set(group.loc[:,'tf'].tolist())

        flagA = tf_A in present_tfs
        flagB = tf_B in present_tfs

        # Record the score for each TF and whether A, B, or both were present
        for _, peak in group.iterrows():

            tf = peak['tf']
            score = peak['anomaly_score']

            if (tf == tf_A) | (tf == tf_B):
                if flagA & flagB: result += [(tf, score,'both')]
                elif flagA: result += [(tf, score,'alone')]
                elif flagB: result += [(tf, score,'alone')]


    # Stock the result in a dataframe
    rdf = pd.DataFrame(result) ; rdf.columns = ['tf','score','status']

	# Violin plot
    p =  ggplot(rdf, aes(x='tf', y = 'score', fill='status'))
    p += geom_violin(position=position_dodge(1))
    p += geom_boxplot(width = 0.1, position=position_dodge(1), outlier_alpha=0.1)
    p += theme(figure_size=(4,4))

    return p, rdf