"""
Functions to compute scores, produce result, and evaluate the model.

These functions are used both by the model (notably anomaly score calculation) and the diagnostic functions.
"""

import os
import sys
import time
import functools
import itertools

#import pybedtools
import numpy as np
import pandas as pd
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt


from plotnine import ggplot, aes, geom_histogram, scale_fill_grey, geom_violin, geom_boxplot, position_dodge, theme

import lib.artificial_data as ad
import lib.model_atypeak as cp
from lib import utils


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




def produce_result_file(all_matrices, output_path, model,
    get_matrix_method, parameters,
    datasets_clean, cl_tfs,
    add_track_header = True,
    thread_nb = 8):
    """
    Will take all CRM whose ID is in all_matrices, rebuild them with the
    provided model, then compute the anomaly score and write the result in a BED
    file at output_path.
    """

    # Open result file in append mode
    rf = open(output_path,'w')
    if add_track_header : rf.write('track name ='+parameters['cell_line']+' description="'+parameters['cell_line']+' peaks with anomaly score" useScore=1'+'\n')



    def produce_result_for_matrix(m):
        """
        Wrapper to produce a result for one matrix. Used in the main loop.
        """
        # TODO Maybe add the possibility to supply a custom generator ?

        # Collect original matrix and lines
        current_matrix, origin_data_file, crm_start = get_matrix_method(m, return_data_lines = True)

        # Pad the matrix (much like we do in model_atypeak.generator_unsparse)
        # TODO The method must also return crm_length then for coordinate correction !
        crm_length = current_matrix.shape[0]
        current_matrix_padded = np.pad(current_matrix, pad_width = ((parameters["pad_to"] - current_matrix.shape[0],0),(0,0),(0,0)), mode='constant', constant_values=0)
 
        # Compute result and anomaly
        squished_matrix = utils.squish(current_matrix_padded[:,:,:,np.newaxis])
        prediction = model.predict(squished_matrix[np.newaxis,:,:,:,:])
        result_matrix = utils.stretch(prediction[0,:,:,:,:])

        # This uses directly the anomaly function defined in this file
        anomaly_matrix = anomaly(current_matrix_padded, result_matrix[:,:,:,0])

        # Now write to the result BED the current peaks with their newfound anomaly score
        result = utils.produce_result_bed(origin_data_file, anomaly_matrix,
                                            datasets_clean, cl_tfs,
                                            crm_start, crm_length,
                                            debug_print = False)


        # WARNING : This is 1 - anomaly, so the score is actually high for good peaks !!
        return result



    ## Main loop iterating over all matrices
    result = list()

    print('Beginning processing...')

    cnt = 0
    for m in all_matrices:

        result += produce_result_for_matrix(m)

        # Progress display
        sys.stdout.write("\r" +"Processed CRM nb. : " + str(cnt+1)+" / "+str(len(all_matrices)))
        sys.stdout.flush()
        cnt += 1

    # Write the final file
    for line in result: rf.write(line+'\n')
    rf.close()
    print(" -- Done.")










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


    for combi in all_combis:




        BEFORE_VALUE = 1 # Careful about affine effect. Write somewhere.




        # Combi information
        curr_dataset = combi[0] # get id in list of the dataset
        curr_tf = combi[1] # get id in list of the tf
        # Convert dataset and tf to the approriate coordinates
        curr_dataset_id = all_datasets.index(curr_dataset)
        curr_tf_id = all_tfs.index(curr_tf)



        # --------------------------  INTRA AND OVERLAPPING GROUP NORMALISATION
        # Build a full CRM. Assuming negative effects are negligible and that in biology more is always better
        # See the maximum score reached by each : they should be "100%" (ie. perfect score, equal to BEFORE with crumbing), if more or less there is bias to be corrected
        full_crm = (average_crm_2d>0).astype(int)

        # full_crm_3d = np.stack([average_crm_2d]*320, axis=0).astype('float64')
        # full_crm_3d = full_crm_3d/np.mean(full_crm_3d)
        # full_crm_3d = cp.look_here_stupid(full_crm_3d)
        
        full_crm_3d = np.stack([full_crm]*320, axis=0).astype('float64')

        # TODO UNHARDCODE THE 320 ABOVE !!


        if use_crumbing: full_crm_3d = cp.look_here_stupid(full_crm_3d)

        predictionf= model.predict(full_crm_3d[np.newaxis,...,np.newaxis])[0,:,:,:,0]
        prediction_2df = np.mean(predictionf, axis=0) # 2D - mean along region axis
        before_2df = np.mean(full_crm_3d, axis=0)

        # Get the weight
        first_weight = 1/(prediction_2df[curr_dataset_id, curr_tf_id]/before_2df[curr_dataset_id, curr_tf_id])



        # -------------------------- INTER GROUP BIAS
        # Put a peak only for this combi and look at the phantoms
        # Here, we consider the correlation group itself

        x = np.zeros((crm_length,len(all_datasets),len(all_tfs))) # Create an empty CRM


        # Add a major peak for this particular dataset+tf pair across the entire region
        # Use a value of 10 or 100 to force MSE to show groups ??
        # No use 1 instead to prevent affine pbs
        x[:,curr_dataset_id, curr_tf_id] = BEFORE_VALUE
        if use_crumbing: x = cp.look_here_stupid(x) # Add crumbing
            
        xp = utils.squish(x, factor = squish_factor)
        xp2 = xp[np.newaxis, ..., np.newaxis]
        prediction = model.predict(xp2)[0,:,:,:,0]
        prediction_2d = np.mean(prediction, axis=0) # 2D - mean along region axis # VERY IMPORTANT THAT HERE IT IS THE MEAN. TODO SAY SO IN PAPER !!!


        # APPLY THE FIRST WEIGHT and proceed as usual
        prediction_2d = prediction_2d * first_weight
        # Rq : since I normalize later, this is useless !!!!!????




        # For ease of understanding, make the average CRM relative to highest abundance
        relative_average_crm_2d = average_crm_2d/np.max(average_crm_2d)

        # Correlation groups reach a rebuilt value of 1 on each peak (hmm after normalizing intra group BE CAREFUL NOTE THIS SOMEWHERE ELSE) when complete, but we
        # need to normalize this by "how complete are they typically"


        # What would mark this combi as "complete" (value of 1) ?
        # MODIFIED : instead I normalize the prediction to have a sum of 1
        requested_group = prediction_2d/np.sum(prediction_2d)



        # # Would it really result in 1 values if fed to the model ?
        # t = model.predict((prediction/rebuilt_this)[np.newaxis,...,np.newaxis])
        # pred_t = np.transpose(np.mean(t[0,:,:,:,0], axis=0))
        # plt.figure(); sns.heatmap(pred_t, annot=True, cmap = 'Greys')
        # # No it does not... o remove this line "What would mark this combi as "complete" (value of 1) ?" and keep the line beginnign with "MODIFIED"


        # How complete is it usually ? To put it another way : how much abundance would that cost ?
        clipped_request = np.clip(requested_group,0,np.inf) # Although it discards negative influences, we need to clip the request to calculate usual completeness
        request_abundance = relative_average_crm_2d*clipped_request

        # Sum only finite
        request_abundance_finite_values = request_abundance[np.isfinite(request_abundance)]
        usual_completeness = np.sum(request_abundance_finite_values)

        # Divide by this_abundance to get proportional completeness
        # Or rather by max, since I want all the group ?
        #second_weight = usual_completeness/abundance_this
        second_weight = usual_completeness/np.max(request_abundance)





        # ------------- FINAL
        # Finally : final value is the product of the two weights
        k = first_weight * second_weight


        # Applying this k directly would tend towards giving 1000 to the "typical" behavior.
        # Let's say typical behaviour should get 750 to have some margin for improvement
        k = 0.75*k


        # To prevent overamplification of not-learned sources (seen as noise), cap k at 10
        k = min(k,10)

 



        coefs_norm[combi] = k # Record it

        # To prevent overamplification of not-learned sources (seen as noise), cap k at 10
        

        # TODO PRINT THIS IN A FILE SOMEWHERE !!!
        logcombifile = open(outfilepath,'a')
        logcombifile.write(str(combi)+'\t--> '+"First weight = " + str(first_weight)+'\n')
        logcombifile.write(str(combi)+'\t--> '+"Second weight = " + str(second_weight)+'\n')
        logcombifile.write(str(combi)+'\t--> '+"k = " + str(k)+'\n')
        logcombifile.write('------'+'\n')



    # So in the end we have a dictionary of combinations.
    # Normalize by the lowest k as otherwise we give 1000 to the "typical" behavior.
    # Let's say we want to give 750 to typical behavior NOTE IN PAPER IMPORTANT FOR INTERPRETATION
    # We want some margin for improvement : lowest k should bso insteaf of highest k, divide by 75 percent of lowest k
    #maxval = 1/0.75 * coefs_norm[min(coefs_norm, key=coefs_norm.get)]
    #coefs_norm_final = {k: v/maxval for k,v in coefs_norm.items()}
    # To prevent overamplification of not-learned sources (seen as noise), cap k at 10



    logcombifile.close()

    return coefs_norm_final




def estimate_corr_group_for_combi(dataset_name, tf_name,
                                    all_datasets, all_tfs, model,
                                    crm_length, squish_factor,
                                    before_value = 1):
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

    # TODO USE CRUMBING INSTEAD OF THIS
    x[:,:, curr_tf_id] += 0.1*before_value
    x[:,curr_dataset_id, :] += 0.1*before_value

    # See what the model rebuilds
    xp = utils.squish(x, factor = squish_factor)
    xp2 = xp[np.newaxis, ..., np.newaxis]
    prediction = model.predict(xp2)[0,:,:,:,0]
    prediction_2d = np.mean(prediction, axis=0) #

    # The plot we want
    plt.figure(figsize = (6,5)); resfig = sns.heatmap(np.transpose(prediction_2d),
        annot=True, cmap = 'Greens',
        xticklabels = all_datasets, yticklabels = all_tfs, fmt = '.2f')

    return resfig


 





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
        #print('mean A alone ='+str(mean_A_alone))
        #print('mean A both ='+str(mean_A_both))

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

        #print("dim tuple A = "+str(dim_tuple_A))
        #print("dim tuple B = "+str(dim_tuple_B))

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





    print("DEBUG THIS LOOP COMPLETE; THIS IS THE LONG LOOP I SHOULD TRY TO MULTIPROCESS IT TO SEE")





    # Sum all original 'before' matrices along the X axis to get the mean_frequencies
    mean_freq = np.mean(list_of_many_befores, axis = (0,1))
    # Useful in q-score weight calculation

    # Finally make it into a matrix and see the contributions
    tnb = dt_nb + tf_nb
    res = np.zeros((tnb,tnb))
    posvar = np.zeros((tnb,tnb))
    q_weights = np.zeros((tnb,tnb))

    for _, row in all_qscores_df.iterrows():




        print("DEBUG BEGINNING")





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
        #posvar[dim1_raw, dim2_raw] = both_higher_than_alone
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



    return (q, qscore_plot, corr_plot, posvar_x_res_plot)











    """
    corrs = [0.1,0.1,0.8,0.8]
    pvals = [0.5,1E-100,1E-100,0.5]
    [10**(1+r) for r in corrs]
    [-np.log10(p) for p in pvals]

    import numpy as np
    def qscore(r,p):
        # Sort of a logical AND : if there is a correlation, there should be
        # a differnce in the score.

        # Correlation term
        C = 10**(1+r)
        # Mean diffrence term
        S = -np.log10(p)

        # We expect S to be from 0 to 100 roughly ?

        diff = C-S

        # The score should be high when the difference is low so use a decreasing function
        return diff

    scores = [qscore(r,p) for r,p in zip(corrs, pvals)]
    """





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
    # HERE IS HOW TO APPEND : df.loc[len(df)] = [1,0.9,"noise",4]


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
    for name, group in df_gp:
        scores = [float(s) for s in group['anomaly_score']]
        npe = len(scores)
        asn = np.mean(scores)
        msn = np.max(scores)

        new_row = pd.DataFrame({'nb_peaks': [npe], 'average_score': [asn], 'max_score': [msn]})
        res = pd.concat([res, new_row], ignore_index = True)


    # Group by number of peaks
    peaks_nb = np.array(res['nb_peaks'])+1E-10
    peaks_nb = peaks_nb.astype(float)# change dtype

    peak_nb_fact = np.log2(peaks_nb)
    res['nb_peaks_fact'] = peak_nb_fact.astype(int).astype(str)

    # Plots
    p = ggplot(res, aes(x='nb_peaks_fact', y='average_score', fill='nb_peaks_fact')) + scale_fill_grey() + geom_violin() + geom_boxplot(width=0.1)
    plots += [p]
    p = ggplot(res, aes(x='nb_peaks_fact', y='max_score', fill='nb_peaks_fact')) + scale_fill_grey() + geom_violin() + geom_boxplot(width=0.1)
    plots += [p]

    return plots


















def get_scores_whether_copresent(tf_A, tf_B, atypeak_result_file, crm_file_path):
    """
    Used for this : if in literature we know A and B are found together truly, get the scores for A and B when they are together and alone.
    Theoretically, known cofactors should have significantly higher scores when found together.
    """

    result = list()


    # TODO get combinations of N TFs
    # import itertools
    #
    #
    # iterable = [1,2,3,4,5,6]
    # all_combis = list()
    # for l in range(1,len(iterable)+1):
    #     all_combis += list(itertools.combinations(iterable,l))


	# Group all the peaks by CRM...
    df = group_peaks_by_crm(atypeak_result_file, crm_file_path)
    df[['dataset','tf','cell_line']] = df['name'].str.split('.', expand = True)     # Get peak info for all crm
    df_gp = df.groupby(['full_crm_coordinate']) # Now group by CRM

    for name, group in df_gp:

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
