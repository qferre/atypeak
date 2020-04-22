# System
import os
import sys
import yaml
import time
import random
from functools import partial

# Data
import numpy as np
import pandas as pd
import scipy

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_violin, geom_boxplot, position_dodge, scale_fill_grey, geom_bar, theme

# ML
# NOPE MUST NOT IMPORT KERAS HERE
# import keras
# import keras.backend as K

## Custom libraries
#import lib.model_atypeak as cp      # Autoencoder functions
import lib.data_read as dr          # Process data produced by Makefile
import lib.artificial_data as ad    # Trivial data for control
import lib.result_eval as er        # Result production and evaluation
import lib.utils as utils           # Miscellaneous

import lib.prepare as prepare 

############################# PARAMETERS AND SETUP #############################

root_path = os.path.dirname(os.path.realpath(__file__))


parameters = yaml.load(open(root_path+'/parameters.yaml').read(), Loader = yaml.Loader)


### Set random seeds - Numpy and TensorFlow
SEED = parameters["random_seed"]
# Python internals
os.environ["PYTHONHASHSEED"]=str(SEED)
    #TODO seems useless ?? Then why did they recommend it ?
random.seed(SEED)
# For Numpy, which will also impact Keras, and Theano if used
np.random.seed(SEED)

# TENSORFLOW SEED SHOULD HAVE NO IMPACT NOW SINCE MODEL IS TRAINED







## Reading corresponding keys. See the documentation of get_data_indexes() for more.
crmtf_dict, datasets, crmid, datapermatrix, peaks_per_dataset, cl_tfs = dr.get_data_indexes(parameters['cell_line'], root_path)


#datasets_clean_ori = [dr.dataset_parent_name(d) for d in datasets] # TODO Datasets : might wish to use a parent name later
datasets_clean = sorted(list(set(datasets)), key=datasets.index) # Make unique while preserving order, which a `set` would not do


# Since those dictionaries are fixed for a cell line, prepare a partial call
get_matrix = partial(dr.extract_matrix,
    all_tfs = cl_tfs,
    cell_line = parameters['cell_line'], cl_crm_tf_dict = crmtf_dict,
    cl_datasets = datasets_clean, crm_coordinates = crmid,
    datapath = root_path+'/data/input/sorted_intersect/')

print('Parameters loaded.')





# Plot output path (different for artificial data)
plot_output_path = './data/output/diagnostic/'+parameters['cell_line']+'/'
if parameters['use_artificial_data'] : plot_output_path += 'artificial/'
if not os.path.exists(plot_output_path): os.makedirs(plot_output_path)




############################### DATA GENERATOR #################################


## First, prepare the parameters 
if parameters['use_artificial_data'] :
    # Override the datasets and TF names, replace them with random
    datasets_clean = ["dat_"+str(i) for i in range(parameters['artificial_nb_datasets'])]
    cl_tfs = ["tf_"+str(i) for i in range(parameters['artificial_nb_tfs'])]

else:
    # ---------------------------- Real data --------------------------------- #
    # Collect all CRM numbers (each one is a *sample*)
    matrices_id = crmid.values()

    # The list of matrices ID must be made only of unique elements. 
    # The order does not matter in and of itself, but must be randomized and the same for a given random seed, as if affects training
    seenid = set() ; all_matrices = [x for x in matrices_id if x not in seenid and not seenid.add(x)]
    random.shuffle(all_matrices)

    print("Using real data for the '"+parameters["cell_line"]+"' cell line.")

















"""
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
EVEYRHTING ABOVE IS IDENTICAL TO TRAIN.PY, functionalize it !!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!

"""





















"""
IMPORTANT NOTE
To ensure multiprocessing success, it is imperative that keras or tensorflow
are not imported in the main process.py code so as not to create a Session !
This must be done only by the subprocesses !
"""













































################################################################################
################################## REAL DATA ###################################
################################################################################



# Path for saving the model later
# Also used in processing the file for multithreading !
if not parameters['use_artificial_data'] :
    save_model_path = root_path+"/data/output/model/trained_model_"+parameters['cell_line']
else:
    save_model_path = root_path+"/data/output/model/trained_model_ARTIFICIAL"


# Switch : the user should calibrate with Q-score before processing the full data !
# So in the parameters if the switch process_full_real_data is not on, it should exit here and there

if not parameters["process_full_real_data"]:
    print("The parameter `process_full_real_data` was set to False, presumably because this was a parameter calibration run.")
    print("Hence, we stop before processing the real data.")
    sys.exit()

else:
    print("Proceeding with real data. Make sure you have calibrated the model correctly, using notably the Q-score !")
    print("Warning : this can be VERY long.")


    ################################ DENOISING/PRODUCE REAL DATA #####################################
    # Use the now-trained model on all data.

    # Only do this on true data of course
    if parameters['use_artificial_data'] :
        raise ValueError("Error : process_full_real_data was set to True, but use_artificial_data is also True.")
    else:
        print("Writing result BED file for peaks, with anomaly score.")
        print("This can be long (roughly 1 second for 10 CRMs with reasonably-sized queries, like 15 datasets x 15 TFs).")
        print("On big queries (like 25*50) it can be 1 second per CRM.")


        # Filepaths
        root_output_bed_path = root_path + '/data/output/bed/' + parameters['cell_line']
        

        output_bed_path = root_output_bed_path + ".bed"
        output_bed_merged = root_output_bed_path + "_merged_doublons.bed"

        output_bed_path_normalized_poub = root_output_bed_path + "_raw_normalized_by_tf_DEBUG.bed"  

        output_path_corr_group_normalized = root_output_bed_path + "_merged_doublons_normalized_corr_group.bed"
        output_bed_path_final = root_output_bed_path + "_FINAL_merged_doublons_normalized_corr_group_normalized_by_tf.bed"




        ### Raw file production

        # Produce a raw, non-normalized file for this cell line

        
        print('Producing scored BED file :',output_bed_path)
        
        
        # start_prod = time.time()
        # er.produce_result_file(all_matrices, output_bed_path,
        #     model, get_matrix, parameters, datasets_clean, cl_tfs)
        # end_prod = time.time()
        # total_time_prod = end_prod-start_prod
        # print('Processed BED file produced in',str(total_time_prod),'seconds.')
        





        # import importlib
        # importlib.reload(er)

        # TODO Make number of threads a parameter

        # TODO it's actually number of processes and optimal might not be 7 ! Try several !

        import functools

        start_prod = time.time()
        er.produce_result_file(all_matrices[0:1000], output_bed_path+"_MULTITHREAD",
            get_matrix, parameters, prepare.prepare_model_with_parameters,
            datasets_clean, cl_tfs,
            nb_threads = 1, save_model_path = save_model_path)
        end_prod = time.time()
        total_time_prod = end_prod-start_prod
        print('Multithread done in',str(total_time_prod),'seconds.')


        """
        WARNING : TOO LARGE MODELS MAY CONSUME TOO MUCH MEMORY WHEN MULTITHREADED ???
        """


        """
        APPARENTLY TENSORFLOW 2 IS MUCH SLOWER, SO GO BACK TO TENSORFLOW 1 AND USE MULTITHREADING
        FOR BEST SPEED. SO I DID WELL TO KEEP BOTH CODES, mention we can us both TF 1 or 2 in the readme

        WAIT. TF2 USES ALL THREADS BY DEFAULT AND IS SLOWER ? RAM SHENANIGANS MAYBE ? TRY ON MY VM !
        I could put it on the readme along with "as of April 2020, we recommend..."
        """


        # start_prod = time.time()
        # er.produce_result_file(all_matrices[0:1000], output_bed_path+"_SINGLETHREAD",
        #     get_matrix, parameters, datasets_clean, cl_tfs,
        #     nb_threads = 1, save_model_path = save_model_path,
        #     call_to_prepare_model = partial_call_to_prepare_model)
        # end_prod = time.time()
        # total_time_prod = end_prod-start_prod
        # print('Single thread done in',str(total_time_prod),'seconds.')
























        # sys.exit()
        # """
        # FOR NOW STOP HERE




        # I AM TRYING MULTITHREADING.
        # """












































        # Put mean score for doublons
        utils.print_merge_doublons(bedfilepath = output_bed_path, outputpath = output_bed_merged)


        # --------- Normalization ----------


        # For reference, get the scores per tf and per dataset for the RAW data, before normalization
        scores_by_tf_df_raw, scores_by_dataset_df_raw = utils.normalize_result_file_score_by_tf(output_bed_merged,
            cl_name = parameters['cell_line'], outfilepath = output_bed_path_normalized_poub)





        """
        CAREFUL, AT TIME OF WRITING THIS I OVERWRITE scores_by_tf_df AND scores_by_dataset_df LATER. DECIDE WHICH ONE TO OUTPUT.
        The last one I think ? After my normalization ?

        Or I can just keep both and print both...
        I have a fig about this now
        """



        ### First normalize by correlation group

        # Estimate corr group scaling factors
        corr_group_scaling_factor_dict = er.estimate_corr_group_normalization_factors(model = model,
            all_datasets = datasets_clean, all_tfs = cl_tfs, list_of_many_crms = list_of_many_crms,
            crm_length = parameters['pad_to'], squish_factor = parameters["squish_factor"],
            outfilepath = './data/output/diagnostic/'+parameters['cell_line']+'/'+"normalization_factors.tsv")

        # Apply them
        utils.normalize_result_file_with_coefs_dict(output_bed_merged,
            corr_group_scaling_factor_dict, cl_name = parameters['cell_line'],
            outfilepath = output_path_corr_group_normalized)


        ### Then, finally normalize the score by TF, under the assumption that no TF is better than another.
        scores_by_tf_df, scores_by_dataset_df = utils.normalize_result_file_score_by_tf(output_path_corr_group_normalized,
            cl_name = parameters['cell_line'], outfilepath = output_bed_path_final)

        # This will be used to quantify the effect of the normalization



        # A new result file labeled "FINAL" has been produced.
        print('Processing complete.')
        plt.close('all') # Close all figures


        # ----------------------------- Diagnostic plots ----------------------------- #

        if parameters['perform_real_data_diagnosis']:

            print('Performing diagnostic plots...')

            # Only if not artificial data.
            # TODO : those could be useful in artificial as well


            # -------- Median of score by TF

            # By TF
            fig, ax = plt.subplots(figsize=(10, 8))
            sub_df = scores_by_tf_df.loc[:,['count','50pc']]
            sub_df.plot('count', '50pc', kind='scatter', ax=ax, s=24, linewidth=0) ; ax.grid()
            for k, v in sub_df.iterrows(): ax.annotate(k, v,xytext=(10,-5), textcoords='offset points')
            plt.savefig(plot_output_path+'scores_median_by_tf_after_corrgroup_normalization_only.pdf')

            # By dataset
            fig, ax = plt.subplots(figsize=(10, 8))
            sub_df = scores_by_dataset_df.loc[:,['count','50pc']]
            sub_df.plot('count', '50pc', kind='scatter', ax=ax, s=24, linewidth=0) ; ax.grid()
            for k, v in sub_df.iterrows(): ax.annotate(k, v,xytext=(10,-5), textcoords='offset points')
            plt.savefig(plot_output_path+'scores_median_by_dataset_after_corrgroup_normalization_only.pdf')





























            ## Quantify effect of normalization

            sub_raw = scores_by_tf_df_raw.loc[:,['mean']]
            sub_raw.columns = ['mean_raw_before_corrgroup_normalization']
            sub_after = scores_by_tf_df.loc[:,['mean']]
            sub_after.columns = ['mean_after_corrgroup_normalization']
            df_cur = sub_raw.join(sub_after)
            df_cur = df_cur.reset_index()
            #df_cur = pd.merge(sub_raw, sub_after, on="tf")
            #df_cur = sub_raw.to_frame(name = 'mean_raw_before_norm').join(sub_after.to_frame(name='mean_after_norm'))
            df_cur_melted = df_cur.melt(id_vars=['tf'], value_vars=['mean_raw_before_corrgroup_normalization','mean_after_corrgroup_normalization'])
            p = ggplot(df_cur_melted, aes(x = "tf", y= "value", fill = "variable")) + geom_bar(stat="identity", width=.7, position = "dodge") + theme(legend_position = "top")
            p.save(plot_output_path+"scores_mean_by_tf_before_and_after_corrgroup_normalization.pdf", height=8, width=16, units = 'in', dpi=400)

            # TODO Same for datasets














            plt.close('all') # Close all figures













            # ----------------------- Scores per CRM  ------------------------- #
            # Computing score distribution per number of peaks in CRMs

            # VERY IMPORTANT PLOTS

            

            # The CRM file path
            CRM_FILE = parameters["CRM_FILE"] 




            score_distrib, avg_score_crm, max_score_crm = er.plot_score_per_crm_density(output_bed_path, CRM_FILE)

            score_distrib.save(plot_output_path+'score_distribution_raw.pdf')
            avg_score_crm.save(plot_output_path+'average_score_per_crm_density_raw.pdf')
            max_score_crm.save(plot_output_path+'max_score_per_crm_density_raw.pdf')



            # REDO THIS ON NORMALIZED FINAL FILE
            # so after tf normalization
            print("... in the final normalized file.")
            score_distrib_tfnorm, avg_score_crm_tfnorm, max_score_crm_tfnorm = er.plot_score_per_crm_density(output_bed_path_final, CRM_FILE)
            score_distrib_tfnorm.save(plot_output_path+'score_distribution_after_final_normalization.pdf')
            avg_score_crm_tfnorm.save(plot_output_path+'average_score_per_crm_density_after_final_normalization.pdf')
            max_score_crm_tfnorm.save(plot_output_path+'max_score_per_crm_density_after_final_normalization.pdf')


            plt.close('all') # Close all figures

            # ----------------- Scores when known cofactors (or known non-cofactors) are present

            print("Retrieving scores for specified TF pairs and estimating correlation groups...")




            # NOTE We work on FINAL NORMALIZED scores after both normalizations. TODO SAY SO IN PAPER FIGURES AND IN DEBUG MESSAGES and/or comments here !!!!!

            # TODO CHECK WHICH FILE I USE PRECISELY



            tfs_to_plot = parameters['tf_pairs']

            tf_alone_both_output_path = plot_output_path + "tf_pairs_alone_both/"
            if not os.path.exists(tf_alone_both_output_path): os.makedirs(tf_alone_both_output_path)

            for pair in tfs_to_plot:
                try:
                # TODO CAREFUL ABOUT CASE !!


                    pair = ["sfmbt1","e2f1"]

                    tf1, tf2 = pair
                    
                    
                    
                    
                    # RE-enable this
                    #p, _ = er.get_scores_whether_copresent(tf1, tf2, output_bed_path_final, CRM_FILE)
                    
                    
                    
                    
                    
                    p, _ = er.get_scores_whether_copresent(tf1, tf2, output_bed_merged, CRM_FILE)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    p.save(tf_alone_both_output_path+"tf_alone_both_"+tf1+"_"+tf2+".pdf")


                    plt.close('all') # Close all figures
                except:
                    print("Error fetching the pair : "+str(pair))
                    print("Ignoring.")



            # ----- Correlation group estimation
            # Try to estimate the correlation groups learned by the model for certain select sources (TF+dataset pair) by looking at what kind of phantoms are added by the model

            corrgroup_estim_output_path = plot_output_path + "estimated_corr_groups/"
            if not os.path.exists(corrgroup_estim_output_path): os.makedirs(corrgroup_estim_output_path)

            for combi in parameters['estimate_corr_group_for']:
                try:
                    dataset, tf = combi
                    
                    output_path_estimation = corrgroup_estim_output_path + "estimated_corr_group_for_"+dataset+"_"+tf+".pdf"
                    
                    fig, _ = er.estimate_corr_group_for_combi(dataset, tf,
                        all_datasets = datasets_clean, all_tfs = cl_tfs, model = model,
                        crm_length = parameters['pad_to'], squish_factor = parameters["squish_factor"])
                    fig.get_figure().savefig(output_path_estimation)
                     
                    plt.close('all') # Close all figures       
                except:
                    print("Error estimating for : "+str(combi))
                    print("Ignoring.")


print("Data succesfully processed !")