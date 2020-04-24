# System
import os, sys, yaml, time, random
from functools import partial

# Data
import numpy as np
import pandas as pd
import scipy

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_violin, geom_boxplot, position_dodge, scale_fill_grey, geom_bar, theme

## Custom libraries
import lib.prepare as prepare       # Parameters and shared functions
import lib.data_read as dr          # Process data produced by Makefile
import lib.result_eval as er        # Result production and evaluation
import lib.utils as utils           # Miscellaneous


############################# PARAMETERS AND SETUP #############################

root_path = os.path.dirname(os.path.realpath(__file__))
parameters = yaml.load(open(root_path+'/parameters.yaml').read(), Loader = yaml.Loader)

### Set random seeds
SEED = parameters["random_seed"]
os.environ["PYTHONHASHSEED"]=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Tensorflow random seed should have no impact now since the model is already trained.

## Reading corresponding keys. See the documentation of get_data_indexes() for more.
crmtf_dict, datasets, crmid, datapermatrix, peaks_per_dataset, cl_tfs = dr.get_data_indexes(parameters['cell_line'], root_path)

# Get dataset, TF, and matrices ID in correct order
datasets_clean, cl_tfs, all_matrices = prepare.get_indexes(parameters, crmid, datasets, cl_tfs)

# Since those dictionaries are fixed for a cell line, prepare a partial call
get_matrix = partial(dr.extract_matrix,
    all_tfs = cl_tfs,
    cell_line = parameters['cell_line'], cl_crm_tf_dict = crmtf_dict,
    cl_datasets = datasets_clean, crm_coordinates = crmid,
    datapath = root_path+'/data/input/sorted_intersect/')

plot_output_path = prepare.get_plot_output_path(parameters) # Plot output path (different for artificial data)

print('Parameters loaded.')


"""
NOTE To ensure multiprocessing works, it is imperative that keras or tensorflow
are not imported in the main process.py code, which would initiate a session.
This must be done only by the subprocesses !
"""


################################################################################
################################## REAL DATA ###################################
################################################################################


# Path for saving the model later
# Also used in processing the file for multithreading !
save_model_path = prepare.get_save_model_path(parameters, root_path)


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


        

        ## Output BED filepaths
        # TODO Clean those up a little !

        # Source directory
        root_output_bed_path = root_path + '/data/output/bed/' + parameters['cell_line']
        if not os.path.exists(root_output_bed_path): os.makedirs(root_output_bed_path)

        
        output_bed_path = root_output_bed_path + ".bed"
        output_bed_merged = root_output_bed_path + "_merged_doublons.bed"

        output_bed_path_normalized_poub = root_output_bed_path + "_raw_normalized_by_tf_DEBUG.bed"  

        output_path_corr_group_normalized = root_output_bed_path + "_merged_doublons_normalized_corr_group.bed"
        output_bed_path_final = root_output_bed_path + "_FINAL_merged_doublons_normalized_corr_group_normalized_by_tf.bed"



        ### Raw file production
        # Produce a raw, non-normalized file for this cell line

        
        print('Producing scored BED file :',output_bed_path)
        
        # TODO number of threads is actually number of processes and optimal might not be 7 ! Try several !

        start_prod = time.time()
        er.produce_result_file(all_matrices[0:1000], output_bed_path+"_MULTITHREAD",
            get_matrix, parameters, prepare.prepare_model_with_parameters,
            datasets_clean, cl_tfs,
            nb_threads = parameters["nb_workers_produce_file"], save_model_path = save_model_path)
        end_prod = time.time()
        total_time_prod = end_prod-start_prod
        print('Multithread done in',str(total_time_prod),'seconds.')


        """
        TODO NOTE THIS IN PAPER MAYBE AND HERE AS NOTES
        WARNING : TOO LARGE MODELS MAY CONSUME TOO MUCH MEMORY WHEN MULTITHREADED ???


        APPARENTLY TENSORFLOW 2 IS MUCH SLOWER, SO GO BACK TO TENSORFLOW 1 AND USE MULTITHREADING
        FOR BEST SPEED. SO I DID WELL TO KEEP BOTH CODES, mention we can us both TF 1 or 2 in the readme

        WAIT. TF2 USES ALL THREADS BY DEFAULT AND IS SLOWER ? RAM SHENANIGANS MAYBE ? TRY ON MY VM !
        I could put it on the readme along with "as of April 2020, we recommend..."
        """

        # Put mean score for doublons of peaks
        utils.print_merge_doublons(bedfilepath = output_bed_path, outputpath = output_bed_merged)


        # --------------------------- Normalization -------------------------- #

        # For reference, get the scores per TF and per dataset for the *raw* data, before normalization
        scores_by_tf_df_raw, scores_by_dataset_df_raw = utils.normalize_result_file_score_by_tf(output_bed_merged,
            cl_name = parameters['cell_line'], outfilepath = output_bed_path_normalized_poub,
            was_corr_group_normalized_before_header = False,
            center_around = 500) # TODO Use 500 here !?


        ### First normalize by correlation group

        # Apply the corr group scaling factors estimated in training
        utils.normalize_result_file_with_coefs_dict(output_bed_merged,
            scaling_factor_dict_filepath = './data/output/diagnostic/'+parameters['cell_line']+'/'+"normalization_factors.tsv",
            cl_name = parameters['cell_line'],
            outfilepath = output_path_corr_group_normalized)

        ### Then, finally normalize the score by TF, under the assumption that no TF is better than another.
        scores_by_tf_df, scores_by_dataset_df = utils.normalize_result_file_score_by_tf(output_path_corr_group_normalized,
            cl_name = parameters['cell_line'], outfilepath = output_bed_path_final,
            was_corr_group_normalized_before_header = True,
            center_around = parameters['tf_normalization_center_around'])
            # NOTE Center around more than 500 so, after corr group normalization,
            # we have a larger palette to see the "weakers than typical", not the "better than typical".
            # Also, only here do we clip at 0-1000, not before

        # A new result file labeled "FINAL" has been produced.
        print('Processing complete.')
        plt.close('all') # Close all figures


        # ----------------------------- Diagnostic plots ----------------------------- #

        if parameters['perform_real_data_diagnosis']:

            print('Performing diagnostic plots...')

            # Only if not artificial data.
            # TODO : those could be useful in artificial as well
            # As such : rename "perform_real_data_diagnosis" to "perform_processed_data_diagnosis"

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
            df_cur_melted = df_cur.melt(id_vars=['tf'], value_vars=['mean_raw_before_corrgroup_normalization','mean_after_corrgroup_normalization'])
            p = ggplot(df_cur_melted, aes(x = "tf", y= "value", fill = "variable")) + geom_bar(stat="identity", width=.7, position = "dodge") + theme(legend_position = "top")
            p.save(plot_output_path+"scores_mean_by_tf_before_and_after_corrgroup_normalization.pdf", height=8, width=16, units = 'in', dpi=400)
            # TODO Same for datasets

            plt.close('all') # Close all figures






            # ----------------------- Scores per CRM  ------------------------- #
            # Computing score distribution per number of peaks in CRMs

            # Both on raw file and after final normalization

            score_distrib, avg_score_crm, max_score_crm = er.plot_score_per_crm_density(output_bed_path, parameters["CRM_FILE"])
            score_distrib.save(plot_output_path+'score_distribution_raw.pdf')
            avg_score_crm.save(plot_output_path+'average_score_per_crm_density_raw.pdf')
            max_score_crm.save(plot_output_path+'max_score_per_crm_density_raw.pdf')

            score_distrib_tfnorm, avg_score_crm_tfnorm, max_score_crm_tfnorm = er.plot_score_per_crm_density(output_bed_path_final, parameters["CRM_FILE"])
            score_distrib_tfnorm.save(plot_output_path+'score_distribution_after_final_normalization.pdf')
            avg_score_crm_tfnorm.save(plot_output_path+'average_score_per_crm_density_after_final_normalization.pdf')
            max_score_crm_tfnorm.save(plot_output_path+'max_score_per_crm_density_after_final_normalization.pdf')


            plt.close('all') # Close all figures

            # ----------------- Scores when known cofactors (or known non-cofactors) are present

            print("Retrieving scores for specified TF pairs and estimating correlation groups...")


            # NOTE We work on FINAL NORMALIZED scores after both normalizations. TODO SAY SO IN PAPER FIGURES AND IN DEBUG MESSAGES and/or comments here !!!!!


            tfs_to_plot = parameters['tf_pairs']

            tf_alone_both_output_path = plot_output_path + "tf_pairs_alone_both/"
            if not os.path.exists(tf_alone_both_output_path): os.makedirs(tf_alone_both_output_path)

            for pair in tfs_to_plot:
                try:
                    tf1, tf2 = pair
        
                    p, _ = er.get_scores_whether_copresent(tf1, tf2, output_bed_path_final, parameters["CRM_FILE"])      
                    #p, _ = er.get_scores_whether_copresent(tf1, tf2, output_bed_merged, CRM_FILE)  # DEBUG before corr group normalization
                    
                    p.save(tf_alone_both_output_path+"tf_alone_both_"+tf1+"_"+tf2+".pdf")

                    plt.close('all') # Close all figures
                except:
                    print("Error fetching the pair : "+str(pair))
                    print("Ignoring.")
            


            # ----- Correlation group estimation
            # Try to estimate the correlation groups learned by the model for 
            # certain select sources (ie. TF+dataset paisr) by looking at what
            # kind of phantoms are added by the model

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