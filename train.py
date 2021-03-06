# System
import os, sys, yaml, time, random, copy
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
import lib.model_atypeak as cp      # Autoencoder functions
import lib.data_read as dr          # Process data produced by Makefile
import lib.artificial_data as ad    # Trivial data for control
import lib.result_eval as er        # Result production and evaluation
import lib.utils as utils           # Miscellaneous

############################# PARAMETERS AND SETUP #############################

root_path = os.path.dirname(os.path.realpath(__file__))
parameters = yaml.load(open(root_path+'/parameters.yaml').read(), Loader = yaml.Loader)

### Set random seeds
SEED = parameters["random_seed"]
# Python internals
os.environ["PYTHONHASHSEED"]=str(SEED)
random.seed(SEED)
np.random.seed(SEED) # For Numpy, which will also impact Keras


## Prepare Tensorflow
# Set random seed and session parameters
prepare.configure_tensorflow_session(SEED, parameters["disable_tensorflow_warnings"])
# Import Keras again, only once TF session configured
import keras
import keras.backend as K


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

print('Parameters loaded.')

# NOTE To compute weights for the loss : based on the `crmtf_dict` object and 
# also on the `datapermatrix` and `peaks_per_dataset` objects, we know the 
# number of elements for each combination of TF/dataset.

# Plot output path (different for artificial data)
plot_output_path = prepare.get_plot_output_path(parameters)



############################### DATA GENERATOR #################################
# You may use either the artificial data generator or the true data.
# Artificial data is mainly used for calibrations and demonstrations.

def produce_data_generator(parameters, matrices_id, get_matrix_func): # remake due to threading conflicts
    # Artificial data
    if parameters['use_artificial_data'] :
        generator = ad.generator_fake(region_length = parameters['pad_to'],
                                        nb_datasets = parameters['artificial_nb_datasets'], nb_tfs=parameters['artificial_nb_tfs'],
                                        squish_factor = parameters['squish_factor'], ones_only=parameters['artificial_ones_only'],
                                        watermark_prob = parameters['artificial_watermark_prob'],
                                        overlapping_groups = parameters['artificial_overlapping_groups'],
                                        tfgroup_split = parameters['artificial_tfgroup_split'],
                                        this_many_groups_of_4_tfs = parameters["artificial_this_many_groups_of_4_tfs"],
                                        split_tfs_into_this_many_groups = parameters["artificial_split_tfs_into_this_many_groups"],
                                        crumb = None)
    
    # Real data
    if not parameters['use_artificial_data'] :
        generator = cp.generator_unsparse(all_matrices, parameters["nn_batch_size"],
                        get_matrix, parameters["pad_to"], parameters["crumb"], parameters["squish_factor"])
   
    return generator

## Now prepare the generator
train_generator = produce_data_generator(parameters, all_matrices, get_matrix)



################################ MODEL #########################################
# Load parameters for the AutoEncoder (AE) from the config file, add missing
# ones and create the model

# Prepare the model
model = prepare.prepare_model_with_parameters(parameters,
    nb_datasets_model = len(datasets_clean), nb_tfs_model = len(cl_tfs),
    root_path = root_path) # Print model parameters


print("-- Dimensions : "+str(parameters["pad_to"])+ 'bp x '+str(len(datasets_clean))+' datasets x '+str(len(cl_tfs))+' TFs.')
print('Model created.')


# Control print of parameters
print("-- PARAMETERS detailed :")
for k, v in parameters.items(): print('\t',k,'=', v)



# ------------------------------ Training ------------------------------------ #
# Train only if not loading a saved model


# Path for saving the model later
# Also used in processing the file for multithreading !
save_model_path = prepare.get_save_model_path(parameters, root_path)

def train_model(model, parameters, train_generator, save_model_path):
    """
    A wrapper function that will train a given model given the current parameters.
    Non-pure, since it depends on the rest of the code. Would be useful later for grid search.
    """

    # NOTE Keras does not like reloading custom optimizers or losses. To get around 
    # this problem, we re-create the model first, then load ONLY the weights
    
    if parameters['load_saved_model']:
        try:
            model.load_weights(save_model_path+".h5") 
        except OSError: raise FileNotFoundError("load_saved_model is True, but trained model was not found at "+save_model_path)

        print('Loaded a saved model.')

    else :
        # Callback for early stopping : don't stop as soon as exact floor is reached (patience)
        es = cp.EarlyStoppingByLossVal(monitor='loss', value=parameters['nn_early_stop_loss_absolute'], patience=2)    
        
        # Stop in any case after several epochs with no improvement
        es2 = keras.callbacks.EarlyStopping(monitor='loss',
            patience = parameters['nn_early_stop_patience'], min_delta = parameters['nn_early_stop_min_delta'],
            restore_best_weights = True)  # Try to combat loss spikes and restore the best weights
        # NOTE To combat overfitting, we use larger batches and lower learning rates 

        """
        # DEBUG : save at every epoch
        class CustomSaver(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                self.model.save("DEBUG_model_at_epoch_{}.hd5".format(epoch))
        saver = CustomSaver()
        """

        print('Beginning training.')
        print('This can be long, depending on your hardware, from minutes to hours.')

        start = time.time()
        model.fit_generator(train_generator, verbose=1,
            steps_per_epoch = parameters["nn_batches_per_epoch"],
            epochs = parameters["nn_number_of_epochs"],
            #epochs = 6,
            callbacks = [es, es2],
            max_queue_size = 10) # Max queue size of 10 to prevent memory leak
        end = time.time()

        total_time = end-start
        print('Training of the model completed in '+str(total_time)+' seconds.')

        # Save trained model and save the loss as text
        loss_history = model.history.history["loss"]
        model.save(save_model_path+'.h5')
        np.savetxt(save_model_path+"_loss_history_by_epoch.txt", np.array(loss_history), delimiter=",")
        print('Model saved.')


        ## Check model was correctly saved and predictions are the same
        tmp_model = prepare.prepare_model_with_parameters(parameters, len(datasets_clean), len(cl_tfs))
        tmp_model.load_weights(save_model_path+".h5") # Do not forget the ".h5"

        for _ in range(3): # Try this a few times
            before_batch = next(train_generator)[0]
            before_raw = np.copy(before_batch[0,:,:,:,0])
            prediction = model.predict(before_raw[np.newaxis,...,np.newaxis])[0,:,:,:,0]
            prediction_tmp = tmp_model.predict(before_raw[np.newaxis,...,np.newaxis])[0,:,:,:,0]

            try: np.testing.assert_allclose(prediction, prediction_tmp)
            except: raise AssertionError("Model produced different results when reloading. Check your Keras and TF installation.")
        del tmp_model
        del before_batch


    return model

# Train the model
model = train_model(model, parameters, train_generator, save_model_path)



################################################################################
################################# DIAGNOSTIC ###################################
################################################################################

# Figure size.
eval_figsize_small = (5,5)
eval_figsize_large = (8,5)

# Get some samples (3D tensor representations of CRMs)
# Produce a few batches (this is a parameter). I like to keep it around 2000*48
# This is used MANY times in the code, for Q-score, many diagnostics, and
# abundance, normalization, etc.
# NOTE To prevent an issue with threading, recreate a new train_generator
print("Generating many CRMs for diagnosis. This may be long...")
train_generator = produce_data_generator(parameters, all_matrices, get_matrix)

start_genlist = time.time()
list_of_many_crms = er.get_some_crms(train_generator,
    nb_of_batches_to_generate = parameters['nb_batches_generator_for_diag_list_of_many_crms'])
stop_genlist = time.time()
print('List of many CRMs for evaluation collated in '+str(stop_genlist-start_genlist)+' seconds.')

# TODO The RAM footprint of list_of_many_crms can be massive for larger cell lines. We currently
# cast to boolean to save RAM because we round off, but later replace it with a generator.



## ----- Normalization
# Estimate the scaling factors to normalize by corr group
# See source code for details. In summary, all groups learned by the models are not of the same size, might have abundance biases, and may overlap
start_norm = time.time()
corr_group_scaling_factor_dict = er.estimate_corr_group_normalization_factors(model = model,
    all_datasets = datasets_clean, all_tfs = cl_tfs, list_of_many_crms = list_of_many_crms,
    crm_length = parameters['pad_to'], squish_factor = parameters["squish_factor"],
    outfilepath = './data/output/diagnostic/'+parameters['cell_line']+'/'+"normalization_factors.tsv")
stop_norm = time.time()
# Needed in true file production, hence it's not optional.
print("Correlation group normalization factors estimated in "+str(stop_norm-start_norm)+" seconds.")


# Perform some diagnosis on the model. Only if the `perform_model_diagnosis` 
# flag in the parameters is True.
# Notably, it could be redundant when reloading a model.
if parameters['perform_model_diagnosis']:

    # ------------------------------ Evaluation ------------------------------ #
    # Plot some examples of CRMs, their reconstructions and anomaly scores.

    # Where to save the figures 
    example_output_path = plot_output_path + "crm_examples/"
    if not os.path.exists(example_output_path): os.makedirs(example_output_path)
    
    i = 0

    for _ in range(parameters["example_nb_batches"]):

        # Data
        before_batch = next(train_generator)[0]

        for n in range(len(before_batch)):

            before_raw = np.copy(before_batch[n,:,:,:,0])
            before = np.around(before_raw-0.11) # Remove crumbing if applicable. TODO Make this more rigorous
                         
            prediction = model.predict(before_raw[np.newaxis,...,np.newaxis])[0,:,:,:,0]

            # 2D - max along region axis
            before_2d = np.max(before_raw, axis=0)
            plt.figure(figsize=eval_figsize_small); before_2d_plot = sns.heatmap(np.transpose(before_2d), cmap = 'Blues', xticklabels = datasets_clean, yticklabels = cl_tfs).get_figure()
            prediction_2d = np.max(prediction, axis=0)
            
            plt.figure(figsize=eval_figsize_small); prediction_2d_plot = sns.heatmap(np.transpose(prediction_2d), annot = True, cmap = 'Greens', fmt='.2f', xticklabels = datasets_clean, yticklabels = cl_tfs).get_figure()
            plt.figure(figsize=eval_figsize_small); prediction_2d_plot_noannot = sns.heatmap(np.transpose(prediction_2d), annot = False, cmap = 'Greens', fmt='.2f', xticklabels = datasets_clean, yticklabels = cl_tfs).get_figure()

            anomaly_matrix = er.anomaly(before, prediction)
            anomaly_plot = utils.plot_3d_matrix(anomaly_matrix, figsize=eval_figsize_large) # Normal in blue, anomalous in red

            # Export plots
            before_2d_plot.tight_layout()
            before_2d_plot.savefig(example_output_path + "example_crm_before_2dmax_"+str(i)+".pdf")
            prediction_2d_plot.tight_layout()
            prediction_2d_plot.savefig(example_output_path + "example_crm_rebuilt_2dmax_"+str(i)+".pdf")  
            prediction_2d_plot_noannot.tight_layout()
            prediction_2d_plot_noannot.savefig(example_output_path + "example_crm_rebuilt_2dmax_no_annot_"+str(i)+".pdf")

            anomaly_plot.savefig(example_output_path + "example_crm_anomaly_"+str(i)+".pdf")

            plt.close('all') # Close all figures

            i = i+1 # Increment counter



    # ----------------------- Abundance evaluation --------------------------- #

    print("Abundance evaluation. Can be long...")

    summed_anomalies = []
    summed_befores = []

    # Should use many samples to compensate for batch effects
    for before in list_of_many_crms:

        # Data
        summed_before = np.max(before, axis = 0)
        summed_befores += [summed_before]

        # Prediction
        prediction = model.predict(before[np.newaxis,...,np.newaxis])[0,:,:,:,0]
        anomaly_matrix = er.anomaly(before, prediction)

        summed_anomaly = np.max(anomaly_matrix, axis = 0)
        summed_anomalies += [summed_anomaly]


    # Do not consider the zeros where a peak was not placed, only the peaks and the rebuilt peaks
    anomaly_values = np.ma.masked_equal(summed_anomalies, 0)
    median_anomaly_values = np.ma.median(anomaly_values, axis=0)
    plt.figure(); median_anomaly_values_plot = sns.heatmap(median_anomaly_values.transpose(), annot = True, fmt='.2f', xticklabels = datasets_clean, yticklabels = cl_tfs, cmap = 'Purples').get_figure()

    # Save this as a diagnostic plot
    median_anomaly_values_plot.tight_layout()
    median_anomaly_values_plot.savefig(plot_output_path+'median_anomaly_score.pdf')
    
    plt.close('all') # Close all figures


    # ------------------------------ Q-score evaluation --------------------------- #
    # For each {A,B} pair of sources (TR+dataset pairs), the model should give 
    # a different score to A alone than when B is present *only* when there is
    # actually a correlation. If not, if either learned too little (not identified)
    # the group) or too much (too precise). Conversely if there is no correlation
    # the presence of one should have no impact on the other.
    # Same reasoning for the presence of phantoms for A when B is present, should
    # be true only if they correlate.

    print("Beginning Q-score evaluation. Can be long...")

    qstart = time.time()
    q, qscore_plot, corr_plot, posvar_x_res_plot, negvar_x_res_plot = er.calculate_q_score(model,
        list_of_many_befores = list_of_many_crms,
        all_datasets_names = datasets_clean, all_tf_names = cl_tfs)
    qend = time.time()


    qtotal_time = qend-qstart
    print('Q-score evaluation completed in '+str(qtotal_time)+' seconds.')

    # Final Q-score (total sum)
    print("-- Total Q-score of the model (lower is better) : "+ str(np.sum(np.sum(q))))

    # Those plots give respectively the Q-score contribution of each pair 
    # (lower is better), the true correlation matrix for comparison, and
    # the third plot says whether the presence of both results in a change in 
    # score and should "look like" the correlation plot.
    qscore_output_path = plot_output_path + "q_score/"
    if not os.path.exists(qscore_output_path): os.makedirs(qscore_output_path)

    qscore_plot.savefig(qscore_output_path+'qscore_plot.pdf')
    corr_plot.savefig(qscore_output_path+'correlation_matrix_dims.pdf')
    posvar_x_res_plot.savefig(qscore_output_path+'posvar_when_both_present.pdf')
    negvar_x_res_plot.savefig(qscore_output_path+'negvar_phantoms_when_present.pdf')

    np.savetxt(qscore_output_path+'q_score.tsv', q, delimiter='\t')

    print("Q-score evaluation results saved.")

    plt.close('all') # Close all figures
    
    
 



    # ---------------------------- Visualize filters ------------------------- #
    # Useful when calibrating the parameters and making sure the model was
    # learning, but otherwise I found it not particularly useful.
    kernel_output_path = plot_output_path + "conv_kernels/"
    if not os.path.exists(kernel_output_path): os.makedirs(kernel_output_path)

    # Datasets
    w = model.layers[2].get_weights()[0]
    for filter_nb in range(w.shape[-1]):
        plt.figure(figsize=eval_figsize_small); dfp = sns.heatmap(w[:,:,0,0,filter_nb])
        #utils.plot_3d_matrix(w[:,:,:,0,filter_nb], figsize=(6,4))
        dfp.get_figure().savefig(kernel_output_path + "conv_kernel_datasets_"+str(filter_nb)+".pdf")
        plt.close('all') # Close all figures

    # TFs
    w = model.layers[5].get_weights()[0]
    for filter_nb in range(w.shape[-1]):
        #utils.plot_3d_matrix(w[:,0,:,:,filter_nb], figsize=(6,4))
        plt.figure(figsize=eval_figsize_small); tfp = sns.heatmap(w[0,0,:,:,filter_nb]) # 2D version only at the first X (often the same anyways)
        tfp.get_figure().savefig(kernel_output_path + "conv_kernel_tf_x0_"+str(filter_nb)+".pdf")
        plt.close('all') # Close all figures





    
    # --------------------- Visualize encoded representation ----------------- #
    # I added some Y and Z blur, to smooth it a little. Keep the blur minimal, makes no sense otherwise. 

    # Verify if it was disabled
    if parameters['urexample_steps'] != 0 :

        ENCODED_LAYER_NUMBER = 15
        # To check this is the correct number
        if not (model.layers[ENCODED_LAYER_NUMBER].name == 'encoded'):
            print("ENCODED_LAYER_NUMBER in the code not set to the encoded dimension. Urexamples will be on a different dimension.")

        print("Computing ur-examples. Can be long...")
        start_ur = time.time()
        urexamples = cp.compute_max_activating_example_across_layer(model, random_state = 42,
                                    selected_layer = ENCODED_LAYER_NUMBER, output_of_layer_select_this_dim = 2,
                                    learning_rate = 1, nb_steps_gradient_ascent = parameters['urexample_steps'], # Should be around 50 steps by default
                                    blurStdX = 0.2, blurStdY = 1E-2,  blurStdZ = 1E-2, blurEvery = 5,
                                    debug_print = False)
        stop_ur = time.time()
        print("Ur-examples computed in "+str(stop_ur-start_ur)+" seconds.")
        
        urexample_output_path = plot_output_path + "urexamples_encoded_dim/"
        if not os.path.exists(urexample_output_path): os.makedirs(urexample_output_path)

        for exid in range(len(urexamples)):
            ex = urexamples[exid]
            ex = ex[...,0]
            x = np.mean(ex, axis = 0)
            plt.figure(figsize=eval_figsize_small); urfig = sns.heatmap(np.transpose(x), cmap ='RdBu_r', center = 0, xticklabels = datasets_clean, yticklabels = cl_tfs).get_figure()
            urfig.tight_layout()
            urfig.savefig(urexample_output_path + "urexample_dim_"+str(exid)+".pdf")
            plt.close('all') # Close all figures
    
        """
        # Get an encoded representation for comparison (from the latest `before`, from above)
        tmp = cp.get_encoded_representation(before[np.newaxis,...,np.newaxis], model)
        tmpd = np.transpose(tmp[0]**2)
        plt.figure(figsize=eval_figsize_large); sns.heatmap(tmpd)
        utils.plot_3d_matrix(before)
        """




    # --------------------- Correlation group estimation --------------------- #
    # Try to estimate the correlation groups learned by the model for certain 
    # select sources (ie. TF + dataset pairs) by looking at what kind of
    # phantoms are added by the model

    corrgroup_estim_output_path = plot_output_path + "estimated_corr_groups/"
    if not os.path.exists(corrgroup_estim_output_path): os.makedirs(corrgroup_estim_output_path)

    print("Estimating correlation groups...")

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
            print("Error estimating for : "+str(combi)+", ignoring.")




    # ----------------------- Proof on artificial data ----------------------- #
    # Plot score of artificial peaks depneding on type (noise, stack, ...)

    if parameters['use_artificial_data'] :

        # Prepare a generator of artificial data that returns the peaks separately
        ad_partial = partial(ad.make_a_fake_matrix, region_length = parameters['pad_to'],
                        nb_datasets = parameters['artificial_nb_datasets'], nb_tfs = parameters['artificial_nb_tfs'],
                        signal = True, noise=True, ones_only = parameters['artificial_ones_only'], return_separately = True)


        # This part should be quick
        arti_start = time.time()
        df, separated_peaks = er.proof_artificial(model, ad_partial,
            region_length = parameters['pad_to'], nb_datasets = parameters['artificial_nb_datasets'], nb_tfs = parameters['artificial_nb_tfs'],
            n_iter = 500, squish_factor = parameters['squish_factor'])
        arti_end = time.time()
        print('Artificial data generalisation completed in '+str(arti_end-arti_start)+' s')


        # The plots
        a = ggplot(df, aes(x="type", y="rebuilt_value", fill="tf_group"))
        a1 = a + geom_violin(position=position_dodge(1), width=1)
        a2 = a + geom_boxplot(position=position_dodge(1), width=0.5)
        b = ggplot(df, aes(x="brothers", y="rebuilt_value", group="brothers")) + scale_fill_grey() + geom_boxplot(width = 0.4)

        a2.save(filename = plot_output_path+'artifical_data_systematisation_value_per_type.png', height=10, width=14, units = 'in', dpi=400, verbose = False)
        b.save(filename = plot_output_path+'artifical_data_systematisation_value_per_brothers.png', height=10, width=14, units = 'in', dpi=400, verbose = False)

        plt.close('all') # Close all figures



    # -------------- Data information plots for result checking -------------- #
    # Diagnostic plots on the data, such as correlation coefficients and abundances
    # Not done on our model, just on the source data. Used to check model results.
    # TODO Those are already calculated for the Q-score, merge code  

    average_crm_fig, tf_corr_fig, tf_abundance_fig, dataset_corr_fig, dataset_abundance_fig, jaccard_tf_fig = er.crm_diag_plots(list_of_many_crms, datasets_clean, cl_tfs)
    # NOTE Some "+" shape ghosts in the information plots (average CRM) can be due to summed crumbing.

    data_stats_output_path = plot_output_path + "data_stats/"
    if not os.path.exists(data_stats_output_path): os.makedirs(data_stats_output_path)

    average_crm_fig.savefig(data_stats_output_path+'average_crm_2d.pdf')
    tf_corr_fig.savefig(data_stats_output_path+'tf_correlation_matrix.pdf')
    dataset_corr_fig.savefig(data_stats_output_path+'dataset_correlation_matrix.pdf')
    tf_abundance_fig.savefig(data_stats_output_path+'tf_abundance_total_basepairs.pdf')
    dataset_abundance_fig.savefig(data_stats_output_path+'dataset_abundance_total_basepairs.pdf')
    jaccard_tf_fig.savefig(data_stats_output_path+'tf_jaccard_index_matrix.pdf')

    plt.close('all') # Close all figures


print("Model succesfully trained and diagnosed !")






# ------------------------------- Grid search -------------------------------- #
# The section below presents an example of code used to perform a grid search
# over possible parameters.
# The model is trained and then evaluated with the Q-score



# NOTE For now, only done on artificial data
if parameters['use_artificial_data']:

    print("Beginning grid search... this may be very long.")


    # Copy the original parameters, replace only the part we want, and add the 
    # newly created parameters to the list of parameters to try in the grid seach
    # Each element of the 'for' loop will create another set of parameters and 
    # add it to the list of parameters to try. Obviously, other 'for' loops 
    # could be nested inside.
    parameters_to_try = []
    for deepdim in [8, 32, 128, 256]:

        # Copy the current parameters from the YAML
        p = copy.deepcopy(parameters) 

        # Apply modifications 
        p["nn_deep_dim"] = deepdim

        # Add to the list of parameters to try
        parameters_to_try += [p]




    # Prepare to stock the models, in the same order as the `parameters_to_try`
    models_gridsearch = [None] * len(parameters_to_try) 
    result_grid = pd.DataFrame() # Will contain results

    # Now do the grid search
    i = 0

    for parameters_custom in parameters_to_try:
        
        # Prepare generator
        datasets_clean, cl_tfs, all_matrices = prepare.get_indexes(parameters_custom, crmid, datasets, cl_tfs)
        train_generator = produce_data_generator(parameters_custom, all_matrices, get_matrix) 
        list_of_many_crms = er.get_some_crms(train_generator, parameters_custom['nb_batches_generator_for_diag_list_of_many_crms'])

        # Prepare model
        models_gridsearch[i] = prepare.prepare_model_with_parameters(parameters_custom, len(datasets_clean), len(cl_tfs))
        models_gridsearch[i] = train_model(models_gridsearch[i], parameters_custom, train_generator, save_model_path)
        i += 1

        # Now calculate the Q-score
        q, _,_,_,_ = er.calculate_q_score(models_gridsearch[i], list_of_many_crms, datasets_clean, cl_tfs)
        plt.close('all') # Close all figures loaded in memory during q-score evaluation


        # Add resulting q-score to parameters and then record it
        parameters_custom.update({'q_score':np.sum(np.sum(q))})
        result_grid = result_grid.append(pd.Series(parameters_custom), ignore_index = True)
        

    # Produce and save the figure

    resdf = result_grid[['nn_deep_dim','q_score']]
    resdf['q_score_display'] = np.around(resdf['q_score'].values, decimals = 3)

    from plotnine import ggplot, aes, geom_bar, geom_text, scale_fill_brewer

    grid_search_plot = (ggplot(resdf, aes(x='factor(nn_deep_dim)', y='q_score')) +
        geom_bar(stat="identity", fill="steelblue") +
        geom_text(aes(label='q_score_display'), vjust='top', nudge_y = -0.03)
    )
    
    grid_search_plot.save(filename = plot_output_path+'artifical_data_grid_search_plot.png', height=10, width=14, units = 'in', dpi=400, verbose = False)
    plt.close('all')

    print("Grid search complete.")