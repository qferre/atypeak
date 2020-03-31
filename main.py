# ------------------------------ Imports ------------------------------------- #

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
from plotnine import ggplot, aes, geom_violin, geom_boxplot, position_dodge, scale_fill_grey

# ML
import keras
import keras.backend as K

## Custom libraries
import lib.model_atypeak as cp      # Autoencoder functions
import lib.data_read as dr          # Process data produced by Makefile
import lib.artificial_data as ad    # Trivial data for control
import lib.result_eval as er        # Result production and evaluation
import lib.utils as utils           # Miscellaneous

############################# PARAMETERS AND SETUP #############################

try: root_path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('Assuming a Jupyter kernel since `__file__` does not exist : root path set to "."')
    root_path = '.'

parameters = yaml.load(open(root_path+'/parameters.yaml').read(), Loader = yaml.Loader)


### Set random seeds - Numpy and TensorFlow
SEED = parameters["random_seed"]
# Python internals
os.environ["PYTHONHASHSEED"]=str(SEED)
random.seed(SEED)
# For Numpy, which will also impact Keras, and Theano if used
np.random.seed(SEED)




# For TensorFlow only
if K.backend() == 'tensorflow' :
    import tensorflow as tf

    # Check TensorFlow version
    from distutils.version import LooseVersion
    USING_TENSORFLOW_2 = (LooseVersion(tf.__version__) >= LooseVersion("2.0.0"))

    if not USING_TENSORFLOW_2:
        tf.set_random_seed(SEED)
        config = tf.ConfigProto()
        #config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth=True # Be parcimonious with RAM usage if on a GPU

        #tf.get_logger().setLevel('INFO') # Disable INFO, keep only WARNING and ERROR messages

        # For reproducibility ? Hmm I'm not sure
        #config.intra_op_parallelism_threads = 1
        #config.inter_op_parallelism_threads = 1

        sess = tf.Session(graph = tf.get_default_graph(), config=config)
        K.set_session(sess)

    if USING_TENSORFLOW_2:

        #import logging
        #logging.getLogger("tensorflow").setLevel(logging.ERROR)

        tf.random.set_seed(SEED) 

        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

        #tf.config.threading.set_inter_op_parallelism_threads(1)
        #tf.config.threading.set_intra_op_parallelism_threads(1)





## Reading corresponding keys. See the documentation of prepare() for more.
crmtf_dict, datasets, crmid, datapermatrix, peaks_per_dataset, cl_tfs = dr.prepare(parameters['cell_line'], root_path)


#datasets_clean_ori = [dr.dataset_parent_name(d) for d in datasets] # TODO Datasets : might wish to use a parent name later
datasets_clean = sorted(list(set(datasets)), key=datasets.index) # Make unique while preserving order, which a `set` would not do


# Since those dictionaries are fixed for a cell line, prepare a partial call
get_matrix = partial(dr.extract_matrix,
    all_tfs = cl_tfs,
    cell_line = parameters['cell_line'], cl_crm_tf_dict = crmtf_dict,
    cl_datasets = datasets_clean, crm_coordinates = crmid,
    datapath = root_path+'/data/input/sorted_intersect/')

print('Parameters loaded.')



"""
# TODO  To compute weights for the loss : based on the `crmtf_dict` object and 
# also on the `datapermatrix` and `peaks_per_dataset` objects, we know the 
# number of elements for each combination of TF/dataset.
"""


# Plot output path (different for artificial data)
plot_output_path = './data/output/diagnostic/'+parameters['cell_line']+'/'
if parameters['use_artificial_data'] : plot_output_path += 'artificial/'
if not os.path.exists(plot_output_path): os.makedirs(plot_output_path)




############################### DATA GENERATOR #################################
# You may use either the artificial data generator or the true data.
# Artificial data is mainly used for calibrations and demonstrations.

if parameters['use_artificial_data'] :
    # ------------------------- Artificial data ------------------------------ #
    train_generator = ad.generator_fake(region_length = parameters['pad_to'],
                                        nb_datasets = parameters['artificial_nb_datasets'], nb_tfs=parameters['artificial_nb_tfs'],
                                        squish_factor = parameters['squish_factor'], ones_only=parameters['artificial_ones_only'],
                                        watermark_prob = parameters['artificial_watermark_prob'],
                                        overlapping_groups = parameters['artificial_overlapping_groups'],
                                        tfgroup_split = parameters['artificial_tfgroup_split'],
                                        crumb = None)
    print('Using artificial data of dimensions : '+str(parameters['artificial_nb_datasets'])+' x '+str(parameters['artificial_nb_tfs']))
else:
    # ---------------------------- Real data --------------------------------- #
    # Collect all CRM numbers (each one is a *sample*)
    matrices_id = crmid.values()
    all_matrices = list(set(matrices_id)) # The sorting  itself is irrelevant here, but it is crucial that it is the same evertyime
    # TODO : set python hash seed !!!

    train_generator = cp.generator_unsparse(all_matrices, parameters["nn_batch_size"],
                            get_matrix, parameters["pad_to"], parameters["crumb"], parameters["squish_factor"])

    print("Using real data for the '"+parameters["cell_line"]+"' cell line.")





################################ MODEL #########################################
# Load parameters for the AutoEncoder (AE) from the config file, add missing
# ones and create the model


def prepare_model_with_parameters(parameters):
    """
    A wrapper function that will prepare an atyPeak model given the current parameters.
    Non-pure, since it depends on the rest of the code.
    """

    # Optimizer : Adam with custom learning rate
    optimizer_to_use = getattr(keras.optimizers, parameters["nn_optimizer"])
    opti_custom = optimizer_to_use(lr=parameters["nn_optimizer_learning_rate"])

    if parameters['use_artificial_data']:
        nb_datasets_model = parameters['artificial_nb_datasets']
        nb_tfs_model = parameters['artificial_nb_tfs']
    else:
        nb_datasets_model = len(datasets_clean)
        nb_tfs_model = len(cl_tfs)

    print("-- Dimensions : "+str(parameters["pad_to"])+
        'bp x '+str(nb_datasets_model)+' datasets x '+
        str(nb_tfs_model)+' TFs.')


    # Parameters checking
    # TODO CHECK THAT THOSE ARE INDEED NECESSARY
    totkernel = parameters['nn_kernel_width_in_basepairs'] * 4
    final_regionsize = int(parameters["pad_to"] / parameters['squish_factor'])
    if final_regionsize < totkernel:
        raise ValueError('Parameters error - Final region size after squishing must be higher than 4 * kernel_width_in_basepairs')
    if final_regionsize % totkernel != 0:
        raise ValueError('Parameters error - Final region size after squishing must be divisible by 4 * kernel_width_in_basepairs')


    # Compute weights for loss
    tf_weights = parameters["tf_weights"]
    datasets_weights = parameters["datasets_weights"]
    # Treat default
    if tf_weights is None: tf_weights = [1] * nb_tfs_model
    if datasets_weights is None: datasets_weights = [1] * nb_datasets_model
    weighted_mse = cp.create_weighted_mse(datasets_weights, tf_weights)
    # TODO : make a 2d matrix of weights instead, one weight for each specific tf+dataset pair. See draft code in the function source.


    # Finally, create the atypeak model
    model = cp.create_atypeak_model(
        kernel_nb=parameters["nn_kernel_nb"],
        kernel_width_in_basepairs=parameters["nn_kernel_width_in_basepairs"],
        reg_coef_filter=parameters["nn_reg_coef_filter"],
        pooling_factor=parameters["nn_pooling_factor"],
        deep_dim=parameters["nn_deep_dim"],
        region_size = int(parameters["pad_to"] / parameters['squish_factor']),
        nb_datasets = nb_datasets_model, nb_tfs = nb_tfs_model,
        optimizer = opti_custom, loss = weighted_mse
        )

    print('Model created.')

    # Print summary of the model (only if not artificial)
    if not parameters['use_artificial_data']  :
        with open(root_path+'/data/output/model/model_'+parameters['cell_line']+'_architecture.txt','w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))


    return model

# Prepare the model
model = prepare_model_with_parameters(parameters)


# Control print of parameters
print("-- PARAMETERS :")
print(parameters)




# ------------------------------ Training ------------------------------------ #
# Train only if not loading a saved model


def train_model(model,parametrs):
    """
    A wrapper function that will train a given model given the current parameters.
    Non-pure, since it depends on the rest of the code.

    Useful for grid search.
    """




    # NOTE This does not like custom optimizers it seems. To get around 
    # this problem, we re-create the model first, then load ONLY the weights !
    
    if parameters['load_saved_model']  :
        model_path = root_path+'/data/output/model/trained_model_'+parameters['cell_line']+'.h5'
        try:
            #model = load_model(model_path, custom_objects={"K": K, "optimizer":opti_custom}))
            model.load_weights(model_path)
       
        except OSError: raise FileNotFoundError("load_saved_model is True, but trained model was not found at "+model_path)


        # NOTE Due to a weird bug, I need to re-train the model for a few epochs after loading it.
        model.fit_generator(train_generator, verbose=0, steps_per_epoch = 1, epochs = 2, max_queue_size = 1)

        print('Loaded a saved model.')

    else :
        # Callback for early stopping
        # Don't stop as soon as exact floor is reached (patience)
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
        print('This can be long, depending on your hardware, from several minutes to around an hour.')

        start = time.time()
        model.fit_generator(train_generator, verbose=1,
            steps_per_epoch = parameters["nn_batches_per_epoch"],
            epochs = parameters["nn_number_of_epochs"],
            callbacks = [es, es2],
            max_queue_size = 1) # Max queue size of 1 to prevent memory leak
        end = time.time()

        total_time = end-start
        print('Training of the model completed in '+str(total_time)+' seconds.')



        # Save trained model and save the loss as text
        if not parameters['use_artificial_data'] :
            save_model_path = root_path+"/data/output/model/trained_model_"+parameters['cell_line']
        else:
            save_model_path = root_path+"/data/output/model/trained_model_ARTIFICIAL"
        loss_history = model.history.history["loss"]
        model.save(save_model_path+'.h5')
        np.savetxt(save_model_path+"_loss_history.txt", np.array(loss_history), delimiter=",")
        print('Model saved.')


    return model

# Train the model
model = train_model(model, parameters)










################################################################################
################################# DIAGNOSTIC ###################################
################################################################################


# Get some samples (3D tensor representations of CRMs, like, 200*48 or something)
# This is used MANY times in the code, for Q-score, many diagnostics, and
# abundance, normalization, etc.
list_of_many_crms = er.get_some_crms(train_generator,
    nb_of_batches_to_generate = parameters['nb_batches_generator_for_diag_list_of_many_crms'])
# TODO Add a timer for this



# Figure size. TODO make this a parameter ?
eval_figsize_small = (5,5)
eval_figsize_large = (8,5)

# Perform some diagnosis on the model. Only if the `perform_model_diagnosis` 
# flag in the parameters is True.
# Notably, it might be redundant when reloading a model.
if parameters['perform_model_diagnosis']:

    # ------------------------------ Evaluation ------------------------------ #
    # Plot some examples of CRMs, their reconstructions and anomaly scores.
    
    i = 0

    for _ in range(parameters["example_nb_batches"]):

        # Data
        before_batch = next(train_generator)[0]

        for n in range(len(before_batch)):

            before_raw = np.copy(before_batch[n,:,:,:,0])

            before = np.around(before_raw-0.11) # Remove crumbing if applicable TODO Make this more rigorous
            prediction = model.predict(before_raw[np.newaxis,...,np.newaxis])[0,:,:,:,0]

            # 2D - max along region axis
            before_2d = np.max(before, axis=0)
            plt.figure(figsize=eval_figsize_small); before_2d_plot = sns.heatmap(np.transpose(before_2d), cmap = 'Blues')
            prediction_2d = np.max(prediction, axis=0)
            plt.figure(figsize=eval_figsize_small); prediction_2d_plot = sns.heatmap(np.transpose(prediction_2d), annot = True, cmap = 'Greens', fmt='.2f')

            # utils.plot_3d_matrix(before, figsize=eval_figsize_large)
            # clipped_pred = np.around(np.clip(prediction,0,999), decimals=1)
            # utils.plot_3d_matrix(clipped_pred, figsize=eval_figsize_large)

            anomaly_matrix = er.anomaly(before, prediction)
            anomaly_plot = utils.plot_3d_matrix(anomaly_matrix, figsize=eval_figsize_large) # Normal in blue, anomalous in red


            # Save the figures 
            example_output_path = plot_output_path + "crm_example/"
            if not os.path.exists(example_output_path): os.makedirs(example_output_path)

            before_2d_plot.get_figure().savefig(example_output_path + "example_crm_before_2dmax_"+str(i)+".pdf")
            prediction_2d_plot.get_figure().savefig(example_output_path + "example_crm_rebuilt_2dmax_"+str(i)+".pdf")
            anomaly_plot.savefig(example_output_path + "example_crm_anomaly_"+str(i)+".pdf")

            i = i+1 # Increment counter





    # ----------------------- Abundance evaluation ------------------------

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
    #sb = np.ma.masked_equal(summed_befores, 0).mean(axis=0)
    sb = np.array(summed_befores).mean(axis=0)
    plt.figure(figsize=eval_figsize_small); mean_before_plot = sns.heatmap(sb, annot = True)

    anomaly_values = np.ma.masked_equal(summed_anomalies, 0)
    median_anomaly_values = np.ma.median(anomaly_values, axis=0)
    plt.figure(figsize=eval_figsize_small); median_anomaly_values_plot = sns.heatmap(median_anomaly_values, annot = True).get_figure()

    # Save this as a diagnostic plot
    median_anomaly_values_plot.savefig(plot_output_path+'median_anomaly_values.pdf')








    # ------------------------------ Q-score evaluation --------------------------- #
    # VERY IMPORTANT TALK ABOUT THIS MORE
    # For each {A,B} pair of sources (TR+dataset pairs, the model should give 
    # a different score to A alone than when B is present *only* when there is
    # actually a correlation. If not, if either learned too little (not identified)
    # the group) or too much (too precise). Conversely if there is no correlation
    # the presence of one should have no impact on the other.

    print("Beginning Q-score evaluation. Can be long...")

    qstart = time.time()

    q, qscore_plot, corr_plot, posvar_x_res_plot = er.calculate_q_score(model,
        list_of_many_befores = list_of_many_crms,
        all_datasets_names = datasets_clean, all_tf_names = cl_tfs)

    qend = time.time()


    qtotal_time = qend-qstart
    print('Q-score evaluation completed in '+str(qtotal_time)+' seconds.')


    # Final q-score (total sum)
    print("-- Total Q-score of the model (lower is better) : "+ str(np.sum(np.sum(q))))

    # Those plots give respectively the Q-score contribution of each pair 
    # (lower is better), the true correlation matrix for comparison, and
    # the third plot says whether the presence of both results in a change in 
    # score and should "look like" the correlation plot. In terms of axis X and
    # Y, you have datasets THEN Tfs. (ie. first datasets 1 to i and THEN TFs 1 to j)
    qscore_plot.savefig(plot_output_path+'qscore_plot.pdf')
    corr_plot.savefig(plot_output_path+'corr_plot_datasets_then_tf.pdf')
    posvar_x_res_plot.savefig(plot_output_path+'posvar_when_both_present_plot_datasets_then_tf.pdf')

    np.savetxt(plot_output_path+'q_score.tsv', q, delimiter='\t')

    print("Q-score evaluation results saved.")



    
    











    """
    ## WIP Grid search part

    # Read the yaml parameters to get a default parameters dict

    parameters_to_try = []
    # Copy the original parameters and replace only the part we want
    parameters_custom = copy(params)
    parameters_custom[key] = new_value
    parameters_to_try += [parameters_custom]

    result_grid = pd.DataFrame()

    # now do the search
    for parameters_custom in parameters_to_try:
        model = prepare_model_with_parameters(parameters_custom)
        trained_model = train_model(model, parameters_custom)
        q, _,_,_ = q_score(model, train_generator)

        # Add resulting q-score to parameters
        parameters_to_try.update({'Q_score':np.sum(np.sum(q))})

        parameters_to_try = parameters

        # And record the q_score
        result_grid = result_grid.append(pd.Series(parameters_to_try), ignore_index = True)
    """




    # ------------------------------ Visualize filters --------------------------- #

    # Visualize filters

    # This was illuminating when calibrating the parameters and making sure the 
    # model was learning, but I have not found it otherwise to be particularly
    # useful.
    kernel_output_path = plot_output_path + "conv_kernels/"
    if not os.path.exists(kernel_output_path): os.makedirs(kernel_output_path)

    # Datasets
    w = model.layers[2].get_weights()[0]
    for filter_nb in range(w.shape[-1]):
        plt.figure(figsize=eval_figsize_small); dfp = sns.heatmap(w[:,:,0,0,filter_nb])
        #utils.plot_3d_matrix(w[:,:,:,0,filter_nb], figsize=(6,4))
        dfp.get_figure().savefig(kernel_output_path + "conv_kernel_datasets_"+str(filter_nb)+".pdf")
    
    # TFs
    w = model.layers[5].get_weights()[0]
    for filter_nb in range(w.shape[-1]):
        #utils.plot_3d_matrix(w[:,0,:,:,filter_nb], figsize=(6,4))
        plt.figure(figsize=eval_figsize_small); tfp = sns.heatmap(w[0,0,:,:,filter_nb]) # 2D version only at the first X (often the same anyways)
        dfp.get_figure().savefig(kernel_output_path + "conv_kernel_tf_x0_"+str(filter_nb)+".pdf")

    # ----------------------- Visualize encoded representation ------------------- #

    # I added some Y and Z blur, just to smooth it a little. Keep the blur minimal, makes no sense otherwise. 

    ENCODED_LAYER_NUMBER = 15 # Starts at 0 I think

    # To check this is the correct number:
    if not (model.layers[ENCODED_LAYER_NUMBER].name == 'encoded'):
        print("ENCODED_LAYER_NUMBER not set to the encoded dimension. Urexamples will be on a different dimension.")
    # TODO Replace with something like model.get_layer("encoded")


    urexamples = cp.compute_max_activating_example_across_layer(model, random_state = 42,
                                selected_layer = ENCODED_LAYER_NUMBER, output_of_layer_select_this_dim = 2,
                                learning_rate = 1, nb_steps_gradient_ascent = 50,
                                blurStdX = 0.2, blurStdY = 1E-2,  blurStdZ = 1E-2, blurEvery = 5)


    urexample_output_path = plot_output_path + "urexample_encoded_dim/"
    if not os.path.exists(urexample_output_path): os.makedirs(urexample_output_path)

    for exid in range(len(urexamples)):
        ex = urexamples[exid]
        ex = ex[...,0]
        #ex = np.around(ex/np.max(ex), decimals = 1)
        x = np.mean(ex, axis = 0)
        plt.figure(figsize=eval_figsize_small); urfig = sns.heatmap(np.transpose(x), cmap ='RdBu_r', center = 0)
        urfig.get_figure().savefig(urexample_output_path + "urexample_dim_"+exid+".pdf")


    """
    # Get an encoded representation for comparison (from the latest `before`, from above)
    tmp = cp.get_encoded_representation(before[np.newaxis,...,np.newaxis], model)
    tmpd = np.transpose(tmp[0]**2)
    plt.figure(figsize=eval_figsize_large); sns.heatmap(tmpd)
    utils.plot_3d_matrix(before)
    """












    # ----------------------- Proof on artificial data ----------------------- #
    # Plot score of artificial peaks depneding on type (noise, stack, ...)

    if parameters['use_artificial_data'] :

        # Prepare a generator of artificial data that returns the peaks separately
        ad_partial = partial(ad.make_a_fake_matrix, region_length = parameters['pad_to'],
                        nb_datasets = parameters['artificial_nb_datasets'], nb_tfs = parameters['artificial_nb_tfs'],
                        signal = True, noise=True, ones_only = parameters['artificial_ones_only'], return_separately = True)


        # Should take roughly one or two minutes
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

        a2.save(filename = plot_output_path+'artifical_data_systematisation_value_per_type.png', height=10, width=14, units = 'in', dpi=3200)
        b.save(filename = plot_output_path+'artifical_data_systematisation_value_per_brothers.png', height=10, width=14, units = 'in', dpi=3200)









################################################################################
################################## REAL DATA ###################################
################################################################################


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
        start_prod = time.time()
        er.produce_result_file(all_matrices, output_bed_path,
            model, get_matrix, parameters, datasets_clean, cl_tfs)
        end_prod = time.time()
        total_time_prod = end_prod-start_prod
        print('Processed BED file produced in',str(total_time_prod),'seconds.')

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
            outfilepath = './data/output/diagnostic/'+parameters['cell_line']+'/'+"normalization_factors.txt")

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
            plt.savefig(plot_output_path+'scores_by_tf_after_corrgroup_normalization_only.pdf')

            # By dataset
            fig, ax = plt.subplots(figsize=(10, 8))
            sub_df = scores_by_dataset_df.loc[:,['count','50pc']]
            sub_df.plot('count', '50pc', kind='scatter', ax=ax, s=24, linewidth=0) ; ax.grid()
            for k, v in sub_df.iterrows(): ax.annotate(k, v,xytext=(10,-5), textcoords='offset points')
            plt.savefig(plot_output_path+'scores_by_dataset_after_corrgroup_normalization_only.pdf')





























            ## Quantify effect of normalization

            sub_raw = scores_by_tf_df_raw.loc[:,['tf','mean']]
            sub_raw.columns = ['tf','mean_raw_before_norm']
            sub_after = scores_by_tf_df.loc[:,['tf','mean']]
            sub_after.columns = ['tf','mean_after_norm']
            df_cur = pd.merge(sub_raw, sub_after, on="tf")
            #df_cur = sub_raw.to_frame(name = 'mean_raw_before_norm').join(sub_after.to_frame(name='mean_after_norm'))
            df_cur_melted = df_cur.melt(id_vars=['tf'], value_vars=['mean_raw_before_norm','mean_after_norm'])
            p = ggplot(df_cur_melted, aes(x = tf, y= value, fill = variable)) + geom_bar(stat="identity", width=.5, position = "dodge")
            p.save(plot_output_path+"scores_by_tf_before_and_after_corrgroup_normalization.pdf", height=8, width=10, units = 'in', dpi=3200)

            # TODO Same for datasets


























            # ------ Informative plots for result checking
            # TODO STOCK AND EXPORT ALL THESE PLOTS

            # Those are CONTROL Plots. Explain that.
            # They are not done on our diagnostic, they just allow you to check the results

            # -Including : Compare to the average CRM
            # Produce a picture of the average CRM for later comparisons

            # TODO : THOSE ARE MAYBE ALREADY CALCULATED BY THE Q-SCORE. MERGE THIS CODE WITH THE Q-SCORE CODE TO AVOID REDUNDANCIES ?




            average_crm_fig, tf_corr_fig, tf_abundance_fig, dataset_corr_fig, dataset_abundance_fig = er.crm_diag_plots(list_of_many_crms, datasets_clean, cl_tfs)








            """
            # Careful : some horizontal "ghosting" might be due to summed crumbing. TODO NOTE IN PAPER !!!! IN THE FIGURE LEGENDS
            """











            average_crm_fig.savefig(plot_output_path+'average_crm_2d.pdf')
            tf_corr_fig.savefig(plot_output_path+'tf_correlation_matrix.pdf')
            dataset_corr_fig.savefig(plot_output_path+'dataset_correlation_matrix.pdf')
            tf_abundance_fig.savefig(plot_output_path+'tf_abundance_total_basepairs.pdf')
            dataset_abundance_fig.savefig(plot_output_path+'dataset_abundance_total_basepairs.pdf')




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




            # ----------------- Scores when known cofactors (or known non-cofactors) are present

            print("Retrieving scores for specified TF pairs and estimating correlation groups...")




            # Work on NORMALIZED scores. TODO SAY SO IN PAPER FIGURES AND IN DEBUG MESSAGES and/or comments here !!!!!

            # TODO CHECK WHICH FILE I USE PRECISELY



            tfs_to_plot = parameters['tf_pairs']

            tf_alone_both_output_path = plot_output_path + "tf_pairs_alone_both/"
            if not os.path.exists(tf_alone_both_output_path): os.makedirs(tf_alone_both_output_path)

            for pair in tfs_to_plot:
                try:
                # TODO CAREFUL ABOUT CASE !!
                    tf1, tf2 = pair
                    p, _ = er.get_scores_whether_copresent(tf1, tf2, output_bed_path_final, CRM_FILE)
                    p.save(tf_alone_both_output_path+"tf_alone_both_"+tf1+"_"+tf2+".pdf")
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
                    
                    fig = er.estimate_corr_group_for_combi(dataset, tf,
                        all_datasets = datasets_clean, all_tfs = cl_tfs, model = model,
                        crm_length = parameters['pad_to'], squish_factor = parameters["squish_factor"])
                    fig.get_figure().savefig(output_path_estimation)
                     
                     
                except:
                    print("Error estimating for : "+str(combi))
                    print("Ignoring.")
