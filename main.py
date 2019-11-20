#!/usr/bin/env python3
# ------------------------------ Imports ------------------------------------- #

# System
import os
import sys
import yaml
import time
from functools import partial

# Data
import numpy as np
import pandas as pd
import scipy

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

# ML
import keras
import keras.backend as K


## Custom libraries
import lib.convpeakdenoise as cp    # Autoencoder functions
import lib.data_read as dr          # Process data produced by Makefile
import lib.artificial_data as ad    # Trivial data for control
import lib.result_eval as er        # Result production and evaluation
import lib.utils as utils           # Miscellaneous


################################## PARAMETERS ##################################

try: root_path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('Assuming a Jupyter kernel since `__file__` does not exist : root path set to "."')
    root_path = '.'

parameters = yaml.load(open(root_path+'/parameters.yaml').read(), Loader = yaml.Loader)


### Set random seeds - Numpy and TensorFlow
# For Numpy, which will also impact Keras, and Theano if used
np.random.seed(parameters["random_seed"])
# For TensorFlow only
if K.backend() == 'tensorflow' :
    import tensorflow as tf
    tf.set_random_seed(parameters["random_seed"])

    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        print('Tensorflow found a GPU. It should use it unless you have custom overrides.')
    else: print('Tensorflow found no GPU. Code will be run on your CPU.')






## Reading corresponding keys. See the documentation of prepare() for more.
crmtf_dict, datasets, crmid, datapermatrix, peaks_per_dataset, cl_tfs = dr.prepare(parameters['cell_line'], root_path)
# Since those dictionaries are fixed for a cell line, prepare a partial call
get_matrix = partial(dr.extract_matrix,
    all_tfs = cl_tfs,
    cell_line = parameters['cell_line'], cl_crm_tf_dict = crmtf_dict,
    cl_datasets = datasets, crm_coordinates = crmid,
    datapath = root_path+'/data/input/sorted_intersect/')

# Datasets : remember to use the parent name
datasets_clean_ori = [dr.dataset_parent_name(d) for d in datasets]
datasets_clean = sorted(list(set(datasets_clean_ori)), key=datasets_clean_ori.index) # Make unique while preserving order, which a `set` would not do
print('Parameters loaded.')


# Execution parameters
#parameters['use_artificial_data']     # Do we use artificial data ?
#= parameters['load_saved_model']           # Should we reload a trained model ?


print('Working on the cell line : '+str(parameters['cell_line']))






"""
# Compute weights for the loss : based on the crmtf_dict object, we know the number
# of each combination of TF/dataset, so we can compute the numbers.
crmtf_dict
"""

"""
# # TODO do something with peaks_per_dataset
# # likely like I told jeanne, need to use it to compute a
# # custom weight, or do oversampling.
#
# # Get the datasets represented
# my_peaks_per_dataset = {k: peaks_per_dataset[k] for k in datasets}
# # Sum for the cleaned datasets
# ppdc = dict()
# for dataset,value in my_peaks_per_dataset.items():
#     dc = dr.dataset_parent_name(dataset)
#     if dc in ppdc : ppdc[dc] = ppdc[dc] + value
#     else: ppdc[dc] = value
# my_peaks_per_dataset_clean = ppdc
# plt.hist(my_peaks_per_dataset_clean.values(),bins=100)

# MOVE THIS AT THE BEGINNING. Produce a dictionary giving the relative abundances
# of TFs and datasets, and use it to generate a weighted loss.
"""












############################### DATA GENERATOR #################################

# You may use either the artificial data generator or the true data.
# I recommend using the artificial one to calibrate the parameters of your model
# to the desired result, as the characteristics of this data are known, before
# using the true one.


# --------------------------- Artificial data -------------------------------- #


if parameters['use_artificial_data']  :
    train_generator = ad.generator_fake(region_length = parameters['pad_to'],
                                        nb_datasets = parameters['artificial_nb_datasets'], nb_tfs=parameters['artificial_nb_tfs'],
                                        squish_factor = parameters['squish_factor'], ones_only=parameters['artificial_ones_only'],
                                        watermark_prob = parameters['artificial_watermark_prob'], crumb = None)
    print('Using artificial data of dimensions : '+str(parameters['artificial_nb_datasets'])+' x '+str(parameters['artificial_nb_tfs']))




# ------------------------------ Real data ----------------------------------- #

if not parameters['use_artificial_data']  :

    # Collect all CRM numbers (each one is a *sample*)
    matrices_id = crmid.values()
    all_matrices = list(set(matrices_id)) # Sorting is irrelevant here.

    train_generator = cp.generator_unsparse(all_matrices, parameters["nn_batch_size"],
                            get_matrix, parameters["pad_to"], parameters["crumb"], parameters["squish_factor"])

    print("Using real data for the '"+parameters["cell_line"]+"' cell line.")




"""
# Before proceeding, profiling the batch generator for true data
# TODO IMPROVE THIS IT IS MOSTLY THE ENTIRE PERFORMANCE HOG
import cProfile
cProfile.run('before_batch = next(train_generator)[0]', 'batchstats')
import pstats

p = pstats.Stats('batchstats')
p.strip_dirs().sort_stats('tottime').print_stats()
"""




















































































################################ MODEL #########################################

# Load parameters for the AutoEncoder (AE) from the config file, add missing
# ones and create the model


# Optimizer : Adam with custom learning rate
optimizer_to_use = getattr(keras.optimizers, parameters["nn_optimizer"])
opti_custom = optimizer_to_use(lr=parameters["nn_optimizer_learning_rate"])


if parameters['use_artificial_data']  :
    nb_datasets_model = parameters['artificial_nb_datasets']
    nb_tfs_model = parameters['artificial_nb_tfs']
else :
    nb_datasets_model = len(datasets_clean)
    nb_tfs_model = len(cl_tfs)


print("-- Dimensions : "+str(parameters["pad_to"])+
    'bp x '+str(nb_datasets_model)+' datasets x '+
    str(nb_tfs_model)+' TFs.')


# TODO Add a warning if big dimension (above 20)
if (nb_tfs_model > 20) | (nb_datasets_model > 20):
        print('TODO High dimension warning')




# Parameters checking
# CHECK THAT THOSE ARE INDEED NECESSARY
totkernel = parameters['nn_kernel_width_in_basepairs'] * 4
final_regionsize = int(parameters["pad_to"] / parameters['squish_factor'])
if final_regionsize < totkernel:
    raise ValueError('Parameters error - Final region size after squishing must be higher than 4 * kernel_width_in_basepairs')
if final_regionsize % totkernel != 0:
    raise ValueError('Parameters error - Final region size after squishing must be divisible by 4 * kernel_width_in_basepairs')













import importlib
importlib.reload(cp)
# Simple proof of concept try
#datasets_weights = [10,10,10,10,1,1,1,1]
tf_weights = [1,1,1,1,1,1,1,1]
# Okay it worked ! Now back to regular weights
datasets_weights = [1,1,1,1,1,1,1,1]
weighted_mse = cp.create_weighted_mse(datasets_weights, tf_weights)











#
# a = np.zeros((300,8,10,1))
# b = np.zeros((300,8,10,1))
#
# a[:,3,3,:] = 1
#
# r = weighted_loss(a,b)
#
# with sess.as_default(): R = K.eval(r)








model = cp.create_convpeakdenoise_model(

    kernel_nb=parameters["nn_kernel_nb"],kernel_width_in_basepairs=parameters["nn_kernel_width_in_basepairs"], reg_coef_filter=parameters["nn_reg_coef_filter"],
    pooling_factor=parameters["nn_pooling_factor"], deep_dim=parameters["nn_deep_dim"],



    region_size = int(parameters["pad_to"] / parameters['squish_factor']),
    nb_datasets = nb_datasets_model, nb_tfs = nb_tfs_model,
    optimizer = opti_custom, loss = weighted_mse)
    #optimizer = opti_custom, loss = 'mse') # TODO change this back to `optimizer = opti_custom, loss = weighted_loss`

print('Model created.')


# Print summary of the model
# Print only if not artificial
if not parameters['use_artificial_data']  :
    with open(root_path+'/data/output/model/model_'+parameters['cell_line']+'_architecture.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))












class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='loss', value=0.00001, verbose=0, patience = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.patience = patience

        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None: warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            self.wait += 1
            print("Waiting...")


        if self.wait >= self.patience:
            if self.verbose > 0: print("Epoch %05d: early stopping after patience for THR" % epoch)
            self.model.stop_training = True











print(parameters)





# ------------------------------ Training ------------------------------------ #
# Train only if not loading a saved model

# Custom session to be parcimonious with RAM usage
if K.backend() == 'tensorflow' :
    # Activate log_device_placement to check which device is being used
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)


# IMPORTANT To get around this problem, we re-create the model first, then load ONLY the weights !
if parameters['load_saved_model']  :
    model_path = root_path+'/data/output/model/trained_model_'+parameters['cell_line']+'.h5'
    try:
        #model = load_model(model_path, custom_objects={"K": K, "optimizer":opti_custom}))
        model.load_weights(model_path)
    # does not like custom optimizers it seems? So I rebuild the model an just reload the weights.
    except OSError: raise FileNotFoundError("load_saved_model is True, but trained model was not found at "+model_path)




    # TODO Due to a weird bug, I need to re-train the model one step after loading it. Try to find out why.
    model.fit_generator(train_generator, verbose=0, steps_per_epoch = 1, epochs = 1, max_queue_size = 1)

    # TODO : might need more than 1 epoch sometimes !?

    print('Loaded a saved model.')
















else :
    # Callback for early stopping
    #es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=0)



    # TODO MAKE THIS A PARAMATERS
    # TODO : Diasble this by default,or make it at 0. Make it a parameter
    #es = EarlyStoppingByLossVal(monitor='loss', value=0.003, patience=2)
    # TODO : does patience work with this custom thing ?
    es = EarlyStoppingByLossVal(monitor='loss', value=0, patience=2)    # Don't stop as soon as exact floor is reached (patience)


    # ALSO  stop in any case after 5 epochs with no improvement
    # Changed the patience to 8 since I diminished the learning rate and added a delta
    es2 = keras.callbacks.EarlyStopping(monitor='loss', patience = 8, min_delta = 0.00025,
        restore_best_weights = True)
        # Try to combat loss spikes : they are often a symptom of hypertraining
        # so when the loss starts going up or starts stagnating, indicating it's
        # likely starting to learn "always the same thing", stop and restore the best weights
    # TODO MAKE THIS MIN DELTA A PARAMATER AND





    # DEBUG : save at every epoch

    class CustomSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            self.model.save("DEBUG_model_at_epoch_{}.hd5".format(epoch))

    saver = CustomSaver()

    print('Beginning training.')
    print('This can be long, depending on your hardware, from several minutes to around an hour.')

    start = time.time()






    #parameters["nn_number_of_epochs"] = 16
    #parameters["nn_number_of_epochs"] = 24

    # TODO REUSE THIS PROPER TRAINER !!!
    """
    model.fit_generator(train_generator, verbose=1,
        steps_per_epoch = parameters["nn_batches_per_epoch"],
        epochs = parameters["nn_number_of_epochs"],
        callbacks = [es, es2, saver],
        max_queue_size = 1) # Model is quite complex with large data, so max queue size of 1 to prevent memory leak
        # TODO  : no model is not 'quite complex with large data' just say the queue size helps
    """




    model.fit_generator(train_generator, verbose=1,
        steps_per_epoch = parameters["nn_batches_per_epoch"],
        epochs = parameters["nn_number_of_epochs"],
        callbacks = [es, es2, saver],
        max_queue_size = 1) # Model is quite complex with large data, so max queue size of 1 to prevent memory leak
        # TODO  : no model is not 'quite complex with large data' just say the queue size helps
    end = time.time()
    print('Training completed in '+str(end-start)+' s')




    # Save trained model
    if not parameters['use_artificial_data'] :
        model.save(root_path+"/data/output/model/trained_model_"+parameters['cell_line']+'.h5')
    else:
        model.save(root_path+"/data/output/model/trained_model_ARTIFICIAL.h5")
    print('Model saved.')





# TODO : Save the loss as text !!
loss_history = model.history.history["loss"]
np.savetxt(root_path+"/data/output/model/"+"trained_model"+parameters['cell_line']+"_loss_history.txt",
    np.array(loss_history), delimiter=",")



















################################################################################
################################# DIAGNOSTIC ###################################
################################################################################
# Now that the data has been processed, produce some diagnostic plots
# HMM NO PRODUCE THE FILE LATER ONLY WHEN NEEDED AS IT IS LONG


# IMPORTANT REMARK :
# The file has been produced here, so I can easily run this part only in a
# Jupyter kernel ? Except for the artificial data proof part I mean.
# HMM NO PRODUCE THE FILE LATER ONLY WHEN NEEDED AS IT IS LONG




# TODO FUNCTIONALIZE MOST OF THIS !!??






# Plot output path
plot_output_path = './data/output/diagnostic/'+parameters['cell_line']+'/'

# If artificial :
if parameters['use_artificial_data'] : plot_output_path = './data/output/diagnostic/'+parameters['cell_line']+'/artificial/'


if not os.path.exists(plot_output_path): os.makedirs(plot_output_path)











# ------------------------------- Evaluation --------------------------------- #
# CAN BE RUN WITH BOTH TRUE AND ARTIFICIAL OF COURSe, but mostly meant for artifical
# NOTE meant be be run in loop in a Jupyter kernel)

# TODO : maybe make this draw 20 examples and produce an evaluation pdf with those figures



# TODO : the eval parameter will plot directly some examples. Set it if Jupyter kernel ?


if eval:

    eval_figsize = (8,4)

    for i in range(1):

        # Data
        before_batch = next(train_generator)[0]
        # TODO Maybe use another ID than 4 ? If was 4 for no good reason
        ID = 2 ; before = before_batch[ID,:,:,:,0]
        before = np.around(before-0.11)
        utils.plot_3d_matrix(before, figsize=eval_figsize)


        """
        With ludicrous deep dims the unrelaible datasets are learned !!
            Ah... No they are not learned... hmm...
        and it looks like it's still adding some phantoms in the correct corr group for the big ones and the scores
        for the peas are still bigger when there are more correlands, although the effect is more minute as the q-score shows :
            of course it is much more precise.
        This suggests scalability is excellent and
        """


        prediction = model.predict(before[np.newaxis,...,np.newaxis])[0,:,:,:,0]
        clipped_pred = np.around(np.clip(prediction,0,999), decimals=1)
        utils.plot_3d_matrix(clipped_pred, figsize=eval_figsize)


        prediction = model.predict(before[np.newaxis,...,np.newaxis])[0,:,:,:,0]
        clipped_pred = np.around(np.clip(prediction-0.1,0,999), decimals=1)
        utils.plot_3d_matrix(clipped_pred, figsize=eval_figsize)


        utils.plot_3d_matrix(prediction, figsize=eval_figsize)
        utils.plot_3d_matrix(prediction, figsize=eval_figsize)
        utils.plot_3d_matrix(prediction, figsize=eval_figsize)







        prediction = model.predict(before[np.newaxis,...,np.newaxis])[0,:,:,:,0]
        clipped_pred = np.around(np.clip(prediction,0-0.01,999), decimals=2)
        utils.plot_3d_matrix(clipped_pred, figsize=eval_figsize)

        anomaly_matrix = er.anomaly(before, prediction)
        utils.plot_3d_matrix(anomaly_matrix, figsize=eval_figsize) # Normal in blue, anomalous in red



        # Remove the complete zeros for comparison
        utils.plot_3d_matrix(np.clip(anomaly_matrix-0.05,0,1), figsize=eval_figsize) # Normal in blue, anomalous in red









# --- Abundance evaluation
# Both artificial and true ?

# Use this now that is is 2/3 1/3 for each TF group to test a frequency theory.
# TODO MAKE SURE THAT IS THE CASE !



summed_anomalies = []
summed_befores = []

# TODO : MAKE THAT A PARAMETER !
for i in range(1000):

    before_batch = next(train_generator)[0]

    for k in range(before_batch.shape[0]):

        # Get max across X axis to transpose back in 2D

        # Data
        before = before_batch[k,:,:,:,0]
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
sns.heatmap(sb, annot = True)

sa = np.array(summed_anomalies).mean(axis=0)
sns.heatmap(sa, annot = True)



mean_anomaly_values = np.ma.masked_equal(summed_anomalies, 0).mean(axis=0)
sns.heatmap(mean_anomaly_values, annot = True)





mean_anomaly_values = np.ma.masked_equal(summed_anomalies, 0).mean(axis=0)
sns.heatmap(mean_anomaly_values, annot = True)
# TODO SAVE THOSE PLOTS ?


parameters








































# ------------------------------ Q-score evaluation --------------------------- #
# VERY IMPORTANT TALK ABOUT THIS MORE
# Main idea : the model should give different score in alone+both when there is
# actually a correlation. If not, if either learned too little (not identified)
# the group) or too much (too precise). Conversely if there is no correlation
# the presence of one should have no impact on the other.



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

scores


"""





import importlib
importlib.reload(er)



# TODO make nb_of_batches_to_generate a parameter in yaml for evaluation !!
# TODO THIS IS CRITICAL BEAUSE IT CAN TAKE VERY LONG FOR THE HIGH DIMENSION REAL DATA !!!


q, qscore_plot, corr_plot, posvar_x_res_plot = er.calculate_q_score(model, train_generator,
    nb_of_batches_to_generate = 200)
# Final q-score :
q_score = np.sum(np.sum(q))
print(q_score)

# TODO SAVE THE ABOVE PLOTS, AND MAYBE EVEN SAVE THE ENTIRE Q MATRIX IN A TXT !!!!! This way Q-score value can be recalculated externally

"""
Add explanation of those plots, and say that in those, in terms of axis X and Y, those are you have datasets THEN Tfs. (ie. first datasets 1 to i and THEN TFs 1 to j)
SAME FOR THE Q-SCORE MATRIX !
"""

qscore_plot.savefig(plot_output_path+'qscore_plot.pdf')
corr_plot.savefig(plot_output_path+'corr_plot_datasets_then_tf.pdf')
posvar_x_res_plot.savefig(plot_output_path+'posvar_when_both_present_plot_datasets_then_tf.pdf')

np.savetxt(plot_output_path+'q_score.tsv', q, delimiter='\t')










# ------------------------------ Visualize filters --------------------------- #


# Visualize filters
FILTERS_TO_PLOT_MAX = 20

# TODO THOSE ARE NOT THE CORRECT LAYER NUMBERS ANYMORE I MOVE THEM !

model.layers

if eval:
    # Datasets
    w = model.layers[2].get_weights()[0]
    for filter_nb in range(w.shape[-1])[0:FILTERS_TO_PLOT_MAX] :
        sns.heatmap(w[:,:,0,0,filter_nb]) ; plt.figure()
        #utils.plot_3d_matrix(w[:,:,:,0,filter_nb], figsize=(6,4))

    # TFs
    w = model.layers[5].get_weights()[0]
    for filter_nb in range(w.shape[-1])[0:FILTERS_TO_PLOT_MAX] :
        #utils.plot_3d_matrix(w[:,0,:,:,filter_nb], figsize=(6,4)) ; plt.figure()
        sns.heatmap(w[0,0,:,:,filter_nb]) ; plt.figure() # 2d version only at the first X (often the same anyways)


# ----------------------- Visualize encoded representation ------------------- #

"""
Do on artificial data first, but should work on real data as well.
"""



# I added some Y and Z blur, just to smooth it a little. KEEP THE BLUR MINIMAL AS IT DOES NOKE MAKE SENSE

ENCODED_LAYER_NUMBER = 15 # Starts at 0 I think

# To check this is the correct number:
print(model.layers[ENCODED_LAYER_NUMBER].name == 'encoded')

# TODO replace with something like model.get_layer("encoded")


# DO this only if eval is True
if eval :
    # TODO RENAME UREXAMPLES, IT's MORE AKIN TO ATTENTION
    urexamples = cp.compute_max_activating_example_across_layer(model, random_state = 42,
                                selected_layer = ENCODED_LAYER_NUMBER, output_of_layer_select_this_dim = 2,
                                #learning_rate = 0.25, nb_steps_gradient_ascent = 250,
                                learning_rate = 1, nb_steps_gradient_ascent = 50,
                                blurStdX = 0.2, blurStdY=1E-2,  blurStdZ=1E-2, blurEvery = 5)

    for ex in urexamples:
        ex = ex[...,0] #; rounded_ex = np.around(ex/np.max(ex), decimals = 1)
        #x = np.mean(rounded_ex, axis = 0)
        x = np.mean(ex, axis = 0)
        #x = np.abs(x)
        sns.heatmap(np.flip(x.transpose(), axis=0), cmap ='RdBu_r', center = 0) ; plt.figure()
        #utils.plot_3d_matrix(np.abs(rounded_ex), figsize = (6,4))
        #utils.plot_3d_matrix(ex, figsize = (6,4))



    # To understand, provide a function that, given a before matrix (or simply a CRM) will output its encoded representation. So you can understand the combinations at play.
    def get_encoded_representation(before_matrix, model, ENCODED_LAYER_NUMBER = 15, disable_learning = True):

        # Disable the learning phase behavior (e.g. Dropout)
        if disable_learning: learnflag = 0
        else: learnflag = 1

        get_kth_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                          [model.layers[ENCODED_LAYER_NUMBER].output])
        result = get_kth_layer_output([before_matrix, learnflag])[0]

        return result




    tmp = get_encoded_representation(before[np.newaxis,...,np.newaxis], model)
    sns.heatmap(np.transpose(tmp[0])**2)
    utils.plot_3d_matrix(before[0,...,0])



"""
VISUALIZATION OF ENCODED SHAPES IS VERY PROMISING ! NEED TO RETRY THIS IN JURKAT WITH REAL DATA !!!!!!!!!
"""












































# ------------------------ Proof on artificial data -------------------------- #
# Plot score of artificial peaks depneding on type (noise, stack, ...)

"""
Call the proof on artificial :

TODO : all the artificial data stuff will go in a separate python file ?

Or just run all of this if artificial data is true ? YEAH DO BOTH : move to a separate python file
and import and run the functions if artificial data is true
"""


if parameters['use_artificial_data'] :
    # do all this below



    # Prepare a generator of artificial data that returns the peaks separately
    mypartial = partial(ad.make_a_fake_matrix, region_length = parameters['pad_to'],
                    nb_datasets = parameters['artificial_nb_datasets'], nb_tfs = parameters['artificial_nb_tfs'],
                    signal = True, noise=True, ones_only = parameters['artificial_ones_only'], return_separately = True)


    # ADD EXPLANATION FOR N_ITER
    # Should take roughly one or two minutes
    arti_start = time.time()
    df, separated_peaks = er.proof_artificial(model, mypartial,
        region_length = parameters['pad_to'], nb_datasets = parameters['artificial_nb_datasets'], nb_tfs = parameters['artificial_nb_tfs'],
        n_iter = 500, squish_factor = parameters['squish_factor'])
    arti_end = time.time()
    print('Artificial data generalisation completed in '+str(arti_end-arti_start)+' s')




    plt.figure(figsize=(10,8))


    # The plots
    # TODO : use geom_boxplot or geom_violin ?
    a = ggplot(df, aes(x="type", y="rebuilt_value", fill="tf_group"))
    a1 = a + geom_violin(position=position_dodge(1), width=1)
    a2 = a + geom_boxplot(position=position_dodge(1), width=0.5)
    b = ggplot(df, aes(x="brothers", y="rebuilt_value", group="brothers")) + scale_fill_grey() + geom_boxplot(width = 0.4)




    # Save them
    a2.save(filename = plot_output_path+'artifical_data_systematisation_value_per_type.png', height=10, width=14, units = 'in', dpi=3200)
    b.save(filename = plot_output_path+'artifical_data_systematisation_value_per_brothers.png', height=10, width=14, units = 'in', dpi=3200)






















################################################################################
################################## REAL DATA ###################################
################################################################################





# Switch : the user should calibrate with Q-score before processing the full data !
# So in the parameters if the switch process_full_real_data is not on, it should exit here and ther

process_full_real_data = parameters["process_full_real_data"]

if not process_full_real_data:
    print("The parameter `process_full_real_data` was set to False, presumably because this was a parameter calibration run.")
    print("Hence, we stop before processing the real data.")
    sys.exit()

else:
    print("Proceeding with real data. Make sure you have calibrated the model correctly, using notably the Q-score !")
    print("Warning : this can be VERY long.")




"""
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
# TODO MOVE BED FILE CREATION HERE !!!! IT IS LONG SO ONLY PRODUCE IT WHEN ABSOLUTELY NECESSARY !
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
"""

################################ DENOISING/PRODUCE REAL DATA #####################################
# Use the now-trained model on all data.

# Only do this on true data of course
if not parameters['use_artificial_data'] :

    print("Writing result BED file for peaks, with anomaly score.")
    print("This can be long (roughly 1 second for 10 CRMs with reasonably-sized queries, like 15 datasets x 15 TFs).")
    print("On big queries (like 25*50) it can be 1 second per CRM.")

    # Produce a raw, non-normalized file for this cell line
    output_bed_path = root_path+'/data/output/bed/'+parameters['cell_line']+'.bed'
    print('Producing bed file : '+output_bed_path)
    er.produce_result_file(all_matrices, output_bed_path, model, get_matrix, parameters, datasets_clean, cl_tfs)


    """
    # DEBUG TEST TIME
    import importlib
    importlib.reload(er)
    import time
    s = time.time()
    er.produce_result_file(all_matrices[0:10], output_bed_path+'.tmp.poubtest',
        model, get_matrix, parameters, datasets_clean, cl_tfs, thread_nb = 1)
    e = time.time()
    print('\n', e-s, 'sec')
    """



    # ---- Normalize by TF -----
    # Normalize the score by TF, to try to counter the frequency problem problem
    scores_by_tf_df, scores_by_dataset_df = utils.normalize_result_file_score_by_tf(output_bed_path, cl_name = parameters['cell_line'])

    # A new result file labeled _normalized has been produced.


#!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!
# TODO ALSO SAY SO  : THIS NORMALIZED FILE IS THE ONE YOU WANT !!!!!!!!!! WELL BOTH ACTUALLY
#!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!#!!!!!!!!!!!!!!!




print('Processing complete.')


# ----------------------------- Diagnostic plots ----------------------------- #

# Only if not artificial data
if not artificial_data:

    # -------- Normalization of score by TF
    # TO CHECK : these plots give info from before the normalization no ?

    # By TF
    fig, ax = plt.subplots(figsize=(10, 8))
    sub_df = scores_by_tf_df.loc[:,['count','50pc']]
    sub_df.plot('count', '50pc', kind='scatter', ax=ax, s=24, linewidth=0) ; ax.grid()
    for k, v in sub_df.iterrows(): ax.annotate(k, v,xytext=(10,-5), textcoords='offset points')
    plt.savefig(plot_output_path+'scores_by_tf.pdf')

    # By dataset
    fig, ax = plt.subplots(figsize=(10, 8))
    sub_df = scores_by_dataset_df.loc[:,['count','50pc']]
    sub_df.plot('count', '50pc', kind='scatter', ax=ax, s=24, linewidth=0) ; ax.grid()
    for k, v in sub_df.iterrows(): ax.annotate(k, v,xytext=(10,-5), textcoords='offset points')
    plt.savefig(plot_output_path+'scores_by_dataset.pdf')



    # TODO MAYBE Remove the dataset plot since we don't normalize by dataset
    """
    """


    # ------ Informative plots for result checking
    # TODO STOCK AND EXPORT ALL THESE PLOTS

    # Those are CONTROL Plots. Explain that.
    # They are not done on our diagnostic, they just allow you to check the results

    # -Including : Compare to the average CRM
    # Produce a picture of the average CRM for later comparisons




    # TODO : THOSE ARE MAYBE ALREADY CALCULATED BY THE Q-SCORE. MERGE THIS CODE WITH THE Q-SCORE CODE TO AVOID REDUNDANCIES ?

    list_of_many_crms = er.get_some_crms(train_generator)


    average_crm_fig, tf_corr_fig, tf_abundance_fig, dataset_corr_fig, dataset_abundance_fig = er.crm_diag_plots(list_of_many_crms, datasets_clean, cl_tfs)

    summed = np.mean(before, axis=0)
    fig, ax = plt.subplots(figsize=(8,8)); sns.heatmap(np.transpose(summed), ax=ax)

    summed = np.mean(clipped_pred, axis=0)
    average_rebuilt_crm_fig, ax = plt.subplots(figsize=(8,8)); sns.heatmap(np.transpose(summed), ax=ax)



# Careful : some horizontal "ghosting" might be due to summed crumbing


"""
SAVE THE ABOVE FIGS SHOMEWHERE
"""


    #
    # max = np.zeros((1,12))
    # N = 1
    # for i in range(N):
    #     current_max = np.max(next(train_generator)[0][0,:,:,:,0], axis=1)
    #     current_max.shape
    #     max = np.concatenate([max,current_max])
    #
    # # Sum, Remove crumbs and reshape
    # sm = (np.sum(max,axis=0) - 0.1*N).reshape((max.shape[1],1))
    # fig, ax = plt.subplots(figsize=(10,1)) ; sns.heatmap(np.transpose(sm) / sum(sm), fmt='.0%', annot = True, cmap = 'Reds') # Numbers
    #
    # # Try a correlation matrix instead
    # max_df = pd.DataFrame(max)
    # fig, ax = plt.subplots(figsize=(10,10)) ; sns.heatmap(max_df.corr(), fmt = '.2f', annot=True, ax=ax)

    # WARNING : the matrix does the have the datasets in the same order as the variable 'datasets', we saw that when trying to look them up with Jeanne. Fix it.





    # NOTE THIS IS DONE ALL THE TIME, NOT JUST IF EVAL IS TRUE


    # # Plot output path
    # plot_output_path = './data/output/diagnostic/'+parameters['cell_line']+'/'
    # if not os.path.exists(plot_output_path): os.makedirs(plot_output_path)


    average_crm_fig.savefig(plot_output_path+'average_crm.pdf')
    tf_corr_fig.savefig(plot_output_path+'tf_corr.pdf')
    dataset_corr_fig.savefig(plot_output_path+'dataset_corr.pdf')
    tf_abundance_fig.savefig(plot_output_path+'tf_abundance.pdf')
    dataset_abundance_fig.savefig(plot_output_path+'dataset_abundance.pdf')




    # ----------------------- Scores per CRM  ------------------------- #

    # VERY IMPORTANT PLOTS

    print('Computing score distribution per number of peaks in CRMs...')

    # TODO the CRM file path should be a parameter in the YAML, it is hardcoded for now
    CRM_FILE = './data/input_raw/remap2018_crm_macs2_hg38_v1_2_selection.bed'
    score_distrib, avg_score_crm, max_score_crm = er.plot_score_per_crm_density(output_bed_path, CRM_FILE)

    score_distrib.save(plot_output_path+'score_distribution.pdf')
    avg_score_crm.save(plot_output_path+'average_score_per_crm_density.pdf')
    max_score_crm.save(plot_output_path+'max_score_per_crm_density.pdf')



    # REDO THIS ON NORMALIZED FILE
    print("... in the normalized file ...")
    output_tfnorm_file = output_bed_path+'_normalized_by_tf.bed'
    score_distrib_tfnorm, avg_score_crm_tfnorm, max_score_crm_tfnorm = er.plot_score_per_crm_density(output_tfnorm_file, CRM_FILE)
    score_distrib_tfnorm.save(plot_output_path+'score_distribution_TFNORM.pdf')
    avg_score_crm_tfnorm.save(plot_output_path+'average_score_per_crm_density_TFNORM.pdf')
    max_score_crm_tfnorm.save(plot_output_path+'max_score_per_crm_density_TFNORM.pdf')











    # ----------------- Scores when known cofactors (or known non-cofactors) are present

    # Work on NORMALIZED scores

    #atypeak_result_file_normalized = './data/output/bed/jurkat.bed_normalized_by_tf.bed'
    atypeak_result_file_normalized = output_bed_path + "_normalized_by_tf.bed"
    crm_file_path = "./data/input_raw/remap2018_crm_macs2_hg38_v1_2_selection.bed"
    # TODO UNHARDCODE THE CRM FILE PATH OF AT LEAST PUT IT AT THE BEGINNING !!!!!


    # TODO MAKE THE TF s OF HERE parameters !!!! DONE

    tfs_to_plot = parameters['tf_pairs']


    # THOSE ARE THE K562 TFs I THINK
    tfs_to_plot = [['ctcf','rad21'],['fos','jun'],['irf1','atf1']]
    tfs_to_plot = [['ctcf','jun'],['gata1','gata2']]

    for pair in tfs_to_plot:
        # TODO CAREFUL ABOUT CASE !!
        tf1, tf2 = pair
        p, _ = er.get_scores_whether_copresent(tf1, tf2, atypeak_result_file_normalized, crm_file_path)
        p.save(plot_output_path+"tfdiag_"+tf1+"_"+tf2+".pdf")
