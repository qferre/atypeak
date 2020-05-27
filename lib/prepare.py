import random, os

"""
Some functions used in train.py and process.py to prepare models or generators.

Used here to get those that are common and functionalize code.
"""

# ------------------------------ Some parameters ----------------------------- #

def get_save_model_path(parameters, root_path):
    # Path for saving the model later
    if not parameters['use_artificial_data'] :
        save_model_path = root_path+"/data/output/model/"+parameters['cell_line']+"_trained_model"
    else:
        save_model_path = root_path+"/data/output/model/ARTIFICIAL_DATA_trained_model"
    return save_model_path



# Plot output path (different for artificial data)
def get_plot_output_path(parameters):
    plot_output_path = './data/output/diagnostic/'+parameters['cell_line']+'/'
    if parameters['use_artificial_data'] : plot_output_path += 'artificial/'
    if not os.path.exists(plot_output_path): os.makedirs(plot_output_path)
    return plot_output_path



def get_indexes(parameters, crmid, datasets, cl_tfs):

    # Datasets: might wish to use a parent name later
    #datasets_clean_ori = [dr.dataset_parent_name(d) for d in datasets] 
    # Make unique while preserving order, which a `set` would not do
    datasets_clean = sorted(list(set(datasets)), key=datasets.index) 


    ## Prepare the indexes
    # Different for artificial and real data
    if parameters['use_artificial_data'] :
        print('Using artificial data of dimensions : '+str(parameters['artificial_nb_datasets'])+' x '+str(parameters['artificial_nb_tfs']))

        # Also override the datasets and TF names, replace them with random
        datasets_clean = ["dat_"+str(i) for i in range(parameters['artificial_nb_datasets'])]
        cl_tfs = ["tf_"+str(i) for i in range(parameters['artificial_nb_tfs'])]

        all_matrices = [None]

    else:
        # Collect all CRM numbers (each one is a *sample*)
        matrices_id = crmid.values()

        # The list of matrices ID must be made only of unique elements. 
        # The order does not matter in and of itself, but must be randomized and the same for a given random seed, as if affects training
        seenid = set() ; all_matrices = [x for x in matrices_id if x not in seenid and not seenid.add(x)]
        random.shuffle(all_matrices) # The random seed defined in main process impacts this

        print("Using real data for the '"+parameters["cell_line"]+"' cell line.")

    return datasets_clean, cl_tfs, all_matrices




def configure_tensorflow_session(seed, disable_tensorflow_warnings = True):
    """
    Macro to create and configure a Tensorflow session for reproducibility.
    Was moved to allow subprocesses to create their own TF sessions.
    """
    # TODO Use this in multiprocessing result creation as well ? Should not be
    # needed since I am just reloading a model

    # Disable INFO and WARNING messages
    if disable_tensorflow_warnings:
        print('Tensorflow INFO and WARNING messages are disabled.')
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # Errors only
        os.environ['KMP_WARNINGS'] = 'off' # Also disable OMP warnings

    import tensorflow as tf
    import keras.backend as K

    # Check TensorFlow version
    # NOTE We enforce Tensorflow<2 in the conda env for now, but we support TF2+
    # But it might be slower and different as the code was originally written in TF 1.14
    from distutils.version import LooseVersion
    USING_TENSORFLOW_2 = (LooseVersion(tf.__version__) >= LooseVersion("2.0.0"))

    # NOTE May need to disable GPU for absolute reproducibility 
    # Not a problem since models are comparatively simple : the big bottleneck is
    # actually the file reading and matrix unsparsing IS IT THOUGH ?
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    # TODO determine it once and for all

    if not USING_TENSORFLOW_2:
        tf.set_random_seed(seed)
        config = tf.compat.v1.ConfigProto()
        #config = tf.compat.v1.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth=True # Be parcimonious with RAM usage if on a GPU
        tf.get_logger().setLevel('INFO') # Disable INFO, keep only WARNING and ERROR messages

        # Disable Tensorflow internal multithreading, 
        # NOTE this was key for reproducibility
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1

        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph(), config=config)
        K.set_session(sess)

    if USING_TENSORFLOW_2:

        tf.random.set_seed(seed) 

        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)







# ----------------------------- Model preparation ---------------------------- #

def prepare_model_with_parameters(parameters, nb_datasets_model, nb_tfs_model, root_path = None):
    """
    A wrapper function that will prepare an atyPeak model given the current parameters.
    Non-pure, since it depends on the rest of the code. WAIT IT IS PURE NOW NO ?
    """

    import keras
    import lib.model_atypeak as cp

    # Optimizer : Adam with custom learning rate
    optimizer_to_use = getattr(keras.optimizers, parameters["nn_optimizer"])
    opti_custom = optimizer_to_use(lr=parameters["nn_optimizer_learning_rate"])


    # Parameters checking
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
    # TODO Make a 2d matrix of weights instead, one weight for each specific tf+dataset pair.
    # See draft code in the `cp.create_weighted_mse()` function source.


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

    # Print summary of the model (only if not artificial)
    if (not parameters['use_artificial_data']) and (root_path is not None) :
        with open(root_path+'/data/output/model/'+parameters['cell_line']+'_model_architecture.txt','w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file = root_path+'/data/output/model/'+parameters['cell_line']+'_model_architecture.png',
            show_shapes = True, show_layer_names = True)

    return model
