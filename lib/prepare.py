
"""
Some functions used in train and process to prepare models or generators


Used here to get those that are common and functionalize code
"""




def prepare_model_with_parameters(parameters, nb_datasets_model, nb_tfs_model, root_path = None):
    """
    A wrapper function that will prepare an atyPeak model given the current parameters.
    Non-pure, since it depends on the rest of the code. WAIT IT IS PURE NOW NO ?
    """


    # WILL THIS BUG THE MULTITHREADING ?? APPRENTLY NOT
    import keras
    import lib.model_atypeak as cp

    # Optimizer : Adam with custom learning rate
    optimizer_to_use = getattr(keras.optimizers, parameters["nn_optimizer"])
    opti_custom = optimizer_to_use(lr=parameters["nn_optimizer_learning_rate"])


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



    # Print summary of the model (only if not artificial)
    if (not parameters['use_artificial_data']) and (root_path is not None) :
        with open(root_path+'/data/output/model/model_'+parameters['cell_line']+'_architecture.txt','w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))


    return model
