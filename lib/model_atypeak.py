import sys, time

import numpy as np
import pandas as pd
import scipy.sparse

import keras
from keras.models import Sequential, Model
from keras.layers import Conv3D, MaxPool3D, UpSampling3D, ZeroPadding3D
from keras.layers import Input, Dense, Dropout, Lambda
from keras import regularizers
from keras import backend as K

from skimage.filters import gaussian as gaussian_filter

import lib.utils as utils


# ------------------------------ Reading data -------------------------------- #

def generator_unsparse(matrices_keys, batch_size, matrix_generating_call,
        pad_to, crumb = 0.1, squish_factor = 10, debug_print = False):
    """
    Generator that reads the data files and generates a matrix.
    ARGUMENTS :
        - matrices_keys : the list of all crm_id
        - batch_size : size of the batch
        - matrix_generating_call : a function that must generate the corresponding matrix.
            Will be called as matrix_generating_call(crm_id)

        - pad_to : matrices will always be padded to this. Must be a multiple of
            the smallest number of filters in the model !
        - crumb : if not None, will add its value as a crumb to positions where
            a peak has been observed but not in this dataset (see add_crumbing documentation)

        - squish_factor : diminish matrix scale along X axis by this much. Saves computing power.
    """

    while True:
        # Choose random indexes in features (as many as the batch size)
        indexes = np.random.choice(len(matrices_keys),size=batch_size)

        batch_features = list()

        for i in indexes:
            # Get the query arguments and generate the matrix

            # NOTE Stacking all TFs along a third dimension is done in the
            # matrix generating call (in data_read.py), not here !
            # Instead we will just supply the crm_id and unsparse the matrix
            crm_id = matrices_keys[i]
            X = matrix_generating_call(crm_id=crm_id)

            if debug_print : print(crm_id)

            if crumb != None :
                # To counter sparsity, add a crumbs (see function documentation)
                X = utils.add_crumbing(X, crumb)

            batch_features.append(X)
           

        # Pad with zeroes
        def pad_batch(arr):
            padded = [utils.zeropad(x, pad_width = ((3200 - x.shape[0],0),(0,0),(0,0))) for x in arr]
            return np.array(padded)

        result = pad_batch(batch_features)

        # We return a tuple duplicate of result, because generators must always
        # return (data, target), ie. the (X,Y) tuple.

        # Add the meaningless 'channel' dimension
        result = result[...,np.newaxis]

        # Remove completely empty matrices
        summed = np.apply_over_axes(np.sum, result, [1,2,3,4])
        idx = np.flatnonzero(summed)
        result = result[idx,:,:,:,:]

        # Squish the matrices along their X axis (region_length)
        # Due to the nature of our data (peaks/regions), we can use this to
        # reduce computational cost with negligible information loss
        result = [utils.squish(m, squish_factor) for m in result] ; result = np.array(result)

        # As all targets (result is the input of the model, target is the 
        # desired output) are simply a copy of the result, we can simply 
        # copy if after squishing
        target = np.copy(result)
       
        # # Override : we can call a function here that modifies results and targets before they are
        # # supplied.
        # result, target = call_override(result, target)
        # # This function must have a signature or result, target = f(result, target)
        # # where result and targets are 5D numpy arrays (lists of 4D numpy arrays of shape region_length,nb_datasets,nb_tfs,1) ;
        # # Remember that the result and target are in respective order (ie. target[0] is the target for result[0])

        yield (result,target)










################################################################################
# ---------------------------------- MODEL ----------------------------------- #
################################################################################

def create_weighted_mse(datasets_weights, tf_weights):
    """
    Creates a weighted MSE.
    The weights should be a list for the dimension 1 (datasets) and 2 (TF).
    The MSE tensor will be multiplied by those values. For example, if a TF
    has a weight of 3, the corresponding loss will be multiplied by 3.
    """

    datasets_weights_arr = np.array(datasets_weights)
    datasets_weights_arr = datasets_weights_arr.reshape((1,len(datasets_weights),1))
    tf_weights_arr = np.array(tf_weights)
    tf_weights_arr = tf_weights_arr.reshape((1,1,len(tf_weights)))

    # Turn to tensors and expand with relevant axis.
    D = K.variable(datasets_weights_arr, dtype=K.floatx())
    T = K.variable(tf_weights_arr, dtype=K.floatx())

    def weighted_loss(y_true, y_pred):
        raw_mse = K.mean(K.square(y_pred - y_true), axis=-1) 
        # Result is 3D : the last meanigless channel has been averaged over
        result = raw_mse * D * T
        return result

    # TODO Separate pos and neg for differential weighting.
    # TODO To use a 2d matrix of weights instead with one specific weight per 
    # source (TF+dataset pair), simply do something like this:
    # raw_mse = np.ones((10000,3,4)) # The data
    # w = np.array([[1,1,1,2],[2,2,2,1],[1,1,2,2]]) # the weights. IMPORTANT : shape is (nb_datasets, nb_tfs)
    # result = raw_mse * w # Perform the multiplication in that order !
    # utils.plot_3d_matrix(result) # Check that it worked

    return weighted_loss # Return the function so you can use it in the encoder



# Callback to potentially stop the model as soon as a given loss value is reached
class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='loss', value=0.00001, verbose=0, patience = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor ; self.value = value
        self.verbose = verbose
        self.patience = patience ; self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None: print("Early stopping requires %s available!" % self.monitor)
        if current < self.value: self.wait += 1
        if self.wait >= self.patience:
            if self.verbose > 0: print("Epoch %05d: early stopping after patience for THR" % epoch)
            self.model.stop_training = True






# ----------------------------- The model itself ----------------------------- #
def create_atypeak_model(nb_datasets, region_size, nb_tfs,
            kernel_nb,kernel_width_in_basepairs, reg_coef_filter,
            pooling_factor, deep_dim,
            optimizer= 'adam', loss = 'mse'):
    """
    Creates and return our custom atypical peak detection model.

    PARAMETERS :

        See details in the code for the parameter values.

        # General model
        optimizer : either a string or the optimizer itself
        loss : either a string or the loss function itself
    """

    # 'CRE-Telomer' padding
    # The data is not an image. The padding introduced later can mask informations
    # about peaks on the flanks (we observed they systematically were poorly rebuilt)
    # This flank seemed to be a problem for roughly 2* kernel_size so we add 
    # a padding of zeroes, to be removed at the end. 
    telomer_len = 2* kernel_width_in_basepairs

    telomer_pad = (
                (telomer_len,telomer_len),
                (0,0),
                (0,0)
            )

    # Pool only on the region size dimension
    filter_pooling = (pooling_factor,1,1)

    ## Kernel sizes
    # Size in dimensions not processed must be equal to shape so there is no convolution here

    # The first convolution is made on datasets only, the second on TFs
    kernel_size_d = (kernel_width_in_basepairs,nb_datasets,1)

    # We use shorter kernels for the TFs combinations
    k=2 # TODO Don't hardcode this ?
    kernel_size_t = (int(kernel_width_in_basepairs/k),1,nb_tfs)


    # After the first CNNs, dim2=dim3=1 since the filter is as large as the
    # image and there is no padding, so subsequent CNNs must use different filters (1D)
    kernel_size_inner = (kernel_width_in_basepairs,1,1)
    # Currently used only in one layer of convo decoding AFAIK

    # The input is 3D ; that will require flattening and no padding on axis 2 and 3 above, which makes sense since the filters cover all the image on these
    # axes, so their dimension is indeed 1, and they can be squeezed (length of only 1) to flatten the tensor
    # To do that : use ZeroPadding3D, then padding='valid' (ie no padding) on the CNN
    padding = (
            (int(kernel_width_in_basepairs/2),int(kernel_width_in_basepairs/2)-1),
            (0,0),
            (0,0)
        )

    padding_shorter = (
            (int(kernel_width_in_basepairs/(2*k)),int(kernel_width_in_basepairs/(2*k))-1),
            (0,0),
            (0,0)
        )


    # For the DECODER kernels are not the same
    kernel_size_decoder = (kernel_width_in_basepairs,deep_dim,1)

    # The input tensor will be : (batch, region_size_or_pooled_size, nb_datasets, nb_tf, channels)
    # Although 'channels' is a meaningless dimension here
    input = Input(shape=(region_size,nb_datasets,nb_tfs,1))




    ## ----- ENCODER

    ## Convolutional layers
    # NOTE Convolutions are made on each axis separtely as the represent info
    # of a different nature, and because 3D convos has trouble learning all at once
    # Only 1 CNN layer each because higher order combis are not what we want ;
    # but the Dense layers (plural) can learn combis of combis

    # Apply 'Telomer' padding
    xpadded = ZeroPadding3D(telomer_pad, name = 'pad')(input)

    # First convolution : datasets only
    x = Conv3D(kernel_nb, kernel_size_d, padding='valid',
        kernel_regularizer=regularizers.l2(reg_coef_filter))(xpadded)
    x = ZeroPadding3D(padding)(x)
    x = MaxPool3D((filter_pooling), padding='valid')(x)

    # Second convolution : TFs
    x = Conv3D(kernel_nb, kernel_size_t, padding='valid',
        kernel_regularizer=regularizers.l2(reg_coef_filter))(x)
    x = ZeroPadding3D(padding_shorter)(x)
    x = MaxPool3D((filter_pooling), padding='valid')(x)


    # Reshape
    x = Lambda(lambda m: K.squeeze(K.squeeze(m,2),2),
            output_shape = lambda input_shape: (input_shape[0],input_shape[1],input_shape[-1])
            )(x) # Reshape


    ## Dense layers provide the "deep" muscle for the model
    # Due to how Keras works, only the last dimension provides weights.
    # This means that, in effect, this layer behaves like a TimeDistributed Dense :
    # There is no communication along the X dimension, only along the "filters" dimension from the previous layer !
    # NOTE This is a change from earlier Keras behavior, which would Flatten. Confirmed by counting the parameters in model.summary()
    # This is UNLIKE what is stated in the doc.
    x = Dense(deep_dim, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(deep_dim, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(deep_dim, activation = 'relu')(x)
    x = Dropout(0.1)(x)


    """
    # TODO Try integrating some LSTM layers, and use many of them in each layer. Although having too many LSTMs may make the information budget too high. Play with budget and adjust consequenty.
    # Add this before the encoded layer, so this layer gets info that the LSTMs have learned.
    # This would obviously require reshaping the output of the latest CNN because LSTMs only get 2d input (not counting the batch_size dimension)
    # So you would have 1 value at each position for the last filter, with the timesetps being the position along the genome

    # Use LSTM layer with return sequences, to get (timesteps, lstm_number) as output dim
    # Then a time distributed Dense(deep_dim) to get (timesteps, deep_dim) as output
    if use_lstm :
        x = LSTM(latent_dim, return_sequences = True)(x)
        x = LSTM(latent_dim, return_sequences = True)(x) # Stack 2 LSTMs for giggles.
        
        # This one will be the encoded layer actually, otherwise I'm just making a new layer of combis and losing the timestep
        x = TimeDistributed(Dense(deep_dim))(x) 
    """

    # Layer of the encoded representation (no regularisation)
    x = Dense(deep_dim, activation = 'relu',
        name = 'encoded')(x)



    ## ----- DECODER

    x = Lambda(lambda m: K.expand_dims(K.expand_dims(m,3),4),
        output_shape = lambda input_shape: input_shape + (1,1)
        )(x) # Reshape
    x = Conv3D(kernel_nb, kernel_size_decoder, padding='valid')(x)
    x = ZeroPadding3D(padding)(x)
    x = UpSampling3D((filter_pooling))(x)

    x = UpSampling3D((filter_pooling))(x)

    # Essentially, one filter per {TF*dataset}
    x = Conv3D(nb_datasets*nb_tfs, kernel_size_inner, padding='same')(x)

    # Reshaping
    x = Lambda(lambda m: K.squeeze(K.squeeze(m,2),2),
        output_shape = lambda input_shape: (input_shape[0],input_shape[1],input_shape[-1])
        )(x)
    x = Lambda(lambda m: K.reshape(m,(K.shape(m)[0],K.shape(m)[1],nb_datasets,nb_tfs)),
        output_shape = lambda input_shape: (input_shape[0],input_shape[1],nb_datasets,nb_tfs)
        )(x)
    decoded = Lambda(lambda m: K.expand_dims(m,4),
        output_shape = lambda input_shape: input_shape+(1,)
        )(x)


    # Reverse the 'telomer' padding :
    final_layer = Lambda(lambda m: m[:,telomer_len:-telomer_len,:,:,:], name = 'unpad')(decoded)


    # Concatenate both encoder and decoder
    autoencoder = Model(input, final_layer)

    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder








# ---------------------------- Interpretability ------------------------------ #

def compute_max_activating_example_across_layer(model,
                            selected_layer,
                            output_of_layer_select_this_dim = 1,
                            learning_rate = 2,
                            random_state = 42,
                            nb_steps_gradient_ascent = 20,
                            blurStdX = 0, blurStdY=0, blurStdZ=0, blurEvery = 4,
                            ur_example_input_shape = None,
                            debug_print = True
                            ):
    """
    model : a Keras model
    selected_layer : the layer on which to perform gradient ascent
    learning_rate & nb_steps_gradient_ascent : parameters for gradient ascent
    blurStdX & blurStdY=0 & blurEvery = 4 : paramters to blur the results to make them more natural

    Return a list : for each class, gives an artificial example for which the
    selected layer best says "this example belong to class X"

    Derived from Keras' blog.

    Use this to visualize what each neuron in a hidden layer is looking for, most notably used for the encoded dimension.
    """

    max_activation_class=[]
    losses=[]


    # Specify input and output of the network
    layer_input = model.layers[0].input
    layer_output = model.layers[selected_layer].output

    # Get the size of the output of the selected layer. Usually 1 since the layer
    # would have shape of (batch_size, layer_size) but can be different for certain layers
    layer_output_size_of_output = int(layer_output.shape[output_of_layer_select_this_dim])


    np.random.seed(random_state)  # for reproducibility

    for class_index in range(layer_output_size_of_output): # For each class
        start_time = time.time()

        # Build a loss function that maximizes
        # the activation of the neuron for the chosen class
        loss = K.mean(layer_output[..., class_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, layer_input)[0]

        # Normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        # also add a flag to disable the learning phase (in our case dropout)
        iterate = K.function([layer_input, K.learning_phase()], [loss, grads])

        # we start from a gray image with some random noise
        if ur_example_input_shape : # if there is a custom input shape, notably for networks with variable input size
            input_data = np.random.uniform(0, 1, (1,) + ur_example_input_shape) # (1,) for batch axis
        else: # Else try to query the shape (fixed one)
            input_data = np.random.uniform(0, 1, (1,) + model.input_shape[1:]) # (1,) for batch axis

        # we run gradient ascent for n steps
        for i in range(nb_steps_gradient_ascent):
            loss_value, grads_value = iterate([input_data, 0]) # Disable the learning phase (useful when the model has a different behavior when learnig, e.g. using Dropout or BatchNormalization)
            input_data += grads_value * learning_rate # Apply gradient to image

            # Add some blur to get more natural results
            if (blurStdX+blurStdY+blurStdZ) is not 0 and i % blurEvery == 0 :
                input_data[0,:,:,:,0] = gaussian_filter(input_data[0,:,:,:,0], sigma=np.array([blurStdX, blurStdY, blurStdZ]))

            # Control print giving the loss value at each step (in our case, the activation)
            if debug_print:
                sys.stdout.write("\r" + 'Dimension '+str(class_index)+' -- Activation = ' + str(np.mean(loss_value))) ; sys.stdout.flush()

        # decode the resulting input image and add it to the list
        max_activation_class.append(input_data[0])
        losses.append(loss_value)
        end_time = time.time()
        if debug_print:
            print('\nOutput dimension %d processed in %ds' % (class_index, end_time - start_time))


    # Return the processed list of optimal class examples
    return max_activation_class




# Given a before matrix (or simply a CRM) this function will output its encoded representation to understand the combinations at play.
def get_encoded_representation(before_matrix, model, ENCODED_LAYER_NUMBER = 15, disable_learning = True):

    # Disable the learning phase behavior (e.g. Dropout)
    if disable_learning: learnflag = 0
    else: learnflag = 1

    get_kth_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[ENCODED_LAYER_NUMBER].output])
    result = get_kth_layer_output([before_matrix, learnflag])[0]

    return result
