import numpy as np
import scipy.sparse

import keras
from keras.models import Sequential, Model
from keras.layers import Conv3D, MaxPool3D, UpSampling3D, ZeroPadding3D
from keras.layers import Input, Dense, Dropout, Lambda
from keras import regularizers
from keras import backend as K


from lib import utils





# ------------------------------ Reading data -------------------------------- #
# Utility funcions preparing the raw read data for use in the model


def look_here_stupid(matrix, crumb = 0.1):
    """
    To counter sparsity, we add crumbs to matrices of peak presence.
    Put a crumb (default 0.1) on all nucleotides where a peak was found in at
    least one dataset. The crumb is much lower than true peak score, and
    prevents the model from always learning zeroes.
    """

    if len(matrix.shape) != 3:
        raise TypeError('ERROR - Trying to crumb a non-3D matrix.')


    result = matrix.copy()

    # Get the indices of all non-zero elements of the array
    nonzero = np.nonzero(matrix)

    for i in range(nonzero[0].shape[0]):
        x,y,z = nonzero[0][i], nonzero[1][i], nonzero[2][i]
        val = matrix[x,y,z]

        # Add crumbs
        result[x,y,:] += crumb * val # Datasets
        result[x,:,z] += crumb * val # Transcription factors

    return result










def generator_unsparse(matrices_keys, batch_size, matrix_generating_call,
        pad_to, crumb = 0.1, squish_factor = 10, debug_print = False ):
    """
    Generator that reads the data files and generates a matrix.
    ARGUMENTS :
        matrices_keys : the list of all crm_id
        batch_size : size of the batch
        matrix_generating_call : a function that must generate the corresponding matrix.
            Will be called as matrix_generating_call(crm_id)

        pad_to : matrices will always be padded to this. Must be a multiple of
        the smallest number of filters in the model !
        crumb : if not None, will add its value as a crumb to positions where
        a peak has been observed but not in this dataset (see look_here_stupid documentation)

        squish_factor : diminish matrix scale along X axis by this much. Saves computing power.
    """

    while True:
        # Choose random indexes in features (as many as the batch size)
        indexes = np.random.choice(len(matrices_keys),size=batch_size)

        batch_features = list()
        batch_status = list()


        # NOTE I already tried multiprocessing this, it slowed it, likely due to disk reading overhead.


        for i in indexes:
            # Get the query arguments and generate the matrix

            # NOTE : stacking all TFs along a third dimension is done in the
            # matrix generating call (in data_read.py), not here !
            # Instead we will just supply the crm_id and unsparse the matrix
            crm_id = matrices_keys[i]
            X = matrix_generating_call(crm_id=crm_id)

            if debug_print : print(crm_id)


            if crumb != None :
                # To counter sparsity, add a crumbs (see function documentation)
                X = look_here_stupid(X, crumb)

            batch_features.append(X)
            batch_status.append('data')


            """
            # Override : from time to time, return a 'noise -> empty' pair
            # to train the network to discard noise
            if rand < p :
                noisy_shape = X.shape
                noisy = noise_generating_call(pad_to,noisy_shape[1],noisy_shape[2])

                # TODO Define parameters dynamically based on data

                batch_features.append('noisy')
                batch_status.append('noise')
            """

        # Pad with zeroes
        def pad_batch(arr):
            #arr = np.array(arr) # Force to numpy array to be safe
            #padded = [np.pad(x,pad_width = ((pad_to - x.shape[0],0),(0,0),(0,0)), mode='constant', constant_values=0) for x in arr]
            padded = [utils.zeropad(x,pad_width = ((3200 - x.shape[0],0),(0,0),(0,0))) for x in arr]

            return np.array(padded)

        result = pad_batch(batch_features)

        # We return a tuple duplicate of result, because generators must always
        # return (data, target), ie. the (X,Y) tuple.

        # Add the meaningless 'channel' dimension
        result = result[...,np.newaxis]



        """
        import time

        import numpy as np
        x = np.ones((2500,20,20))

        s = time.time()
        for i in range(1000):
            xpadded = np.pad(x,pad_width = ((3200 - x.shape[0],0),(0,0),(0,0)),
                mode='constant', constant_values=0)
        e = time.time()
        print(e-s)

        s = time.time()
        for i in range(1000):
            xpadded2 = utils.zeropad(x,pad_width = ((3200 - x.shape[0],0),(0,0),(0,0)))
        e = time.time()
        print(e-s)

        np.array_equal(xpadded, xpadded2)

        """





        ### Now prepare the targets
        target = list()
        for i in range(len(result)) :
            if batch_status[i] == 'data' : target.append(result[i]) # The target is the data
            if batch_status[i] == 'noisy' : target.append(np.zeros(result[i].shape)) # This was just noise. The target should be an empty matrix
        target = np.array(target)

        # Remove completely empty matrices
        # TODO maybe also remove matrices with one or txo peaks only
        summed = np.apply_over_axes(np.sum, result, [1,2,3,4])
        idx = np.flatnonzero(summed)
        result = result[idx,:,:,:,:]
        target = target[idx,:,:,:,:]

        # Squish the matrices along their X axis (region_length)
        # Due to the nature of our data (peaks/regions), we can use this to
        # reduce computational cost with negligible information loss
        """
        result = utils.squish(np.array(result), squish_factor, squishing_a_batch = True)
        target = utils.squish(np.array(target), squish_factor, squishing_a_batch = True)
        """

        result = [utils.squish(m, squish_factor) for m in result] ; result = np.array(result)

        # TODO Time saver : squishing takes some time, so only do it if the target is different from result !
        # If all elements of batch_status are 'data', it means the target was NOT modified
        # and as such there is no need to squish it : we can simply copy the `result` batch array
        # TODO use it above as well
        if all(status == 'data' for status in batch_status):
            target = np.copy(result)
        else:
            target = [utils.squish(m, squish_factor) for m in target] ; target = np.array(target)


        """
        # Override : we can call a function here that modifies results and targets before they are
        # supplied.
        # Mainly used in diagnostic and debug
        # result, target = call_override(result, target)

        # Add to doc : this function must have a signature or result, target = f(result, target)
        # where result and targets are 5D numpy arrays (lists of 4D numpy arrays of shape region_length,nb_datasets,nb_tfs,1) ;
        # please note that the result and target are in respective order
        """

        yield (result,target)

































































################################################################################
# ---------------------------------- MODEL ----------------------------------- #
################################################################################

# Potential weigted loss

def create_weighted_mse(datasets_weights, tf_weights):

    # Creates a weighted MSE.
    # The weights should be a list for the dimension 1 (datasets) and 2 (tf).
    # The MSE tensor will be multiplied by those values. For example, if a TF
    # has a weight of 3, the corresponding loss will be multiplied by 3.

    datasets_weights_arr = np.array(datasets_weights).reshape(1,len(datasets_weights),1)
    tf_weights_arr = np.array(tf_weights).reshape(1,1,len(tf_weights))

    # Turn to tensors and expand with relevant axis.

    D = K.variable(datasets_weights_arr, dtype=K.floatx())
    T = K.variable(tf_weights_arr, dtype=K.floatx())

    def weighted_loss(y_true, y_pred):
        raw_mse = K.mean(K.square(y_pred - y_true), axis=-1) # Result is 3D : the last meaniless channel has been averaged over
        result = raw_mse * D * T
        return result

    # TODO : must separate pos and neg for additional weighting (loss due to added phantoms and loss due to removed peaks)

    # TODO : to use a 2d matrix of weights instead with one specific weight per series (TF+dataset pair), simply do
    # something like this :
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
        if current is None: warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if current < self.value: self.wait += 1
        if self.wait >= self.patience:
            if self.verbose > 0: print("Epoch %05d: early stopping after patience for THR" % epoch)
            self.model.stop_training = True





















# TODO REPLACE "model_atypeak" EVERYWHERE WITH "ATYPEAK" OR "MODEL"






def create_model_atypeak_model(nb_datasets, region_size, nb_tfs,
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


    """
    # nb_datasets = 10
    # region_size = 320
    # nb_tfs = 8
    # kernel_nb = 16
    # kernel_width_in_basepairs = 20
    # reg_coef_filter = 0.01
    # pooling_factor = 2
    # optimizer= 'adam'
    # loss = 'mse'
    # deep_dim = 32
    #

    #
    # # As I have removed padding, pad_to must be at least 4 times higher than kernel size and
    # # pad_to/2 - kernsl_siz, etc... must be an integer.
    # dividend = ((region_size-kernel_width_in_basepairs+1)/2-kernel_width_in_basepairs+1)/2
    # if (int(dividend) != dividend):
    #     raise ValueError("When creating the model ((region_size-kernel_width_in_basepairs+1)/2-kernel_width_in_basepairs+1)/2 must be integer, it is currently "+ str(dividend))
    #
    #
    # NO ! To remove flank problem, a better idea is to pad by ONE nucleotide to ensure it is divisible by 2 each time if kernel_width_in_basepairs is even.
    # Or simply make kernel_width_in_basepairs equal to itself minus one if even ?
    # And simply make sure that region_size is divisible by 4 !
    #
    # Wait. Do I even need to take care about that ?
    #
    """





    # 'CRE-Telomer' padding
    # The data is not an image. The padding introduced later can mask informations
    # about peaks that are on the flanks. (W we observed they systematically were poorly rebuilt)
    # This flank seemed to be a problem for roughly 2* kernel size NO IT WAS MORE ACTUALLY, but we add more to be safe.
    # So to the data we add a padding of 20% at the left and 20% at the right
    # Then we remove this padding at the end.
    telomer_len = int(region_size/5)


    telomer_len = 2* kernel_width_in_basepairs


    telomer_pad = (
                (telomer_len,telomer_len),
                (0,0),
                (0,0)
            )

    # To apply it :
    # y = ZeroPadding3D(padding)(first_layer)

    # TO REVERSE IT :
    # final_layer = Lambda(lambda x: x[telomer_len:-(telomer_len-1),0,:,:])(x)
    #



    # TODO REMOVE USELESS PADDINGS

    # Pool only on the region size dimension
    filter_pooling = (pooling_factor,1,1)

    ##  Kernel sizes
    #kernel_size_d = (kernel_width_in_basepairs,nb_datasets,nb_tfs) # Size in dimensions 2 and 3 must be equal to shape so there is no convolution here



    # The first convolution is made on datasets only, the second on TFs
    kernel_size_d = (kernel_width_in_basepairs,nb_datasets,1)

    # We use shorter kernels for the TFs combinations
    k=2 # TODO DON'T HARDCODE THIS ?
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
    # reverse_padding = (
    #         (0,0),
    #         (int(nb_datasets/2),int(nb_datasets/2)-1),
    #         (int(nb_tfs/2),int(nb_tfs/2)-1)
    #     )
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





    ## ENCODER

    ## Convolutional layers
    # NOTE Convolutions are made on each axis separtely as the represent info
    # of a different nature, and because 3D convos has trouble learning all at once
    # Only 1 CNN layer each because higher order combis are not what we want ;
    # but the Dense layers (plural !!) can learn combis of combis just fine with *deep* learning

    # TODO : ZERO PADDING IS USELESS IN ENCODING, REMOVE IT
    # todo : remove padding in the model bc it's useless, but make sure that kernel size is a MULTIPLE of padding to prevent size rounding problems
    # Finally we keep it and use telomers instead

    """
    # Source : https://github.com/keras-team/keras/issues/6676
    # Rewrite the original method to make its shapes become int type:
    from keras.layers import UpSampling2D
    from keras.utils.generic_utils import transpose_shape
    class UpSamplingUnet(UpSampling2D):
    def compute_output_shape(self, input_shape):
        size_all_dims = (1,) + self.size + (1,)
        spatial_axes = list(range(1, 1 + self.rank))
        size_all_dims = transpose_shape(size_all_dims,
                                        self.data_format,
                                        spatial_axes)
        output_shape = list(input_shape)
        for dim in range(len(output_shape)):
            if output_shape[dim] is not None:
                output_shape[dim] *= size_all_dims[dim]
                output_shape[dim]=int(output_shape[dim])
        return tuple(output_shape)

    # Then alter UpSampling2D(size=us_size) to UpSamplingUnet(size=us_size)
    """






    # Apply 'Telomer' padding
    xpadded = ZeroPadding3D(telomer_pad, name = 'pad')(input)








    # First convolution : datasets only
    x = Conv3D(kernel_nb, kernel_size_d, padding='valid',
        kernel_regularizer=regularizers.l2(reg_coef_filter))(xpadded)
        #activity_regularizer=regularizers.l2(reg_coef_filter)# ADD THIS so learning filters {A}+{B} is more expensive later on than filter {AB}
    x = ZeroPadding3D(padding)(x)
    x = MaxPool3D((filter_pooling), padding='valid')(x)


    # TRY : reuse this layer, but with LESS filters. The goal is not to learn high
    # order combis of combis, but to reduce dimension so the TF layer can learn, well, the TFs,
    # instead of combis of combis.
    # The result would be a see-saw : 64 filters dataset -> 16 filters dataset -> 64 filters TF -> 16 filters TF
    # x = Conv3D(kernel_nb, kernel_size_inner, padding='valid',
    #     kernel_regularizer=regularizers.l2(reg_coef_filter))(x)
    #     #activity_regularizer=regularizers.l2(reg_coef_filter)
    # x = ZeroPadding3D(padding)(x)
    # x = MaxPool3D((filter_pooling), padding='valid')(x)

    # Second convolution : TFs.

    x = Conv3D(kernel_nb, kernel_size_t, padding='valid',
        kernel_regularizer=regularizers.l2(reg_coef_filter))(x)
        #activity_regularizer=regularizers.l2(reg_coef_filter)
    x = ZeroPadding3D(padding_shorter)(x)
    x = MaxPool3D((filter_pooling), padding='valid')(x)

    # x = Conv3D(kernel_nb, kernel_size_inner, padding='valid',
    #     kernel_regularizer=regularizers.l2(reg_coef_filter))(x)
    #     #activity_regularizer=regularizers.l2(reg_coef_filter)
    # x = ZeroPadding3D(padding)(x)
    # x = MaxPool3D((filter_pooling), padding='valid')(x)





    # Reshape
    x = Lambda(lambda m: K.squeeze(K.squeeze(m,2),2),
        output_shape = lambda input_shape: (input_shape[0],input_shape[1],input_shape[-1])
        )(x) # Reshape



        # IMPORANT NOTE : THIS DENSE THIS ORT OF BEHAVES LIKE A DISTRIBUTED, NO ? SINCE OUTPUT OF PREVIOUS LAYER IS 2D ?
        # YEP IT DOES; There is no




    # Dense layers provide the "deep" muscle for the model


    # NOTE : TO DOUBLE CHECK
    # Due to how Keras works, only the last dimension provides weights. the previous
    # input is 2D per element, so the weights are calculated only on the last dimension.
    # This means that, in effect, this layer behaves like a TimeDistributed Dense :
    # There is no communication along the X dimension, only along the "filters" dimension from the previous layer !

    # This is a hange from earlier Keras behavior, which would flatten the input prior to a Dense layer.
    # This can be confirmed by counting the parameters in the exported model.summary()

    # This is UNLIKE what is stated in the doc. Source : https://stackoverflow.com/questions/52089601/keras-dense-layers-input-is-not-flattened

    x = Dense(deep_dim, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(deep_dim, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(deep_dim, activation = 'relu')(x)
    x = Dropout(0.1)(x)



    """
    # TODO STACK A COUPLE MORE LSTM LAYERS, and use many of them in each layer (hundreds)
    # Add this before the encoded layer, so this layer gets info that the LSTMs have learned.

    # This would obviously require reshaping the output of the latest CNN because LSTMs only get 2d input (not counting the batch_size dimension)
    # So you would have 1 value at each position for the last filter, with the timesetps being the position along the genome

    # Use  LSTM layer with return sequences, to get (timesteps, lstm_number) as output dim
    # Then a time distributed Dense(deep_dim) to get (timesteps, deep_dim) as output
    # if use_lstm :
        # x = LSTM(latent_dim, return_sequences = True)(x)
        # x = LSTM(latent_dim, return_sequences = True)(x) # Stack 2 LSTMs for giggles.
        # NOTE as usual, having too many LSTMs may make the information budget too high. Play with budget and adjust consequenty.
        # x = TimeDistributed(Dense(deep_dim))(x) # THIS SHOULD BE THE ENCODED LAYER ACTUALLY, otherwise I'm just making a new layer of combis and losing the timestep
    # else :
    """




    # Layer of the encoded representation (no regularisation)
    x = Dense(deep_dim, activation = 'relu',
        name = 'encoded')(x)














    ## DECODER





    x = Lambda(lambda m: K.expand_dims(K.expand_dims(m,3),4),
        output_shape = lambda input_shape: input_shape + (1,1)
        )(x) # Reshape
    x = Conv3D(kernel_nb, kernel_size_decoder, padding='valid')(x)
    x = ZeroPadding3D(padding)(x)
    x = UpSampling3D((filter_pooling))(x)

    #x = Conv3D(kernel_nb, kernel_size_inner, padding='valid')(x)
    #x = ZeroPadding3D(padding)(x)
    x = UpSampling3D((filter_pooling))(x)
    #x = Conv3D(kernel_nb, kernel_size_inner, padding='valid')(x)
    #x = ZeroPadding3D(padding)(x)
    #x = UpSampling3D((filter_pooling))(x)
    # x = Conv3D(kernel_nb, kernel_size_inner, padding='valid')(x)
    # x = ZeroPadding3D(padding)(x)
    # x = UpSampling3D((filter_pooling))(x)


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


    # Reverse 'telomer' padding :
    final_layer = Lambda(lambda m: m[:,telomer_len:-telomer_len,:,:,:], name = 'unpad')(decoded)





    # Concatenate both encoder and decoder
    autoencoder = Model(input, final_layer)

    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder






































































# ---------------------------- Interpretability ------------------------------ #




"""
TODO : USE THIS TO VISUALIZE HIDDEN LAYERS AND/OR THE "ENCODED REPRESENTATION" !!!!!!!!
VERY IMPORTANT FOR DIAGNOSTIC !!!!

Can probably be used with LSTMs too according to https://medium.com/@plusepsilon/visualizations-of-recurrent-neural-networks-c18f07779d56


"""






















import sys

import numpy as np
import pandas as pd
# from keras import *
# TODO never do that kind of import ! relaunch it to make sure all functions were properly imported
from keras.models import *
from keras.layers import *
from skimage.filters import gaussian as gaussian_filter
import time
from keras import backend as K


def compute_max_activating_example_across_layer(model,
                            selected_layer,
                            output_of_layer_select_this_dim = 1,
                            learning_rate = 2,
                            random_state = 42,
                            nb_steps_gradient_ascent = 20,
                            blurStdX = 0, blurStdY=0, blurStdZ=0, blurEvery = 4,
                            ur_example_input_shape = None
                            ):
    """
    model : a Keras model
    selected_layer : the layer on which to perform gradient ascent
    learning_rate & nb_steps_gradient_ascent : parameters for gradient ascent
    blurStdX & blurStdY=0 & blurEvery = 4 : paramters to blur the results to make them more natural

    Return a list : for each class, gives an artificial example for which the
    selected layer best says "this example belong to class X"

    Derived from : http://ankivil.com/visualizing-deep-neural-networks-classes-and-features/
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

            # Control print
            sys.stdout.write("\r" + '-- Current loss value : ' + str(np.mean(loss_value))) ; sys.stdout.flush()

        # decode the resulting input image and add it to the list
        max_activation_class.append(input_data[0])
        losses.append(loss_value)
        end_time = time.time()
        print('\nOutput dimension %d processed in %ds' % (class_index, end_time - start_time))


    # Return the processed list of optimal class examples
    return max_activation_class






# To understand, provide a function that, given a before matrix (or simply a CRM) will output its encoded representation. So you can understand the combinations at play.
def get_encoded_representation(before_matrix, model, ENCODED_LAYER_NUMBER = 15, disable_learning = True):

    # Disable the learning phase behavior (e.g. Dropout)
    if disable_learning: learnflag = 0
    else: learnflag = 1

    get_kth_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[ENCODED_LAYER_NUMBER].output])
    result = get_kth_layer_output([before_matrix, learnflag])[0]

    return result
