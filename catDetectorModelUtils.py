# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:14:47 2018

@author: tom.s (at) halina9000.com
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, TFOptimizer
#from keras import regularizers
from time import time


def model_define(input_dim, initializer='zeros', activation='sigmoid'):
    """Model definition

    Arguments:
        input_dim (int): Input dimension.
        initializer (str): Keras initializer type.
        activation (str): Activation function.

    Returns:
        model: defined model
    """
    
    if initializer == 'random_uniform':
        initializer = RandomUniform(minval=-1.0, maxval=1.0, seed=None)
    model = Sequential()
    # single neuron model definition
    model.add(Dense(1,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    input_dim=input_dim,
                    activation=activation))
    return model


def model_compile(model, loss='binary_crossentropy', opt=['gradient', 0.005]):
    """ Compiles previously defined model

    Arguments:
        model: Model to compile.
        loss (str): Loss function.
        lr (float): Learning rate. If no lr provided - default Keras lr for
            Adam will be used.
        decay (float): Learning rate decay. If no decay provided - default
            Keras decay for Adam will be used.

    Returns:
        model: Compiled model.
    """

    if opt[0] == 'gradient':
        optimizer = TFOptimizer(tf.train.GradientDescentOptimizer(opt[1]))
    else:
        optimizer = Adam(lr=opt[1], decay=opt[2])
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def model_fit(model, train_x, train_y, test_x, test_y,
              path, title,
              epochs, batch_size,
              cb_tensorboard=[False],
              cb_stopper=[False],
              cb_checkpointer=[False],
              cb_reduce_lr=[False],
              verbose=0):
    """ Model fitting

    Arguments:
        model: model to fit
        train_x (np.array(float)): Train set features.
        train_y (np.array(int)): Train labels (0 or 1 i.e. cat or non-cat).
        test_x (np.array(float)): Test set features.
        test_y (np.array(int)): Test labels (0 or 1 i.e. cat or non-cat).
        path (str): Path where checkpointer h5 file should be stored.
        title (str): Common string in file names.
        epochs (int): Number of epochs.
        batch_size (int): Size of batch when fitting.
        cb_tensorboard (list): List of tensorboard parameters:
            First element determines if stopper has to be used.
            [True/False, path].
            Default: [False].
        cb_stopper (list): List of stopper callback parameters.
            First element determines if stopper has to be used.
            [True/False, patience].
            Default: [False].
        cb_checkpointer (list): List of checkpointer callback parameters.
            First element determines if checkpointer has to be used.
            [True/False, prefix, monitor, save_best_only, save_weights_only, 
            mode].
            Default: [False].
        cb_reduce_lr (list): List of reduce_lr callback parameters for
            ReduceLROnPlateau.
            First element determines if checkpointer has to be used.
            [True/False, factor, patience, min_lr].
            Default: [False].
        verbose: 0, 1 or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.

    Returns:
        model: Fitted model.
        history: History of model.
    """
    # define callbacks set
    
    callbacks = []
    if cb_tensorboard[0]:
        log_dir = cb_tensorboard[1] + '{}'
        tensorboard = TensorBoard(log_dir=log_dir.format(time()))
        callbacks.append(tensorboard)
    if cb_stopper[0]:
        stopper = EarlyStopping(monitor='acc', patience=cb_stopper[1])
        callbacks.append(stopper)
    if cb_checkpointer[0]:
        if cb_checkpointer[1] == 'av':
            metrics = '{acc:.2f}-{val_acc:.2f}-'
        elif cb_checkpointer[1] == 'va':
            metrics = '{val_acc:.2f}-{acc:.2f}-'
        else:
            metrics = ''
        title = title.replace(':', '')
        title = title.replace(' ', '-')
        file = path + metrics + title + '.h5'
        checkpointer = ModelCheckpoint(filepath=file,
                                       monitor=cb_checkpointer[2],
                                       save_best_only=cb_checkpointer[3],
                                       save_weights_only=cb_checkpointer[4],
                                       mode=cb_checkpointer[5])
        callbacks.append(checkpointer)
    if cb_reduce_lr[0]:
        reduce_lr = ReduceLROnPlateau(monitor='acc',
                                      factor=cb_reduce_lr[1],
                                      patience=cb_reduce_lr[2],
                                      min_lr=cb_reduce_lr[3])
        callbacks.append(reduce_lr)

    # model fitting
    history = model.fit(train_x, train_y,
                   epochs=epochs,
                   callbacks=callbacks,
                   validation_data=(test_x, test_y),
                   batch_size=batch_size,
                   verbose=verbose)
    return model, history