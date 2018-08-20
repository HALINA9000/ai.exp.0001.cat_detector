# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:20:07 2018

@author: tom.s (at) halina9000.com
"""
import numpy as np
import h5py


def load_data():
    """ Load, reshape and normalize train and test data

    Returns:
        train_x (np.array((float)): Training x set (training features).
        train_y (np.array(int)): Training y set.
        test_x (np.arrayx(float)): Test x set (test features).
        test_y np.array(int)): Test y set.
    """
    # loading training dataset (cat/non-cat)
    train_dataset = h5py.File("datasets\\train_catvnoncat.h5", "r")
    train_x = np.array(train_dataset["train_set_x"][:]) # train set features
    train_y = np.array(train_dataset["train_set_y"][:]) # train set labels
    # loading test dataset (cat/non-cat)
    test_dataset = h5py.File("datasets\\test_catvnoncat.h5", "r")
    test_x = np.array(test_dataset["test_set_x"][:]) # test set features
    test_y = np.array(test_dataset["test_set_y"][:]) # test set labels

    # reshape training and test examples from (m, width, height, channels)
    # to (m, width * height * channels) form where m is number of samples
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)

    # normalization
    train_x = train_x / 255.
    test_x = test_x / 255.
    return train_x, train_y, test_x, test_y