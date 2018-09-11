# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:20:07 2018

@author: tom.s (at) halina9000.com
"""
import numpy as np
import h5py
import os


def load_data(data_path='datasets', reshape=True, normalize=True):
    """
    Load, reshape and normalize train and test data.

    Parameters
    ----------
        data_path : str, optional
            Path where dataset files are located.
        reshape : bool, optional
            If True data will be reshaped to form
            (m, height * width * number of channels).
        normalize : bool, optional
            If True data will be normalized in range (0, 1).

    Returns
    -------
        train_x : np.array(float)
            Training `x` set (training features).
        train_y : np.array(int)
            Training `y` set (training labels).
        test_x : np.array(float)
            Test `x` set (test features).
        test_y : np.array(int)
            Test `y` set (test labels).

    """
    train_datafile = os.path.join(data_path, 'train_catvnoncat.h5')
    test_datafile = os.path.join(data_path, 'test_catvnoncat.h5')
    train_dataset = h5py.File(train_datafile, 'r')
    test_dataset = h5py.File(test_datafile, 'r')

    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    train_dataset.close()
    test_dataset.close()

    if reshape:
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)

    if normalize:
        train_x = train_x / 255.
        test_x = test_x / 255.
    return train_x, train_y, test_x, test_y


def show_data_stats(set_y):
    """
    Show basic dataset statistics.

    Total amount of images, amount of images with cat and percent
    of cat images in dataset.

    Parameters
    ----------
        set_y : np.array(int)
            Training or test `y` set (labels).

    Returns
    -------
        set_size : int
            Total amount of images in dataset.
        set_amount : int
            Amount of cat images in dataset.
        set_cat_percent : int
            Percent of cat images in dataset.

    """
    set_size = set_y.shape[0]
    cat_amount = np.sum(set_y)
    cat_percent = np.int(cat_amount / set_size * 100)
    return cat_percent, cat_amount, set_size
