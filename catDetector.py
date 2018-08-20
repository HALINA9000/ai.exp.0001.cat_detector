# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:02:16 2018

@author: tom.s (at) halina9000.com
"""
#%%
import numpy as np
import time

from catDetectorDataUtils import load_data
from catDetectorModelUtils import model_define, model_compile, model_fit
from catDetectorChartUtils import history_plot

#%%
""" Load train and test dataset. """

train_x, train_y, test_x, test_y = load_data()

#%%
""" Find most efficient batch size. """

# Model compile variables
lr=0.005
# Model fitting variables
path = ''
title = ''
epochs = 200

model = model_define(train_x.shape[1])
model = model_compile(model, lr=lr)

# Maximum batch size
batch_size_limit = int(np.log2(train_x.shape[0])) + 1
# Set of batch sizes
batch_size_set = [2**x for x in range(5, batch_size_limit + 1)]
# Measeure of execution time for each batch_size
batch_exe_time = []
for batch_size in batch_size_set:
    time_start = time.time()
    model, history = model_fit(model, train_x, train_y, test_x, test_y,
                               path, title,
                               epochs=epochs, batch_size=batch_size)
    time_end = time.time() - time_start
    batch_exe_time.append([time_end, batch_size]) # Add new score
batch_exe_time.sort() # Fastest batch size will be on top.
batch_size = batch_exe_time[0][1] # Extracting fastest batch size.
print("\nMost efficient batch size is:", batch_size)

#%%
""" Asssignment in Keras. """

# Model fitting variables
path = 'originalAssignment\\'
title = 'Original course assignment result'
epochs = 2000

model = model_define(train_x.shape[1])
model = model_compile(model, lr=lr)
_, history = model_fit(model, train_x, train_y, test_x, test_y,
                       path, title,
                       epochs=epochs, batch_size=batch_size)
history_set = [history]
history_plot(history_set, path, title, acc=True, val_acc=True)

#%%
""" Datasets discussion. """

# Training set
# Size of set
train_size = train_y.shape[0]
# Amount of images with cat
train_amnt = np.sum(train_y)
# Percent of cat images in training set
train_pcnt = np.sum(train_y)/len(train_y) * 100
print('Cat images in training set: %2d%% (%2d/%2d).' % (train_pcnt,
                                                        train_amnt,
                                                        train_size))
# Test set
# Size of set
test_size = test_y.shape[0]
# Amount of images with cat
test_amnt = np.sum(test_y)
# Percent of cat images in test set
test_pcnt = np.sum(test_y)/len(test_y) * 100
print('Cat images in testing set: %2d%% (%2d/%2d).' % (test_pcnt,
                                                       test_amnt,
                                                       test_size))

#%%
""" Hyperparameter tuning.
"""
# Model compilation variables
lr = 0.0001
# Model fitting variables
path = 'lrTuning\\'
title = 'Learning rate: ' + str(lr)
epochs = 100000

model = model_define(train_x.shape[1])
model = model_compile(model, lr=lr)
_, history = model_fit(model, train_x, train_y, test_x, test_y,
                       path, title,
                       epochs=epochs, batch_size=batch_size)
history_set = [history]
history_plot(history_set, path, title, acc=True, val_acc=True)