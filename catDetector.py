# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:02:16 2018

@author: tom.s (at) halina9000.com
"""
#%%
"""Load libraries."""
import os
from itertools import combinations
import numpy as np

from data import load_data, show_data_stats
from model import best_batch_size, course_assignment, compare_kernels
from model import best_learning_rate, sampling_hypersurface
from model import best_model_evaluate
from presentation import history_plot

#%%
"""Load datasets"""
train_x, train_y, test_x, test_y = load_data()

#%%
"""Show basic statistics for both training and test datasets."""
cat_percent, cat_amount, set_size = show_data_stats(train_y)
stat_output = 'Cat images in training set: %2d%% (%2d/%2d).'
print(stat_output % (cat_percent, cat_amount, set_size))

cat_percent, cat_amount, set_size = show_data_stats(test_y)
stat_output = 'Cat images in test set: %2d%% (%2d/%2d).'
print(stat_output % (cat_percent, cat_amount, set_size))

#%%
"""Determine best batch size."""
batch_size, batches_exe_time = best_batch_size(train_x, train_y)

print('%5s %8s' % ('size:', 'time:'))
print(5 * '-', 8 * '-')
for exe_time, size in batches_exe_time:
    print('%5d %8.4f' % (size, exe_time))
print('Most efficient batch size is:', batch_size)

#%%
"""Recreate in Keras original course assignment with its results."""
file_output_path = 'originalAssignment'
history = course_assignment(train_x, train_y,
                            test_x, test_y,
                            file_output_path,
                            batch_size=batch_size)

chart_title = 'Original course assignment results'
history_plot([history], file_output_path, chart_title)

#%%
"""Course assignment with random initialization."""
history = course_assignment(train_x, train_y,
                            test_x, test_y,
                            file_output_path,
                            initializer='random_uniform',
                            batch_size=batch_size)

chart_title = 'Modified course assignment results'
history_plot([history], file_output_path, chart_title)

#%%
"""Find best weights for zero and random initialization saved by BestAccs."""
suffix = 'zeros.h5'
weight_zero = [f for f in os.listdir(file_output_path) if f[18:] == suffix]
best_weight_zero = weight_zero[-1]
print('Best weight with zero initialization in', best_weight_zero, 'file.')
suffix = 'random_uniform.h5'
weight_random = [f for f in os.listdir(file_output_path) if f[18:] == suffix]
best_weight_random = weight_random[-1]
print('Best weight with random initialization in', best_weight_random, 'file.')

#%%
"""Quick review of the result: zero vs. random initialization."""
norms, angle = compare_kernels([best_weight_zero, best_weight_random],
                               file_output_path)

print('Norm of kernel with zeros initialization:  %.4f' % norms[0])
print('Norm of kernel with random initialization: %.4f' % norms[1])
print('Norm of vector difference between them:    %.4f' % norms[2])
print('Angle between kernels (rad):               %.4f' % angle)

#%%
"""Finding learning rate that gives most unstable charts."""
lr_set = [0.1, 0.01, 0.001, 0.0001]
for lr in lr_set:
    history_set = best_learning_rate(train_x, train_y,
                                     test_x, test_y,
                                     lr=lr,
                                     batch_size=batch_size)

    file_output_path = 'learningRateTuning'
    chart_title = 'Learning rate: ' + str(lr)
    history_plot(history_set, file_output_path, chart_title)

#%%
"""Sampling hypersurface with random initialization (uniform)."""
file_output_path = 'samplingHypersurface'
sampling_hypersurface(train_x, train_y, test_x, test_y,
                      file_output_path,
                      batch_size=batch_size,
                      iterations=1000)

#%%
"""File with best weights."""
files = [f for f in os.listdir(file_output_path)]
files.sort(reverse=True)
print(files[0])

#%%
"""Final results of best file."""
acc_train, acc_test = best_model_evaluate(train_x, train_y, test_x, test_y,
                                          file_output_path, files[0],
                                          batch_size=batch_size)
print('Accuracy on training set: %.3f' % acc_train)
print('Accuracy on test set:     %.3f' % acc_test)

#%%
"""Analysis of best weights"""
# Files with accuracy greater equal to 90%
prefixes = ['0.9', '1.0']
files_90 = [f for f in files if f[:3] in prefixes]

# Iteration has to be unique
iterations = []
files_90_unique = []
for file in files_90:
    if file[28:] not in iterations:
        iterations.append(file[28:])
        files_90_unique.append(file)

angles = []
norms = []
diffs = []
for file_1, file_2 in combinations(files_90_unique, 2):
    [norm_1, norm_2, diff], angle = compare_kernels([file_1, file_2],
                                                     file_output_path)
    angles.append(angle)
    norms.append([norm_1, norm_2])
    diffs.append(diff)

print('Angles between vectors')
print('Minimum: %.4f' % np.min(angles))
print('Maximum: %.4f' % np.max(angles), end='\n\n')

print('Norm of vectors')
print('Minimum: %.4f' % np.min(norms))
print('Maximum: %.4f' % np.max(norms), end='\n\n')

print('Norm of difference between vectors')
print('Minimum: %.4f' % np.min(diffs))
print('Maximum: %.4f' % np.max(diffs))

