# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:14:47 2018

@author: tom.s (at) halina9000.com
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def history_plot(history_set,
                 file_output_path, chart_title,
                 y_axis_min=0.0, y_axis_max=1.0,
                 acc=True, val_acc=True,
                 alpha_acc=0.7, alpha_val_acc=0.7):
    """
    Summarize history of model(s).

    Generates history of model(s) accuracy as a plot and saves it to file.

    Parameters
    ----------
        history_set : list
            Set of model history.
        file_output_path : str
            Path to store chart.
        chart_title : str
            Title of chart. Also used as filename.
        y_axis_min : float, optional
            Lower limit of y axis.
        y_axis_max : float, optional
            Upper limit of y axis.
        acc : bool, optional
            Defines if accuracy on training set should be present
            on chart.
        val_acc : bool, optional
            Defines if accuracy on test set should be present on chart.
        alpha_acc : float, optional
            Value of alpha for accuracy on training set plot.
            Range 0.0 - 1.0.
        alpha_val_acc : float
            Value of alpha for accuracy on test set plot.
            Range 0.0 - 1.0.

    """
    if not os.path.exists(file_output_path):
        os.makedirs(file_output_path)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    ax.xaxis.grid(linestyle='dotted', color='#000000')
    ax.yaxis.grid(linestyle='dotted', color='#000000')
    y_axis_tick = (y_axis_max - y_axis_min) / 10.
    plt.yticks(np.arange(y_axis_min, y_axis_max, step=y_axis_tick))
    ax.set_ylim(y_axis_min, y_axis_max)

    plt.title('Model accuracy' + '\n' + chart_title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    # Plot accuracy chart for training (blue line) and test (orange line)
    for history in history_set:
        if acc:
            plt.plot(history.history['acc'],
                     c='#1976D2',
                     linewidth=1,
                     alpha=alpha_acc)
        if val_acc:
            plt.plot(history.history['val_acc'],
                     c='#FF9800',
                     linewidth=1,
                     alpha=alpha_val_acc)

    # Saving chart as file
    chart_title = chart_title.replace('-', '')
    chart_title = chart_title.replace(':', '')
    chart_title = chart_title.replace(' ', '_')
    path_and_filename = os.path.join(file_output_path, chart_title + '.svg')
    plt.savefig(path_and_filename)
    plt.close()
