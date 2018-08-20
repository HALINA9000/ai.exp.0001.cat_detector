# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:14:47 2018

@author: tom.s (at) halina9000.com
"""
import matplotlib.pyplot as plt
import numpy as np


def history_plot(history_set, path, title, 
                 y_min=0.0, y_max=1.0, 
                 acc=True, val_acc=False,
                 alpha_acc=0.9, alpha_val_acc=0.7):
    """Summarizes history of model(s)
    
    Generates history of model(s) accuracy as a plot and saves it to file.
    
    Arguments:
        history_set: Set of model history.
        path (str): Path to store chart.
        title (str): Chart title.
        y_min (float): Lower limit of y axis. Default: 0.0.
        y_max (float): Upper limit of y axis. Default: 1.0.
        acc (boolean): Defines if accuracy on training set should be present
            on chart. Default: True.
        val_acc (boolean): Defines if accuracy on test set should be present
            on chart. Default: False.
    """

    # plot 10 ticks on y axis    
    y_tick = (y_max - y_min) / 10.
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    
    ax.xaxis.grid(linestyle='dotted', color='#000000')
    ax.yaxis.grid(linestyle='dotted', color='#000000')
    plt.yticks(np.arange(y_min, y_max, step=y_tick))
    ax.set_ylim(y_min, y_max)
    
    plt.title('Model accuracy' + '\n' + title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    
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
    title = title.replace(':', '_')                 
    title = title.replace(' ', '_')
    plt.savefig(path + title + '.png')
    plt.close()