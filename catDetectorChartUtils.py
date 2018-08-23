# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:14:47 2018

@author: tom.s (at) halina9000.com
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def history_plot(history_set, path, title, 
                 y_min=0.0, y_max=1.0, 
                 acc=True, val_acc=False,
                 alpha_acc=0.9, alpha_val_acc=0.7,
                 show=False):
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
        alpha_acc (float): value of alpha for accuracy on training set plot. 
            Range 0.0 - 1.0.
        alpha_val_acc (float): value of alpha for accuracy on test set plot. 
            Range 0.0 - 1.0.
        show (boolean): True if saved chart should be shown. Default: False.
    """

    # plot 10 ticks on y axis    
    y_tick = (y_max - y_min) / 10.
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    
    ax.xaxis.grid(linestyle='dotted', color='#000000')
    ax.yaxis.grid(linestyle='dotted', color='#000000')
    plt.yticks(np.arange(y_min, y_max, step=y_tick))
    ax.set_ylim(y_min, y_max)
    
    plt.title('Model accuracy' + '\n' + title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    
    for _, history in history_set:
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
    title = title.replace('-', '')
    title = title.replace(':', '')                 
    title = title.replace(' ', '_')
    plt.savefig(path + title + '.png')
    plt.close()
    if show:
        img = Image.open(path + title + '.png')
        img.show()