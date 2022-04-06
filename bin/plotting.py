import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import torch
from dataset import TissueDataset

S_PATH = os.path.dirname(__file__)
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "JacksonFischer")
PREPROSSED_DATA_PATH = os.path.join(S_PATH, "../data", "raw")
PLOT_PATH = os.path.join(S_PATH, "../plots")

def plot_cell_count_distribution():


    df_dataset = pd.read_csv(os.path.join(PREPROSSED_DATA_PATH, "basel_zurich_preprocessed_compact_dataset.csv"))
    number_of_cells = df_dataset.groupby(by=["ImageNumber"]).size().reset_index(name='Cell Counts')
    
    low_quartile = np.quantile(number_of_cells["Cell Counts"], 0.25)
    
    cell_count_dist_plot = sns.boxplot(data=number_of_cells, x='Cell Counts')
    fig = cell_count_dist_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/cell_count_distribution.png")

# plot_cell_count_distribution()

def box_plot_save(df, axis_name, path):
    osmonth_dist_plot = sns.boxplot(data=df, x=axis_name)
    fig = osmonth_dist_plot.get_figure()
    fig.savefig(path)
    plt.clf()


def plot_nom_distribution():
    dataset = TissueDataset("../data")
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    osmonhth_list = []
    for i in range(len(dataset)):
        osmonhth_list.append(dataset.__getitem__(i).y[0].item())
    
        #osmonhth_list.append(dataset.__getitem__(i))
        # print(i)
    df_osmonth = pd.DataFrame(osmonhth_list)
    
    box_plot_save(df_osmonth, 0, f"{PLOT_PATH}/osmonth_all_distribution.png")
    box_plot_save(df_osmonth[:180], 0, f"{PLOT_PATH}/osmonth_all_distribution.png")
    box_plot_save(df_osmonth[180:200], 0, f"{PLOT_PATH}/osmonth_all_distribution.png")
    box_plot_save(df_osmonth[200:], 0, f"{PLOT_PATH}/osmonth_all_distribution.png")

# plot_nom_distribution()

def plot_pred_vs_real(df, x_axis, y_axis, color, fl_name):
    # print(out_list, pred_list)
    sns_plot = sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=color)
    plt.ylim(0, None)
    plt.xlim(0, None)
    fig = sns_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/{fl_name}.png")
    plt.clf()


def plot_pred_vs_real_lst(df_lst, x_axis_lst, y_axis_lst, color, order, fl_name):
    # print(out_list, pred_list)
    n_cols = len(df_lst)
    fig, axs = plt.subplots(ncols=n_cols, figsize=(18,5))

    for ind in range(n_cols):

        sns_plot = sns.scatterplot(data=df_lst[ind], x=x_axis_lst[ind], y=y_axis_lst[ind], hue=color, hue_order=order, ax = axs[ind])
        sns_plot.set_xlim(0, 6)
        sns_plot.set_ylim(0, 6)
        plt.legend(loc='upper left')    
        
        #plt.ylim(0, None)
        #plt.xlim(0, None)
    
    fig = sns_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/{fl_name}.png")
    plt.clf()

import matplotlib.lines as mlines

def plot_pred_(df, color, fl_name):
    # print(out_list, pred_list)
    fig, axs = plt.subplots(1,3,figsize=(20,5))
    colors = {'TripleNeg':'red', 'HR+HER2-':'green', 'HR-HER2+':'blue', 'HR+HER2+':'orange'}
    labels = ['TripleNeg', 'HR+HER2-', 'HR-HER2+', 'HR+HER2+']
    colors = ['red', 'green', 'blue', 'orange']

    
    for idx, val in enumerate(["train", "validation", "test"]):

        
        for idxlbl, lbl in enumerate(labels):
            df_temp = df.loc[(df['Train Val Test'] == val) & (df['Clinical Type'] == lbl)]
            axs[idx].scatter(x=df_temp['OS Month (log)'], y=df_temp['Predicted'], color= colors[idxlbl], label=lbl)
            axs[idx].set_xlim(0, 6)
            axs[idx].set_ylim(0, 6)
            axs[idx].set_xlabel('OS Month (log)')
            axs[idx].set_ylabel('Predicted')
    
        
    plt.legend(loc='lower right')

    # line = mlines.Line2D([0, 1], [0, 1], color='red')

    # plt.scatter(df['OS Month (log)'], df['Predicted'], c=df['Clinical Type'].map(colors))
    
    plt.savefig(f"{PLOT_PATH}/{fl_name}.png")
    """for ind in range(n_cols):

        sns_plot = sns.scatterplot(data=df_lst[ind], x=x_axis_lst[ind], y=y_axis_lst[ind], hue=color, hue_order=order, ax = axs[ind])
        sns_plot.set_xlim(0, 6)
        sns_plot.set_ylim(0, 6)
        plt.legend(loc='upper left')    
        
        #plt.ylim(0, None)
        #plt.xlim(0, None)
    
    fig = sns_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/{fl_name}.png")
    plt.clf()"""