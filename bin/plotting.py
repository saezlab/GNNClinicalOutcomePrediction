import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import torch
from dataset import TissueDataset

RAW_DATA_PATH = os.path.join("../data", "JacksonFischer")
PREPROSSED_DATA_PATH = os.path.join("../data", "raw")
PLOT_PATH = os.path.join( "../plots")

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
