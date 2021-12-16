import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

RAW_DATA_PATH = os.path.join("../data", "JacksonFischer")
PREPROSSED_DATA_PATH = os.path.join("../data", "raw")
PLOT_PATH = os.path.join( "../plots")

def plot_cell_count_distribution():


    df_dataset = pd.read_csv(os.path.join(PREPROSSED_DATA_PATH, "basel_zurich_preprocessed_compact_dataset.csv"))
    number_of_cells = df_dataset.groupby(by=["ImageNumber"]).size().reset_index(name='Cell Counts')
    
    low_quartile = np.quantile(number_of_cells["Cell Counts"], 0.25)
    print(gene_thr)
    
    cell_count_dist_plot = sns.boxplot(data=number_of_cells, x='Cell Counts')
    fig = cell_count_dist_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/cell_count_distribution.png")

# plot_cell_count_distribution()