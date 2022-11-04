import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import torch
from torch_geometric import utils
import networkx as nx
from dataset import TissueDataset
from scipy.spatial import Voronoi, voronoi_plot_2d
from evaluation_metrics import r_squared_score, mse, rmse


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

"""def plot_pred_vs_real(df, x_axis, y_axis, color, fl_name):
    # print(out_list, pred_list)
    sns_plot = sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=color)
    plt.ylim(0, None)
    plt.xlim(0, None)
    fig = sns_plot.get_figure()
    fig.savefig(f"{PLOT_PATH}/{fl_name}.png")
    plt.clf()"""


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

def plot_pred_vs_real(df, exp_name, fl_name):

    fig, axs = plt.subplots(1,3,figsize=(20,5))
    colors = {'TripleNeg':'red', 'HR+HER2-':'green', 'HR-HER2+':'blue', 'HR+HER2+':'orange'}
    labels = ['TripleNeg', 'HR+HER2-', 'HR-HER2+', 'HR+HER2+']
    colors = ['red', 'green', 'blue', 'orange']

    for idx, val in enumerate(["train", "validation", "test"]):

        df_tvt = df.loc[(df['Fold#-Set'].str[2:] == val)]
        for idxlbl, lbl in enumerate(labels):
            # df_temp = df.loc[(df['Train Val Test'] == val) & (df['Clinical Type'] == lbl)]
            df_temp = df_tvt.loc[(df_tvt['Clinical Type'] == lbl)]
            axs[idx].scatter(x=df_temp['OS Month (log)'], y=df_temp['Predicted'], color= colors[idxlbl], label=lbl)
            axs[idx].set_xlim(0, 6)
            axs[idx].set_ylim(0, 6)
            axs[idx].set_xlabel('OS Month (log)')
            axs[idx].set_ylabel('Predicted')
            
        r2_score = r_squared_score(df_tvt['OS Month (log)'], df_tvt['Predicted'])
        mser = mse(df_tvt['OS Month (log)'], df_tvt['Predicted'])
        axs[idx].set_title(f"MSE: {mser:.3f}   R2 Score: {r2_score:.3f}")
        axs[idx].legend(loc='lower right')

    plt.savefig(os.path.join(PLOT_PATH, exp_name, f"{fl_name}.png"))


def plot_graph(test_graph, path, file_name, coordinates_arr):
    original_graph = utils.to_networkx(test_graph)
    pos_1 = coordinates_arr
    nx.draw_networkx_nodes(original_graph,  pos=pos_1, node_size=10)
    nx.draw_networkx_labels(original_graph, pos=pos_1, font_size=4)
    nx.draw_networkx_edges(original_graph,  pos=pos_1, arrows=False)
    plt.savefig(os.path.join(path, file_name+"_original"), dpi = 300)

def plot_subgraph(test_graph, path, file_name, coordinates_arr, edges_idx, cc_threshold = 5):

    options = ['r','b','y','g']
    colors_edge = []
    pos_1 = coordinates_arr
    
    original_graph = utils.to_networkx(test_graph)

    plt.clf()
    explained_graph = nx.Graph()
    explained_edges = []
    explained_graph.add_nodes_from(original_graph.nodes)
    for ind, val in enumerate(edges_idx):
        if val.item():
            explained_edges.append((test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()))
    
    print(f"Number of explained edges: {len(explained_edges)}")

    explained_graph.add_edges_from(list(explained_edges))
    nx.draw_networkx_nodes(explained_graph,  pos=pos_1, node_size=10)
    nx.draw_networkx_edges(explained_graph,  pos=pos_1)
    #print(explained_graph.edges)
    plt.savefig(os.path.join(path, file_name+"onlynodes"), dpi = 300)
    plt.clf()

    for component in list(nx.connected_components(explained_graph)):
        if len(component)<=cc_threshold:
            for node in component:
                explained_graph.remove_node(node)
    nx.draw_networkx_nodes(explained_graph,  pos=pos_1, node_size=10)
    nx.draw_networkx_edges(explained_graph,  pos=pos_1)
    plt.savefig(os.path.join(path, file_name+"removedsmallcc"), dpi = 300)
    plt.clf()
    
    colors_node = ['b']*len(original_graph.nodes)
    for ind, val in enumerate(edges_idx):
        #for g_exp in gexp_edges:
        if val.item():
            colors_edge.append(options[0])
            n1, n2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
            colors_node[n1]=options[0]
            colors_node[n2]=options[0]
        else:
            colors_edge.append(options[1])   

    nx.draw_networkx_nodes(original_graph, node_color=colors_node, pos=pos_1, node_size=10)
    nx.draw_networkx_edges(original_graph, edge_color=colors_edge, pos=pos_1,  arrows=False)
    plt.savefig(os.path.join(path, file_name), dpi = 300)
    plt.clf()


def plot_khop(test_graph, path, file_name, coordinates_arr, edgeid_to_mask_dict):
    # The method returns (1) the nodes involved in the subgraph, (2) the filtered edge_index connectivity, (3) the mapping from node indices in node_idx to their new location, and (4) the edge mask indicating which edges were preserved.
    subset_nodes, subset_edge_index, mapping, edge_mask = utils.k_hop_subgraph(0, 2, test_graph.edge_index)
    original_graph = utils.to_networkx(test_graph)
    # original_edges = list(original_graph.edges)
    # print("khop subset_edge_index:", subset_edge_index)
    pos_1 = coordinates_arr
    explained_graph = nx.Graph()
    explained_edges = []
    explained_graph.add_nodes_from(original_graph.nodes)
    
    total_score=0.0
    for ind, val in enumerate(edge_mask):
        if val.item():
            # print(original_edges[ind])
            n1, n2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
            explained_edges.append((n1,n2))
            total_score += edgeid_to_mask_dict[(n1,n2)]
            # print((n1,n2), edgeid_to_mask_dict[(n1,n2)])

    print(f"Total score: {total_score/len(explained_edges)}")

    explained_graph.add_edges_from(list(explained_edges))
    nx.draw_networkx_nodes(explained_graph,  pos=pos_1, node_size=10)
    nx.draw_networkx_labels(explained_graph, pos=pos_1, font_size=4)
    nx.draw_networkx_edges(explained_graph,  pos=pos_1)
    plt.savefig(os.path.join(path, file_name+"khop"), dpi = 300)
    plt.clf()
    
    # print(test_graph.edge_index.shape)
    




def plot_voronoi(test_graph, path, file_name, coordinates_arr, edges_idx, cc_threshold = 5):

    options = ['r','b','y','g']
    colors_edge = []
    pos_1 = coordinates_arr
    
    original_graph = utils.to_networkx(test_graph)
    original_edges = list(original_graph.edges)
    
    explained_graph = nx.Graph()
    explained_edges = []
    explained_graph.add_nodes_from(original_graph.nodes)
    for ind, val in enumerate(edges_idx):
        if val.item():
            explained_edges.append(original_edges[ind])

    vor = Voronoi(coordinates_arr)
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=2)
    plt.savefig(os.path.join(path, file_name+"voronoi"), dpi = 100)
    plt.clf()
    
    explained_graph.add_edges_from(list(explained_edges))
    nx.draw_networkx_nodes(explained_graph,  pos=pos_1, node_size=10)
    nx.draw_networkx_edges(explained_graph,  pos=pos_1)
    plt.savefig(os.path.join(path, file_name+"onlynodes"), dpi = 100)
    plt.clf()

    for component in list(nx.connected_components(explained_graph)):
        if len(component)<=cc_threshold:
            for node in component:
                explained_graph.remove_node(node)
    nx.draw_networkx_nodes(explained_graph,  pos=pos_1, node_size=10)
    nx.draw_networkx_edges(explained_graph,  pos=pos_1)
    plt.savefig(os.path.join(path, file_name+"removedsmallcc"), dpi = 100)
    plt.clf()
    


    colors_node = ['b']*len(original_graph.nodes)

    for id,e_idx in enumerate(original_graph.edges):
        #for g_exp in gexp_edges:
            if edges_idx[id]:
                colors_edge.append(options[0])
                n1, n2 = e_idx
                colors_node[n1]=options[0]
                colors_node[n2]=options[0]
            else:
                colors_edge.append(options[1])   

    nx.draw_networkx_nodes(original_graph, node_color=colors_node, pos=pos_1, node_size=10)
    nx.draw_networkx_edges(original_graph, edge_color=colors_edge, pos=pos_1)
    # graphs = list(nx.connected_component(g))
    # print(graphs)
    # graphs_m = max(nx.connected_component(g), key=len)
    # print("largest,", graphs_m)
    plt.savefig(os.path.join(path, file_name), dpi = 100)
    plt.clf()
    

def plot_node_importances(test_graph, path, file_name, coordinates_arr, node_score_dict):
    original_graph = utils.to_networkx(test_graph)

    color_list = []
    for n_id in original_graph.nodes:
        color_list.append(node_score_dict[n_id])

    nx.draw(original_graph, pos=coordinates_arr, node_color=color_list, node_size=80, cmap=plt.cm.afmhot)
    plt.savefig(os.path.join(path, file_name), dpi = 300)
    plt.clf()
