import os
import torch
import numpy as np
import custom_tools
import pickle
from torch_geometric import utils
import plotting as plotting
from tqdm import tqdm   
import matplotlib.pyplot as plt
import torch_geometric as pyg
from explainer import GNNExplainer
from explainer_base import GNNExplainer
import largest_connected_component
from data_processing import OUT_DATA_PATH
import networkx as nx
import pytorch_lightning as pl


device = custom_tools.get_device()
S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

    
class Explainer:
 
    # init method or constructor
    def __init__(self, model,dataset = None, seed=42):

        if seed!=42:
            pl.seed_everything(seed)
        self.seed = seed
        self.model = model
        self.dataset = dataset


    def set_dataset(self, dataset):
        self.dataset = dataset

    def explain_by_gnnexplainer(self,lr: float=0.1,  epoch: int=100, return_type: str = "regression", feat_mask_type: str = "feature"):
        """
        Learns and returns a node feature mask and an edge mask that play a crucial role to explain the prediction made by the GNN for a graph.
        test_graph.x (torch.Tensor): The node feature matrix
        test_graph.edge_index (torch.Tensor): The edge indices.
        
        Additional hyper-parameters to override default settings in coeffs.
        coeffs = {'edge_ent': 1.0, 'edge_reduction': 'sum', 'edge_size': 0.005, 'node_feat_ent': 0.1, 'node_feat_reduction': 'mean', 'node_feat_size': 1.0}
        """
        explainer = GNNExplainer(self.model, epochs = epoch, lr = lr,
                                    return_type = return_type, feat_mask_type = feat_mask_type).to(device)
        count = 0
        for test_graph in tqdm(self.dataset):
            print(f"Sample id: {test_graph.img_id}_{test_graph.p_id}")

            with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
                coordinates_arr = pickle.load(handle)
            
            # number of nodes
            # test_graph.num_nodes g.number_of_nodes() g.number_of_edges()

            (feature_val_mask, edge_value_mask) = explainer.explain_graph(test_graph.x.to(device), test_graph.edge_index.to(device))
            quant_thr = 0.80

            edge_exp_score_mask_arr = np.array(edge_value_mask.cpu())


            edge_thr = np.quantile(np.array(edge_value_mask.cpu()), quant_thr)

            # print(f"Edge thr: {edge_thr:.3f}\tMin: {np.min(edge_exp_score_mask_arr)}\tMax: {np.max(edge_exp_score_mask_arr):.3f}\tMin: {np.min(edge_exp_score_mask_arr):.3f}")
            

            exp_edges_bool = edge_exp_score_mask_arr > edge_thr


            
            explained_edge_indices = exp_edges_bool.nonzero()[0]

            """for ind, val in enumerate(exp_edges_bool):
                if val:
                    print(val, edge_exp_score_mask_arr[ind])"""
            # print(edge_exp_score_mask_arr)
            # print(list(set(range(len(edge_value_mask)))- set(explained_edge_indices)))
            np.put(edge_exp_score_mask_arr, list(set(range(len(edge_value_mask)))- set(explained_edge_indices)), 0.0)


            edgeid_to_mask_dict = dict()
            for ind, m_val in enumerate(edge_exp_score_mask_arr):
                # print(ind, m_val, exp_edges_bool[ind], edge_value_mask[ind], edge_thr)
                node_id1, node_id2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
                edgeid_to_mask_dict[(node_id1, node_id2)] = m_val.item()
            
            
            n_of_hops = 2   
            # TODO: Check if the scores are calculated over the ccs
        
            node_to_score_dict = custom_tools.get_all_k_hop_node_scores(test_graph, edgeid_to_mask_dict, n_of_hops)

            custom_tools.convert_graph_to_anndata(test_graph, node_to_score_dict)
            
            plt.rcParams['figure.figsize'] = 100, 100
            fig, axs = plt.subplots(2, 2)
            plotting.plot_graph(test_graph, coordinates_arr, axs[0][0])
            # plotting.plot_khop(test_graph, coordinates_arr, edgeid_to_mask_dict,  n_of_hops, axs[0][1])
            plotting.plot_node_importances_voronoi(test_graph, coordinates_arr, node_to_score_dict,  axs[0][1])
            plotting.plot_connected_components(test_graph, coordinates_arr, exp_edges_bool, axs[1][0])
            plotting.plot_node_importances(test_graph, coordinates_arr, node_to_score_dict,  axs[1][1])
            
            fig.savefig(f"../plots/subgraphs/{test_graph.img_id}_{test_graph.p_id}")
            plt.close()
            count +=1
            # if count ==10:
            #     break
        

            
            

    def explain_by_lime(self, epoch, return_type, feat_mask_type):
        ## TODO: Include lime implementation 
        return





 
 


# python gnnexplainer.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 1