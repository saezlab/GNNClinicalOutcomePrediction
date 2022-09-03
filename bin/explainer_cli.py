import os
import torch
import numpy as np
import bin.custom_tools as custom_tools
import pickle
import torch_geometric as pyg
from explainer import GNNExplainer
from explainer_base import GNNExplainer
from data_processing import OUT_DATA_PATH

device = custom_tools.get_device()
S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

    
class Explainer:
 
    # init method or constructor
    def __init__(self, model, dataset = None):
        self.model = model
        self.dataset = dataset


    def set_dataset(self, dataset):
        self.dataset = dataset

    def explain_by_gnnexplainer(self,lr: float=0.1,  epoch: int=100, return_type: str = "regression", feat_mask_type: str = 'feature'):
        '''
        Learns and returns a node feature mask and an edge mask that play a crucial role to explain the prediction made by the GNN for a graph.
        test_graph.x (torch.Tensor): The node feature matrix
        test_graph.edge_index (torch.Tensor): The edge indices.

        
        Additional hyper-parameters to override default settings in coeffs.
        coeffs = {'edge_ent': 1.0, 'edge_reduction': 'sum', 'edge_size': 0.005, 'node_feat_ent': 0.1, 'node_feat_reduction': 'mean', 'node_feat_size': 1.0}
        '''
        explainer = GNNExplainer(self.model, epochs = epoch, lr = lr,
                                    return_type = return_type, feat_mask_type = feat_mask_type).to(device)
        
        for test_graph in self.dataset:
            coordinates_arr = None
            with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
                coordinates_arr = pickle.load(handle)
            result = explainer.explain_graph(test_graph.x.to(device), test_graph.edge_index.to(device))
            
            (feature_mask, edge_mask) = result
            edges_idx = edge_mask > 0.5

            explanation = pyg.data.Data(test_graph.x, test_graph.edge_index[:, edges_idx], pos= coordinates_arr)
            explanation = pyg.transforms.RemoveIsolatedNodes()(pyg.transforms.ToUndirected()(explanation))
        
            return explanation, edges_idx

    
    def explain_by_lime(self, epoch, return_type, feat_mask_type):
        ## TODO: Include lime implementation 
        return





 
 


# python gnnexplainer.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 10

