import argparse
import os
from unittest import result

import matplotlib.pyplot as plt
#from GNNClinicalOutcomePrediction.bin.train_test_dgermen import test
import networkx as nx
import pickle5 as pickle
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from svx_explainers_base import GraphSVX
import numpy as np

from dataset import TissueDataset
from model_dgermen import CustomGCN


S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

parser = argparse.ArgumentParser(description='GNN Arguments')
parser.add_argument(
    '--bs',
    type=int,
    default=16,
    metavar='BS',
    help='batch size (default: 32)')

parser.add_argument(
    '--idx',
    type=int,
    default=42,
    metavar='IDX',
    help='Index Number of the graph (default: 14)')

parser.add_argument(
    '--exp_epoch',
    type=int,
    default=50,
    metavar='EXEPC',
    help='Number of epochs (default: 50)')

parser.add_argument(
    '--exp_lr',
    type=float,
    default=0.001,
    metavar='ELR',
    help='Explainer learning rate (default: 0.001)')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.001,
    metavar='WD',
    help='Explainer weight decay (default: 0.001)')

parser.add_argument(
    '--retun_type',
    type=str,
    default="regression",
    metavar='RT',
    help='return type of the explainer (default: regression)')

parser.add_argument(
    '--feat_mask_type',
    type=str,
    default="individual_feature",
    metavar='FMT',
    help='feature mask type of the explainer (default: individual_feature)')

parser.add_argument(
    '--relevant_edges',
    type=float,
    default=0.5,
    metavar='RE',
    help='Get the highest values inside the edge mask (relevant edges) (default: 0.001)')

parser.add_argument(
    '--model',
    type=str,
    default="PNAConv",
    metavar='mn',
    help='model name (default: PNAConv)')

parser.add_argument(
    '--num_of_gcn_layers',
    type=int,
    default=2,
    metavar='NGCNL',
    help='Number of GCN layers (default: 2)')

parser.add_argument(
    '--num_of_ff_layers',
    type=int,
    default=1,
    metavar='NFFL',
    help='Number of FF layers (default: 2)')

parser.add_argument(
    '--gcn_h',
    type=int,
    default=64,
    metavar='GCNH',
    help='GCN hidden channel (default: 128)')

parser.add_argument(
    '--fcl',
    type=int,
    default=256,
    metavar='FCL',
    help='Number of neurons in fully-connected layer (default: 128)')

parser.add_argument(
    '--dropout',
    type=float,
    default=0.00, 
    metavar='DO',
    help='dropout rate (default: 0.20)')

parser.add_argument(
    '--aggregators',
    nargs='+',
    type = str,
    default= ["max"], 
    metavar='AGR',
    help= "aggregator list for PNAConv")

parser.add_argument(
    '--scalers',
    nargs='+',
    type= str,
    default= ["identity"],
    metavar='SCL',
    help='Set of scaling function identifiers,')

parser.add_argument(
    '--factor', # 0.5 0.8, 0.2
    type=float,
    default=0.5,
    metavar='FACTOR',
    help='learning rate reduce factor (default: 0.5)')

parser.add_argument(
    '--patience', # 5, 10, 20
    type=int,
    default=5,
    metavar='PA',
    help='patience for learning rate scheduling (default: 5)')

parser.add_argument(
    '--min_lr',
    type=float,
    default=0.00002,#0.0001
    metavar='MLR',
    help='minimum learning rate (default: 0.00002)')

parser.add_argument(
    '--edge_size',
    type=float,
    default=0.005,#0.005
    metavar='ES',
    help='Edge Size (default: 0.005)')

parser.add_argument(
    '--node_feat_size',
    type=float,
    default=1.0,
    metavar='NFS',
    help='Node Feature size (default: 1.0)')

parser.add_argument(
    '--edge_ent',
    type=float,
    default=1.0,
    metavar='ENT',
    help='Edge Entropy (default: 1.0)')

parser.add_argument(
    '--node_feat_ent',
    type=float,
    default=0.1,
    metavar='NFE',
    help='Node Feature Entropy (default: 0.1)')

parser.add_argument(
    '--gpu', 
    default=True, 
    help='whether to use GPU.')

parser.add_argument("--indexes", 
                    type=list, 
                    default=[10],
                    help="indexes of the nodes/graphs whose prediction are explained")

parser.add_argument("--hops", 
                    type=int,
                    default = 2,
                    help="number k for k-hops neighbours considered in an explanation")

parser.add_argument("--num_samples", 
                    type=int,
                    default = 40,
                    help="number of coalitions sampled and used to approx shapley values")

parser.add_argument("--info", 
                    type=bool,
                    default = True,
                    help='True if want to print info')

parser.add_argument("--multiclass", 
                    type=bool,
                    default = False,
                    help='False if we consider explanations for the predicted class only')
                    
parser.add_argument("--fullempty", 
                    type=str,
                    default = None,
                    help='True if want to discard full and empty coalitions')

parser.add_argument("--S", 
                    type=int,
                    default = 3,
                    help='Max size of coalitions sampled in priority and treated specifically')

parser.add_argument("--feat", 
                    type=str,
                    default = 'Expectation',
                    help="method used to determine the features considered")

parser.add_argument("--coal", 
                    type=str,
                    default = 'Smarter',
                    help="type of coalition sampler")

parser.add_argument("--g", 
                    type=str,
                    default = 'WLS',
                    help="surrogate model used to train g on derived dataset")

parser.add_argument("--regu", 
                    type=float,
                    default = None,
                    help='None if we do not apply regularisation, \
                        1 if we focus only on features in explanations, 0 for nodes')


S_PATH = os.path.dirname(__file__)
pl.seed_everything(42)
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

device = "cpu"

if use_gpu:
    device = "cuda"
else:
    print("CPU is available on this device!")


dataset = TissueDataset(os.path.join(S_PATH,"../data"))

dataset = dataset.shuffle()

num_of_train = int(len(dataset)*0.80)
num_of_val = int(len(dataset)*0.10)

train_dataset = dataset[:num_of_train]
validation_dataset = dataset[num_of_train:num_of_train+num_of_val]
test_dataset = dataset[num_of_train+num_of_val:]

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

idx = args.idx
test_graph = test_dataset[idx]


coordinates_arr = None
with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
        coordinates_arr = pickle.load(handle)

deg = 1

if args.model == "PNAConv":
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

def load_checkpoint(filepath):
    
    model = CustomGCN(type = args.model,
                num_node_features = dataset.num_node_features,
                num_gcn_layers=args.num_of_gcn_layers, 
                num_ff_layers=args.num_of_ff_layers, 
                gcn_hidden_neurons=args.gcn_h, 
                ff_hidden_neurons=args.fcl, 
                dropout=args.dropout,
                aggregators=args.aggregators,
                scalers=args.scalers,
                deg = deg).to(device)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_checkpoint('../data/models/model.pth')


def plot_original_graph():

    pos_1 = coordinates_arr
    g = utils.to_networkx(test_graph, to_undirected=True)
    nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
    nx.draw_networkx_edges(g,edge_color='b', pos=pos_1)

    return plt.savefig(f'../plots/original_graphs/original_graph_{args.idx}_{args.exp_epoch}_{args.exp_lr}_{args.retun_type}_{args.feat_mask_type}.png', dpi=1000)

print('test_graph:', test_graph)


def GraphSVXEx():

    # Define graphs used for evaluation
    for i in test_loader:
        allgraphs = np.nonzero(i.y).T[0].tolist()[:10]
        print('i:',i)

        for graph_idx in allgraphs:
        
            explainer = GraphSVX(i, model, args.gpu)

            print('self.data.edge_index.shape[1]:', i.edge_index.shape[1])
            print('self.data.edge_index[graph_indices]:', i.edge_index[graph_idx].shape)
            print('i.edge_index:', i.edge_index.shape)
            print('np.diag(self.data.edge_index[graph_indices]):', np.diag(i.edge_index[graph_idx]))
            print('np.sum(np.diag(self.data.edge_index[graph_indices])):', np.sum(np.diag(i.edge_index[graph_idx])).shape)
            print('(self.data.edge_index.shape[1] - np.sum(np.diag(self.data.edge_index[graph_indices]))):', (i.edge_index.shape[1] - np.sum(np.diag(i.edge_index[graph_idx]))).shape)
            print('range(int(self.data.edge_index.shape[1] - np.sum(np.diag(self.data.edge_index[graph_indices])))):',range(int(i.edge_index.shape[1] - np.sum(np.diag(i.edge_index[graph_idx])))))
            print('list(range(int(self.data.edge_index.shape[1] - np.sum(np.diag(self.data.edge_index[graph_indices]))))):', list(range(int(i.edge_index.shape[1] - np.sum(np.diag(i.edge_index[graph_idx]))))))    

            result = explainer.explain_graphs([graph_idx],
                                            args.hops,
                                            args.num_samples,
                                            args.info,
                                            args.multiclass,
                                            args.fullempty,
                                            args.S,
                                            'graph_classification',
                                            args.feat,
                                            args.coal,
                                            args.g,
                                            args.regu,
                                            True)[0]
    
    return result

results = GraphSVXEx()

'''
def plot_subgraph():

    options = ['r','b','y','g']
    colors_edge = []
    

    g = utils.to_networkx(test_graph, to_undirected=True)
    

    colors_node = ['b']*len(g.nodes)

    for id,e_idx in enumerate(g.edges):
        #for g_exp in gexp_edges:
            if edges_idx[id]:
                colors_edge.append(options[0])
                n1, n2 = e_idx
                colors_node[n1]=options[0]
                colors_node[n2]=options[0]
            else:
                colors_edge.append(options[1])   
                

    pos_1 = coordinates_arr

    nx.draw_networkx_nodes(g, node_color=colors_node, pos=pos_1, node_size=1)
    nx.draw_networkx_edges(g, edge_color=colors_edge, pos=pos_1)

    return plt.savefig(f'../plots/subgraphs/subgraph_{args.idx}_{args.exp_epoch}_{args.exp_lr}_{args.weight_decay}_{args.factor}_{args.patience}_{args.min_lr}_{args.edge_size}_{args.node_feat_size}_{args.edge_ent}_{args.node_feat_ent}_{args.retun_type}_{args.feat_mask_type}_{args.relevant_edges}.png', dpi=1000)


plot_original_graph()
plot_subgraph()

'''