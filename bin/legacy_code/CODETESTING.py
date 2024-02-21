# from tabnanny import check
# from turtle import color
# import torch
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from captum.attr import IntegratedGradients
# from torch_sparse import coalesce
# from torch_geometric import utils
# import pickle
# import os
# import argparse
# import pytorch_lightning as pl
# from dataset import TissueDataset
# import torch_geometric as pyg
# from torch_geometric.loader import DataLoader
# from model_dgermen import CustomGCN 
# from torch_geometric.utils import degree
# import customTools

# import matplotlib.colors as mcolors


# S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
# OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
# RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

# parser = argparse.ArgumentParser(description='GNN Arguments')
# parser.add_argument(
#     '--bs',
#     type=int,
#     default=32,
#     metavar='BS',
#     help='batch size (default: 32)')

# parser.add_argument(
#     '--idx',
#     type=int,
#     default=15,
#     metavar='IDX',
#     help='Index Number of the graph (default: 14)')

# parser.add_argument(
#     '--exp_epoch',
#     type=int,
#     default=50,
#     metavar='EXEPC',
#     help='Number of epochs (default: 50)')

# parser.add_argument(
#     '--exp_lr',
#     type=float,
#     default=0.001,
#     metavar='ELR',
#     help='Explainer learning rate (default: 0.001)')

# parser.add_argument(
#     '--retun_type',
#     type=str,
#     default="regression",
#     metavar='RT',
#     help='return type of the explainer (default: regression)')

# parser.add_argument(
#     '--feat_mask_type',
#     type=str,
#     default="individual_feature",
#     metavar='FMT',
#     help='feature mask type of the explainer (default: individual_feature)')

# parser.add_argument(
#     '--relevant_edges',
#     type=float,
#     default=0.5,
#     metavar='RE',
#     help='Get the highest values inside the edge mask (relevant edges) (default: 0.001)')

# parser.add_argument(
#     '--model',
#     type=str,
#     default="PNAConv",
#     metavar='mn',
#     help='model name (default: PNAConv)')

# parser.add_argument(
#     '--num_of_gcn_layers',
#     type=int,
#     default=2,
#     metavar='NGCNL',
#     help='Number of GCN layers (default: 2)')

# parser.add_argument(
#     '--num_of_ff_layers',
#     type=int,
#     default=1,
#     metavar='NFFL',
#     help='Number of FF layers (default: 2)')

# parser.add_argument(
#     '--gcn_h',
#     type=int,
#     default=64,
#     metavar='GCNH',
#     help='GCN hidden channel (default: 128)')

# parser.add_argument(
#     '--fcl',
#     type=int,
#     default=256,
#     metavar='FCL',
#     help='Number of neurons in fully-connected layer (default: 128)')

# parser.add_argument(
#     '--dropout',
#     type=float,
#     default=0.0, 
#     metavar='DO',
#     help='dropout rate (default: 0.20)')

# parser.add_argument(
#     '--aggregators',
#     nargs='+',
#     type = str,
#     default= ["max"], 
#     metavar='AGR',
#     help= "aggregator list for PNAConv")

# parser.add_argument(
#     '--scalers',
#     nargs='+',
#     type= str,
#     default= ["identity"],
#     metavar='SCL',
#     help='Set of scaling function identifiers,')

# S_PATH = os.path.dirname(__file__)
# pl.seed_everything(42)
# args = parser.parse_args()

# use_gpu = torch.cuda.is_available()

# device = "cpu"

# if use_gpu:
#     device = "cuda"
# else:
#     print("CPU is available on this device!")


# dataset = TissueDataset(os.path.join(S_PATH,"../data"))

# dataset = dataset.shuffle()

# num_of_train = int(len(dataset)*0.80)
# num_of_val = int(len(dataset)*0.10)

# train_dataset = dataset[:num_of_train]
# validation_dataset = dataset[num_of_train:num_of_train+num_of_val]
# test_dataset = dataset[num_of_train+num_of_val:]

# train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
# validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

# idx = args.idx
# test_graph = test_dataset[idx]
# coordinates_arr = None
# with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
#         coordinates_arr = pickle.load(handle)

# deg = 1

# if args.model == "PNAConv":
#     max_degree = -1
#     for data in train_dataset:
#         d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
#         max_degree = max(max_degree, int(d.max()))

#     deg = torch.zeros(max_degree + 1, dtype=torch.long)
#     for data in train_dataset:
#         d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
#         deg += torch.bincount(d, minlength=deg.numel())

# def load_checkpoint(filepath):
    
#     filepath = "/Users/dgermen/Documents/GCN Clinical Clinical Outcome Prediction/GNNClinicalOutcomePrediction/data/models/model.pth"

#     model = CustomGCN(type = args.model,
#                 num_node_features = dataset.num_node_features,
#                 num_gcn_layers=args.num_of_gcn_layers, 
#                 num_ff_layers=args.num_of_ff_layers, 
#                 gcn_hidden_neurons=args.gcn_h, 
#                 ff_hidden_neurons=args.fcl, 
#                 dropout=args.dropout,
#                 aggregators=args.aggregators,
#                 scalers=args.scalers,
#                 deg = deg).to(device)
#     checkpoint = torch.load(filepath, map_location="cpu")
#     model.load_state_dict(checkpoint)
#     model.eval()
#     return model

# model = load_checkpoint('../data/models/model.pth')

# customTools.save_model(model = model, fileName="TheModel", mode = "SDH",path = (os.getcwd() + "/data/models/"))

# import bin.custom_tools as custom_tools
# import os

# aa = custom_tools.load_model(fileName="TheModel_EM", path = (os.getcwd() + "/data/models/"))



# print("nooo")

# import custom_tools

# args  = custom_tools.general_parser()

# print(args.lr)



model = customTools.model_fast()

pass