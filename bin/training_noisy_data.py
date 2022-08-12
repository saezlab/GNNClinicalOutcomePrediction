import argparse
from platform import node
import numpy as np
import random
import time
from itertools import product
import torch
import warnings
import os


import matplotlib.pyplot as plt
import networkx as nx
import pickle5 as pickle
import pytorch_lightning as pl
import torch_geometric as pyg
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from explainer_base import GNNExplainer, LIME, SHAP
from torch_geometric.data import Data
import pandas as pd
import plotting
from torch.optim.lr_scheduler import ReduceLROnPlateau


from dataset import TissueDataset
from model_dgermen import CustomGCN
from evaluation_metrics import r_squared_score



parser = argparse.ArgumentParser(description='GNN Arguments')

parser.add_argument(
    '--epoch',
    type=int,
    default=2,
    metavar='EPC',
    help='Number of epochs (default: 50)')

parser.add_argument(
    '--en',
    type=str,
    default="my_experiment",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')

parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.001)')

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
                    default=[0],
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

parser.add_argument("--node_explainers", type=list, default=['GraphSVX', 'Greedy', 'GNNExplainer'],
                        help="Name of the benchmarked explainers among Greedy, GNNExplainer and GraphSVX")
parser.add_argument("--test_samples", type=int,
                        help='number of test samples for evaluation')
parser.add_argument("--K", type=float,
                        help='proportion of most important features considered, among non zero ones')
parser.add_argument("--prop_noise_feat", type=float,
                        help='proportion of noisy features')
parser.add_argument("--prop_noise_nodes", type=float, default=0.20,
                        help='proportion of noisy nodes')
parser.add_argument("--connectedness", type=str,
                        help='how connected are the noisy nodes we define: low, high or medium')
parser.add_argument("--evalshap", type=bool,
                        help='True if want to compare GraphSVX with SHAP for features explanations')
    

S_PATH = os.path.dirname(__file__)
seed = pl.seed_everything(42)
pl.seed_everything(42)
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

device = "cpu"

if use_gpu:
    device = "cuda"
else:
    print("CPU is available on this device!")

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")
S_PATH = os.path.dirname(__file__)
dataset = TissueDataset(os.path.join(S_PATH,"../data"))

dataset = dataset.shuffle()
print('dataset:', type(dataset))
num_of_train = int(len(dataset)*0.80)
num_of_val = int(len(dataset)*0.10)


# Handling string inputs
if type(args.aggregators) != list:
    args.aggregators = args.aggregators.split()

if type(args.scalers) != list:
    args.scalers = args.scalers.split()


num_samples = int(len(dataset))

#print('train_loader:',type(train_loader))

def extract_test_nodes(i, num_samples, seed):

    ####  WE HAVE ISSUES
    """Select some test samples - without repetition

    Args:
        num_samples (int): number of test samples desired

    Returns:
        [list]: list of indexes representing nodes used as test samples
    """
    num_samples = 40
    nodes = []
    np.random.seed(seed)
    dat = i.edge_index.t()
    for k in dat:
        t1= k[0].item()
        t2 = k[1].item()
        nodes.append(t1)
        nodes.append(t2)
    nodes = list(dict.fromkeys(nodes))
    nodes = np.array(nodes)
    node_indices = np.random.choice(nodes, num_samples, replace=False).tolist()

    return node_indices



def add_noise_neighbours(data, num_noise, node_indices, binary=False, p=0.5, connectedness='medium', c=0.001):
    """Add noisy nodes to original dataset

    Args:
        data (torch_geometric.Data): downloaded dataset 
        num_noise (int): number of noise features we want to add
        node_indices (list): list of test samples 
        binary (bool, optional): True if want binary node features. Defaults to False.
        p (float, optional): proba that each binary feature = 1. Defaults to 0.5.
        connectedness (str, optional): how connected are new nodes, either 'low', 'medium' or 'high'.
            Defaults to 'high'.

    Returns:
        [torch_geometric.Data]: dataset with additional nodes, with random features and connections
    """
    if not num_noise:
        return data

    # Number of features in the dataset
    num_feat = data.x.size(1)
    num_nodes = data.x.size(0)

    # Add new nodes with random features
    m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
    noise_nei_feat = m.sample((num_feat, num_noise)).T[0]
    if not binary:
        noise_nei_feat_bis = torch.rand((num_noise, num_feat))
        noise_nei_feat = torch.min(noise_nei_feat, noise_nei_feat_bis)
    data.x = torch.cat([data.x, noise_nei_feat], dim=0)
    new_num_nodes = data.x.size(0)

    # Add random edges incident to these nodes - according to desired level of connectivity
    if connectedness == 'high':  # few highly connected new nodes
        adj_matrix = torch.randint(2, size=(num_noise, new_num_nodes))

    elif connectedness == 'medium':  # more sparser nodes, connected to targeted nodes of interest
        m = torch.distributions.bernoulli.Bernoulli(torch.tensor([c]))
        adj_matrix = m.sample((new_num_nodes, num_noise)).T[0]
        # each node of interest has at least one noisy neighbour
        for i, idx in enumerate(node_indices):
            try:
                adj_matrix[i, idx] = 1
            except IndexError:  # in case num_noise < test_samples
                pass
    # low connectivity
    else: 
        adj_matrix = torch.zeros((num_noise, new_num_nodes))
        for i, idx in enumerate(node_indices):
            try:
                adj_matrix[i, idx] = 1
            except IndexError:
                pass
        while num_noise > i+1:
            l = node_indices + list(range(num_nodes, (num_nodes+i)))
            i += 1
            idx = random.sample(l, 2)
            adj_matrix[i, idx[0]] = 1
            adj_matrix[i, idx[1]] = 1

    # Add defined edges to data adjacency matrix, in the correct form
    for i, row in enumerate(adj_matrix):
        indices = (row == 1).nonzero()
        indices = torch.transpose(indices, 0, 1)
        a = torch.full_like(indices, i + num_nodes)
        adj_row = torch.cat((a, indices), 0)
        data.edge_index = torch.cat((data.edge_index, adj_row), 1)
        adj_row = torch.cat((indices, a), 0)
        data.edge_index = torch.cat((data.edge_index, adj_row), 1)

    # Update train/test/val masks - don't include these new nodes anywhere as there have no labels

    test_mask = torch.empty(num_noise, data.x.shape[1])
    test_mask = torch.full_like(test_mask, False).bool()
    data.x = torch.cat((data.x, test_mask), 0)

    return data



def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    pred_list = []
    out_list = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
    
        loss.backward()  # Derive gradients.
        out_list.extend([val.item() for val in out.squeeze()])
        
        pred_list.extend([val.item() for val in data.y])

        total_loss += float(loss.item())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients


    return total_loss

def test(model, loader, criterion, label=None, fl_name=None, plot_pred=False):
    model.eval()

    total_loss = 0.0
    pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list = [], [], [], [], [], [], []

    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
        total_loss += float(loss.item())

        true_list.extend([val.item() for val in data.y])
        pred_list.extend([val.item() for val in out.squeeze()])
        tumor_grade_list.extend([val for val in data.tumor_grade])
        clinical_type_list.extend([val for val in data.clinical_type])
        osmonth_list.extend([val for val in data.osmonth])
        pid_list.extend([val for val in data.p_id])
        img_list.extend([val for val in data.img_id])
    
    if plot_pred:
        #plotting.plot_pred_vs_real(df, 'OS Month (log)', 'Predicted', "Clinical Type", fl_name)
        
        label_list = [label]*len(clinical_type_list)
        df = pd.DataFrame(list(zip(pid_list, img_list, true_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list, label_list)),
               columns =["Patient ID","Image Number", 'OS Month (log)', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month", "Train Val Test"])
        
        return total_loss, df
    else:
        return total_loss




def filter_useless_nodes_multiclass(args_hops=args.hops,
                                    args_num_samples=args.num_samples,
                                    args_test_samples=40,
                                    args_prop_noise_nodes=args.prop_noise_nodes,
                                    args_connectedness=args.connectedness,
                                    node_indices=None,
                                    args_K=5,
                                    info=args.info,
                                    #args_hv=args.hv,
                                    args_feat=args.feat,
                                    args_coal=args.coal,
                                    args_g=args.g,
                                    args_multiclass=args.multiclass,
                                    args_regu=args.regu,
                                    args_gpu=args.gpu,
                                    args_fullempty=args.fullempty,
                                    args_S=args.S): 
                                    #seed=args.seed):
    """ Add noisy neighbours to dataset and check how many are included in explanations
    The fewest, the better the explainer.
    Args:
        Arguments defined in argument parser of script_eval.py
    
    Referance: https://github.com/AlexDuvalinho/GraphSVX/blob/master/src/eval_multiclass.py#L269
    """

    noisy_dataset = []
    for i in dataset:

        args_num_noise_nodes = int(args_prop_noise_nodes * i.x.size(0))
        if not node_indices:
            node_indices = extract_test_nodes(i, args_test_samples, seed=seed)

        # Add noisy neighbours to the graph, with random features
        data = add_noise_neighbours(i, args_num_noise_nodes, node_indices,
                                    binary=False, p=0.5, connectedness='medium')
        noisy_dataset.append(data)
            #print('data:',type(data))
    print('noisy_dataset:', type(noisy_dataset))
    num_of_train = int(len(dataset)*0.80)
    num_of_val = int(len(dataset)*0.10)
    
    train_dataset = noisy_dataset[:num_of_train]
    validation_dataset = noisy_dataset[num_of_train:num_of_train+num_of_val]
    test_dataset = noisy_dataset[num_of_train+num_of_val:]

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    best_val_loss = np.inf
    best_train_loss = np.inf
    best_test_loss = np.inf

    deg = 1

    # Calculating degree
    if args.model == "PNAConv":
        max_degree = -1
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

    model = CustomGCN(
                type = args.model,
                num_node_features = 33, ####### LOOOOOOOOK HEREEEEEEEEE
                num_gcn_layers=args.num_of_gcn_layers, 
                num_ff_layers=args.num_of_ff_layers, 
                gcn_hidden_neurons=args.gcn_h, 
                ff_hidden_neurons=args.fcl, 
                dropout=args.dropout,
                aggregators=args.aggregators,
                scalers=args.scalers,
                deg = deg # Comes from data not hyperparameter
                    ).to(device)

    print_at_each_epoch = True

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    args_list = sorted(vars(args).keys())
    args_str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in args_list])

    print(args_str)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor= args.factor, patience=args.patience, min_lr=args.min_lr, verbose=True)


    for epoch in range(1, args.epoch):

        train(model, train_loader, optimizer, criterion)

        train_loss, validation_loss, test_loss = np.inf, np.inf, np.inf
        plot_at_last_epoch = True
        if epoch== args.epoch-1:

            train_loss, df_train = test(model, train_loader, criterion, "train", "train", plot_at_last_epoch)
            validation_loss, df_val= test(model, validation_loader, criterion, "validation", "validation", plot_at_last_epoch)
            test_loss, df_test = test(model, test_loader, criterion, "test", "test", plot_at_last_epoch)
            list_ct = list(set(df_train["Clinical Type"]))
            r2_score = r_squared_score(df_val['OS Month (log)'], df_val['Predicted'])

            if r2_score>0.7:

                df2 = pd.concat([df_train, df_val, df_test])
                df2.to_csv(f"{OUT_DATA_PATH}/{args_str}.csv", index=False)
                plotting.plot_pred_(df2, list_ct, args_str)


        else:
            train_loss = test(model, train_loader, criterion)
            validation_loss= test(model, validation_loader, criterion)
            test_loss = test(model, test_loader, criterion)
        

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_train_loss = train_loss
            best_test_loss = test_loss
        
        if print_at_each_epoch:
            print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}, Test loss: {test_loss:.4f}')


    torch.save(model.state_dict(), '../data/models/noisy_model.pth')
    
    return print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")


filter_useless_nodes_multiclass(args_hops=args.hops,
                                    args_num_samples=args.num_samples,
                                    args_test_samples=args.test_samples,
                                    args_prop_noise_nodes=args.prop_noise_nodes,
                                    args_connectedness=args.connectedness,
                                    node_indices=None,
                                    args_K=5,
                                    info=args.info,
                                    #args_hv=args.hv,
                                    args_feat=args.feat,
                                    args_coal=args.coal,
                                    args_g=args.g,
                                    args_multiclass=args.multiclass,
                                    args_regu=args.regu,
                                    args_gpu=args.gpu,
                                    args_fullempty=args.fullempty,
                                    args_S=args.S) 
                                    #seed=args.seed)
