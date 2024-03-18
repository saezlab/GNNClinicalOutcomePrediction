import argparse
from platform import node
from turtle import color
import numpy as np
import random
import time
from itertools import product
import torch
import warnings
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
import networkx as nx
import pickle5 as pickle
import pytorch_lightning as pl
import torch_geometric as pyg
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from bin.explainer_base import GNNExplainer, LIME, SHAP
from torch_geometric.data import Data
import pandas as pd
import plotting
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter

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

parser.add_argument("--training", 
                    type=bool,
                    default = False,
                    help='False if we consider training noisy data for explanation')
                    
parser.add_argument("--save_model", 
                    type=bool,
                    default = False,
                    help='False if we consider training noisy data for explanation')

parser.add_argument("--att", 
                    type=bool,
                    default = False,
                    help='False if we consider attention weights in GATConv')

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

parser.add_argument("--explainer_name", 
                    type=str,
                    default = 'GNNExplainer',
                    help="Choosen explainer")

parser.add_argument("--regu", 
                    type=float,
                    default = None,
                    help='None if we do not apply regularisation, \
                        1 if we focus only on features in explanations, 0 for nodes')

parser.add_argument("--node_explainers", type=list, default=['GraphSVX', 'Greedy', 'GNNExplainer'],
                        help="Name of the benchmarked explainers among Greedy, GNNExplainer and GraphSVX")
parser.add_argument("--test_samples", type=int, default=20,
                        help='number of test samples for evaluation')
parser.add_argument("--K", type=float,
                        help='proportion of most important features considered, among non zero ones')
parser.add_argument("--prop_noise_nodes", type=float, default=0.020,
                        help='proportion of noisy nodes')
parser.add_argument("--prop_noise_feat", type=float, default=0.20,
                        help='proportion of noisy features')
parser.add_argument("--connectedness", type=str,
                        help='how connected are the noisy nodes we define: low, high or medium')
parser.add_argument("--noise", type=str, default='Node',
                        help='how connected are the noisy nodes we define: low, high or medium')
parser.add_argument("--noisy_data", type=bool, default=True,
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
num_of_train = int(len(dataset)*0.80)
num_of_val = int(len(dataset)*0.10)

original_test_dataset = dataset[num_of_train+num_of_val:]

# Handling string inputs
if type(args.aggregators) != list:
    args.aggregators = args.aggregators.split()

if type(args.scalers) != list:
    args.scalers = args.scalers.split()


num_samples = int(len(dataset))

#print('train_loader:',type(train_loader))

def extract_test_nodes(args_prop_noise_nodes, seed):

    ####  WE HAVE ISSUES
    """Select some test samples - without repetition

    Args:
        num_samples (int): number of test samples desired

    Returns:
        [list]: list of indexes representing nodes used as test samples
    """
    

    test_dataset = dataset[num_of_train+num_of_val:]

    for i in test_dataset:
        #num_samples = int(args_prop_noise_nodes * i.x.size(0))
        num_samples = 20
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
        #test_indices = nodes.numpy().nonzero()[0]
        #test_indices = data.test_mask.cpu().numpy().nonzero()[0]
        node_indices = np.random.choice(nodes, num_samples, replace=False).tolist()

    return node_indices


def add_noise_features(data, num_noise, binary=False, p=0.5):
    """Add noisy features to original dataset

    Args:
        data (torch_geometric.Data): downloaded dataset 
        num_noise ([type]): number of noise features we want to add
        binary (bool, optional): True if want binary node features. Defaults to False.
        p (float, optional): Proportion of 1s for new features. Defaults to 0.5.

    Returns:
        [torch_geometric.Data]: dataset with additional noisy features
    """

    # Do nothing if no noise feature to add
    if not num_noise:
        return data, None

    # Number of nodes in the dataset
    #print('original data:',data)
    num_nodes = data.x.size(0)

    # Define some random features, in addition to existing ones
    m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
    noise_feat = m.sample((num_noise, num_nodes)).T[0]
    # noise_feat = torch.randint(2,size=(num_nodes, num_noise))
    if not binary:
        noise_feat_bis = torch.rand((num_nodes, num_noise))
        # noise_feat_bis = noise_feat_bis - noise_feat_bis.mean(1, keepdim=True)
        noise_feat = torch.min(noise_feat, noise_feat_bis)
    data.x = torch.cat([noise_feat, data.x], dim=-1)
    #print('new data:',data)

    return data, noise_feat


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
    #print('data:',data)
    # Number of features in the dataset
    num_feat = data.x.size(1)
    num_nodes = data.x.size(0)

    #print('num_noise:',num_noise)

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
    
    # Define some random positios, in addition to existing ones
    data_pos_t = data.pos.T

    max_pos_value_x = torch.max(data_pos_t[0])
    max_pos_value_y = torch.max(data_pos_t[1])
    min_pos_value_x = torch.min(data_pos_t[0])
    min_pos_value_y = torch.min(data_pos_t[1])

    noise_pos_x = (max_pos_value_x-min_pos_value_x)*torch.rand((num_noise,1)) + min_pos_value_x
    
    noise_pos_y = (max_pos_value_y-min_pos_value_y)*torch.rand((num_noise,1)) + min_pos_value_y
    noise_pos = torch.cat([noise_pos_x, noise_pos_y], dim=1)
    
    '''m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
    noise_pos = m.sample((2, num_noise)).T[0]

    if not binary:
        noise_pos_bis = torch.rand((num_noise, 2))
        noise_pos = torch.min(noise_pos_bis, noise_pos)

    noise_pos = noise_pos * 1000'''

    data.pos = torch.cat([data.pos, noise_pos], dim=0)
    #print("new data:", data)


    return data



def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    pred_list = []
    out_list = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        #print('type out:', type(model(data.x.to(device), data.edge_index.to(device), data.batch.to(device), att=args.att)))
        if args.att is True:
            out, alpha = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device), att=args.att)#.type(torch.DoubleTensor).to(device) # Perform a single forward pass.
            out = out.type(torch.DoubleTensor).to(device) # Perform a single forward pass
        else:
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device), att=args.att).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        
        # print(data[1])
        # print(data.batch)
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
    
        loss.backward()  # Derive gradients.
        out_list.extend([val.item() for val in out.squeeze()])
        
        pred_list.extend([val.item() for val in data.y])

        total_loss += float(loss.item())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients
        # print("After", model.input.grad)

    return total_loss

def test(model, loader, criterion, label=None, fl_name=None, plot_pred=False):
    model.eval()

    total_loss = 0.0
    pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list = [], [], [], [], [], [], []

    for data in loader:  # Iterate in batches over the training/test dataset.
        if args.att is True:
            out, alpha = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device), att=args.att)#.type(torch.DoubleTensor).to(device) # Perform a single forward pass.
            out = out.type(torch.DoubleTensor).to(device) # Perform a single forward pass
        else:
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device), att=args.att).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
        total_loss += float(loss.item())

        true_list.extend([val.item() for val in data.y])
        pred_list.extend([val.item() for val in out.squeeze()])
        #pred_list.extend([val.item() for val in data.y])
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


class Random:
    """Random explainer - selects random variables as explanations
    """

    def __init__(self, num_feats, K):
        self.num_feats = num_feats
        self.K = K

    def explain(self):
        return np.random.choice(self.num_feats, self.K)

def noise_nodes_for_random(data, model, K, node_indices, total_num_noisy_nei, total_neigbours):
    """ Random explainer (for neighbours)

    Args: 
        K: number of important features for each test sample
        total_neigbours: list of number of neighbours for node_indices
        total_num_noisy_nei: list of number of noisy features for node_indices
        node_indices: indices of test nodes 
    
    Returns:
        Number of times noisy features are provided as explanations 
    """
    pred_class_num_noise_neis = []

    for j, _ in enumerate(node_indices):

        # Use Random explainer - on neighbours (not features)
        explainer = Random(total_neigbours[j], K[j])

        # Store indexes of K most important features, for each class
        nei_indices = explainer.explain()

        # Number of noisy features that appear in explanations - use index to spot them
        num_noise_nei = sum(
                    idx >= (total_neigbours[j]-total_num_noisy_nei[j]) for idx in nei_indices)

        pred_class_num_noise_neis.append(num_noise_nei)

    return pred_class_num_noise_neis

def noise_feats_for_random(dataset, model, K, args_num_noise_feat, node_indices):
    """ Random explainer

    Args: 
        K: number of most important features we look at 
        args_num_noise_feat: number of noisy features 
        node_indices: indices of test nodes 
    
    Returns:
        Number of times noisy features are provided as explanations 
    """
    # Loop on each test sample and store how many times do noise features appear among
    # K most influential features in our explanations
    for data in dataset:
        pred_class_num_noise_feats = []

        for j, node_idx in enumerate(node_indices):

            # Use Random explainer
            explainer = Random(data.x.size(1), K[j])

            # Store indexes of K most important features, for each class
            feat_indices = explainer.explain()

            # Number of noisy features that appear in explanations - use index to spot them
            num_noise_feat = sum(
                idx < args_num_noise_feat for idx in feat_indices)

            pred_class_num_noise_feats.append(num_noise_feat)

        return pred_class_num_noise_feats



def study_attention_weights(data, model, args_test_samples, batch):
    """
        Studies the attention weights of the GAT model
        """
    #_, alpha, alpha_bis = model(data.x.cuda(), data.edge_index.cuda(), batch=batch).type(torch.DoubleTensor).cuda()

    _, alpha = model(data.x.cuda(), data.edge_index.cuda(), batch=batch, att = True)
    #print('alpha[0]:', alpha[0])
    #print('alpha[1]:', alpha[1].shape)
    #alpha = alpha.type(torch.DoubleTensor).cuda()
    alpha = alpha[1]

    # remove self loops att
    #edges, alpha1 = alpha[0][:, :-
    #                      (data.x.size(0))], alpha[1][:-(data.x.size(0)), :]
    edges, alpha1 = alpha[:, :-
                          (data.x.size(0))], alpha[:-(data.x.size(0)), :]
    #alpha2 = alpha_bis[1][:-(data.x.size(0))]

    # Look at all importance coefficients of noisy nodes towards normal nodes
    att1 = []
    #att2 = []
    for i in range(data.x.size(0) - args_test_samples, (data.x.size(0)-1)):
        ind = (edges == i).nonzero()
        for j in ind[:, 1]:
            att1.append(torch.mean(alpha1[j]))
    #        att2.append(alpha2[j][0])
    #print('shape attention noisy', len(att2))

    # It looks like these noisy nodes are very important
    #print('att1:',att1)
    print('av attention',  (torch.mean(alpha1) ))#+ torch.mean(alpha2))/2)  # 0.18

    ## There is Issue Here, attention list becomes empty ---> Why????
    #(torch.mean(torch.stack(att1))) #+ torch.mean(torch.stack(att2)))/2  # 0.32

    # In fact, noisy nodes are slightly below average in terms of attention received
    # Importance of interest: look only at imp. of noisy nei for test nodes
    ## There is Issue Here, attention list becomes empty ---> Why????
    #print('attention 1 av. for noisy nodes: ',
    #        torch.mean(torch.stack(att1[0::2])))
    #print('attention 2 av. for noisy nodes: ',
    #        torch.mean(torch.stack(att2[0::2])))

    return torch.mean(alpha[1], axis=0)

def plot_original_graph(or_test_graph):
    clinical_type = or_test_graph.clinical_type
    img_id = or_test_graph.img_id
    p_id = or_test_graph.p_id
    pos_1 = or_test_graph.pos.numpy()
    g = utils.to_networkx(or_test_graph, to_undirected=True)
    nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
    nx.draw_networkx_edges(g, edge_color='b', pos=pos_1)

    plt.savefig(f'../plots/original_graphs_2/original_graph_{clinical_type}_{img_id}_{p_id}.png', dpi=1000)

    return plt.close()


def filter_useless_features(args_hops=args.hops,
                                    args_num_samples=args.num_samples,
                                    args_test_samples=40,
                                    args_prop_noise_feat=args.prop_noise_feat, ##change here
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
        '''print('dataset:',dataset)
        print('i:',i)'''
        args_num_noise_feat = int(i.x.size(1) *args_prop_noise_feat)
        if not node_indices:
            node_indices = extract_test_nodes(args_prop_noise_feat, seed=seed)

        #for k in dataset[:num_of_train]:
            #print('k:', k)
        # Add noisy neighbours to the graph, with random features
        data, noise_feat = add_noise_features(i, num_noise=args_num_noise_feat,
                            binary=False, p=0.5)
        noisy_dataset.append(data)
            #print('data:',type(data))
        #print('data3:', i)
        #print('new_data:', data)
    #print('noisy_dataset:', type(noisy_dataset))
    num_of_train = int(len(dataset)*0.80)
    num_of_val = int(len(dataset)*0.10)
    
    train_dataset = noisy_dataset[:num_of_train]
    validation_dataset = noisy_dataset[num_of_train:num_of_train+num_of_val]
    test_dataset = noisy_dataset[num_of_train+num_of_val:]

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    #plot_random_graph(dataset, test_dataset, idx = 1)
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
                num_node_features = data.x.size(1), ####### LOOOOOOOOK HEREEEEEEEEE
                num_gcn_layers=args.num_of_gcn_layers, 
                num_ff_layers=args.num_of_ff_layers, 
                gcn_hidden_neurons=args.gcn_h, 
                ff_hidden_neurons=args.fcl, 
                dropout=args.dropout,
                aggregators=args.aggregators,
                scalers=args.scalers,
                deg = deg # Comes from data not hyperparameter
                    ).to(device)

    print('Model:',model)
    print_at_each_epoch = True
    

    if args.training is True:
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

                '''if r2_score>0.7:

                    df2 = pd.concat([df_train, df_val, df_test])
                    df2.to_csv(f"{OUT_DATA_PATH}/{args_str}.csv", index=False)
                    # print(list_ct)
                    # plotting.plot_pred_vs_real_lst(df2, ['OS Month (log)']*3, ["Predicted"]*3, "Clinical Type", list_ct, args_str)
                    plotting.plot_pred_(df2, list_ct, args_str)'''


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
        
        if args.save_model is True:
            print('Saving trained model...')
            if args.model == 'GATConv':
                torch.save(model.state_dict(), '../data/models/gat_noise_node_feature_model.pth')
            elif args.model == 'PNAConv':
                torch.save(model.state_dict(), '../data/models/pna_noise_node_feature_model.pth')

            print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")

        else:
            print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")

        # Define dataset
    else:
        args_list = sorted(vars(args).keys())
        args_str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in args_list])
        print(args_str)
        print('Loading pretrained model...')
        if args.model == 'PNAConv':
            checkpoint = torch.load('../data/models/pna_noise_node_feature_model.pth')
        elif args.model == 'GATConv':
            checkpoint = torch.load('../data/models/gat_noise_node_feature_model.pth')        
        model.load_state_dict(checkpoint)
        model = model.eval()

    
    # Adaptable K - top k explanations we look at for each node
    # Depends on number of existing features considered for GraphSVX
    '''if 'GraphSVX' in args_explainers:
        K = []
    else:
        K = [10]*len(node_indices)'''
    K = [10]*len(node_indices)
    #for node_idx in node_indices:
    #	K.append(int(data.x[node_idx].nonzero().shape[0] * args_K))

    if args_regu == 0:
        args_regu = 1
    
    pred_class_num_noise_feats_in_test_dataset = []

    total_num_noise_feat_considered_in_test_dataset = []
    # count number of features   
    F_in_test_dataset = []
    
    for i in range(len(test_dataset)):
        test_graph = test_dataset[i]

        # Define explainer
        print('EXPLAINER: ', args.explainer_name)

        # count noisy features found in explanations 
        pred_class_num_noise_feats = []
        # count number of noisy features considered
        total_num_noise_feat_considered = []
        # count number of features   
        F = []

        # Loop on each test sample and store how many times do noise features appear among
        # K most influential features in our explanations
        j=0
        for node_idx in tqdm(node_indices, desc='explain feature', leave=False):

            # Look only at coefficients for nodes (not node features)
            if args.explainer_name == 'GNNExplainer':

                explainer = GNNExplainer(model, test_graph, epochs = args.exp_epoch, lr = args.exp_lr, weight_decay=args.weight_decay,
                                    factor = args.factor, patience = args.patience, min_lr = args.min_lr,
                                    edge_size = args.edge_size, node_feat_size= args.node_feat_size, edge_ent = args.edge_ent, node_feat_ent = args.node_feat_ent,
                                    return_type = args.retun_type, feat_mask_type = args.feat_mask_type, att=args.att).to(device)
            
                if args.att is True:
                    coefs, _ = explainer.explain_graph(node_idx, test_graph.x.to(device), test_graph.edge_index.to(device))
                    coefs = coefs[1]
                    coefs = coefs[:explainer.Feat]
                else:
                    coefs, _ = explainer.explain_graph(node_idx, test_graph.x.to(device), test_graph.edge_index.to(device))
                    coefs = coefs[1]
                    #coefs = coefs.squeeze()
                    
                    coefs = coefs[:explainer.Feat]

                # All features are considered
                num_noise_feat_considered = args_num_noise_feat
            
            elif args.explainer_name == 'SHAP':

                explainer = SHAP(test_graph, model)

                coefs = explainer.explain(node_idx, hops=2, num_samples=10, info=True, multiclass=False)
            
            elif args.explainer_name == 'LIME':

                explainer = LIME(test_graph, model)

                coefs = explainer.explain(node_index=0, hops=2, num_samples=10, info=True, multiclass=False)
             
                
            # if explainer.F > 50:
            # 	K.append(10)
            # else:
            # 	K.append(int(explainer.F * args_K))

            # Check how many non zero features
            F.append(explainer.Feat)

            # Store indexes of K most important node features, for each class
            feat_indices = coefs.argsort()[-K[j]:].tolist()

            # Number of noisy features that appear in explanations - use index to spot them
            #for indice in feat_indices:

            num_noise_feat = [idx for idx in feat_indices if idx > (explainer.Feat - num_noise_feat_considered)]

            # If node importance of top K features is unsignificant, discard 
            # Possible as we have importance informative measure, unlike others.

            # Count number of noisy that appear in explanations
            num_noise_feat = len(num_noise_feat)
            pred_class_num_noise_feats.append(num_noise_feat)

            # Return number of noisy features considered in this test sample
            total_num_noise_feat_considered.append(num_noise_feat_considered)

            j+=1

        print('Noisy features included in explanations: ',
                        sum(pred_class_num_noise_feats) )
        print('For the predicted class, there are {} noise features found in the explanations of {} test samples, an average of {} per sample'
                        .format(sum(pred_class_num_noise_feats), args_test_samples, sum(pred_class_num_noise_feats)/args_test_samples))

        print(pred_class_num_noise_feats)

        if sum(F) != 0:
            perc = 100 * sum(total_num_noise_feat_considered) / sum(F)
            print(
                'Proportion of considered noisy features among features: {:.2f}%'.format(perc))
        if sum(K) != 0:
            perc = 100 * sum(pred_class_num_noise_feats) / sum(K)
            print('Proportion of explanations showing noisy features: {:.2f}%'.format(perc))

        if sum(total_num_noise_feat_considered) != 0:
            perc = 100 * sum(pred_class_num_noise_feats) / (sum(total_num_noise_feat_considered))
            perc2 = 100 * (sum(K) - sum(pred_class_num_noise_feats)) / (sum(F) - sum(total_num_noise_feat_considered)) 
            print('Proportion of noisy features found in explanations vs proportion of normal features (among considered ones): {:.2f}% vs {:.2f}%, over considered features only'.format(
                perc, perc2))

        print('------------------------------------')
        pred_class_num_noise_feats_in_test_dataset.append(sum(pred_class_num_noise_feats))
        total_num_noise_feat_considered_in_test_dataset.append(sum(total_num_noise_feat_considered))
        F_in_test_dataset.append(sum(F))


    print('Noisy features included in explanations: ',
                        sum(pred_class_num_noise_feats_in_test_dataset) )
    print('For the predicted class, there are {} noise features found in the explanations of {} test samples, an average of {} per sample'
                        .format(sum(pred_class_num_noise_feats_in_test_dataset), args_test_samples * len(test_dataset), sum(pred_class_num_noise_feats_in_test_dataset)/(args_test_samples*len(test_dataset))))

    #if sum(F_in_test_dataset) != 0:
    perc_test_1 = 100 * sum(total_num_noise_feat_considered_in_test_dataset) / sum(F_in_test_dataset)
    print(
                'Proportion of considered noisy features among features: {:.2f}%'.format(perc_test_1))
    #if sum(K) != 0:
    perc_test_2 = 100 * sum(pred_class_num_noise_feats_in_test_dataset) / sum(K)
    print('Proportion of explanations showing noisy features: {:.2f}%'.format(perc_test_2))

    #if sum(total_num_noise_feat_considered_in_test_dataset) != 0:
    perc_test_3 = 100 * sum(pred_class_num_noise_feats_in_test_dataset) / (sum(total_num_noise_feat_considered_in_test_dataset))
    perc2_test = 100 * (sum(K) - sum(pred_class_num_noise_feats_in_test_dataset)) / (sum(F_in_test_dataset) - sum(total_num_noise_feat_considered_in_test_dataset)) 
    print('Proportion of noisy features found in explanations vs proportion of normal features (among considered ones): {:.2f}% vs {:.2f}%, over considered features only'.format(
                perc_test_3, perc2_test))
    
    #################################################################
    print('Noise features found in the explanations of {} test samples: {}'.format(args_test_samples * len(test_dataset), sum(pred_class_num_noise_feats_in_test_dataset)),
            'An average of per sample: {}'.format(sum(pred_class_num_noise_feats_in_test_dataset)/(args_test_samples * len(test_dataset))),
            'Proportion of considered noisy features among features: {:.2f}%'.format(perc_test_1),
            'Proportion of explanations showing noisy features: {:.2f}%'.format(perc_test_2),
            'Proportion of noisy features found in explanations: {:.2f}%'.format(perc_test_3),
            'Proportion of normal features (among considered ones): {:.2f}%'.format(perc2_test)
    )
    # Random explainer - plot estimated kernel density
    total_num_noise_feats = noise_feats_for_random(
        noisy_dataset, model, K, args_num_noise_feat, node_indices)
    #save_path = 'results/eval1_feat'
    #plot_dist(total_num_noise_feats, label='Random', color='y')

    # Store graph - with key params and time
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #plt.savefig('results/eval1_feat_{}_{}_{}_{}_{}.pdf'.format(data.name,
    #                                                       args_coal, 
    #                                                       args_feat, 
    #                                                       args_hv, 
    #                                                       current_time))
    #plt.close()
    # plt.show()
    return total_num_noise_feats



def plot_subgraph(test_graph, original_test_graph, edges_idx, pred_class_num_noise_neis, total_num_noisy_nei, total_neigbours, args_test_samples, K, args_num_noise_nodes):

    

    options = ['r','b','y','g']
    

    G = nx.Graph() 

    noisy = utils.to_networkx(test_graph, to_undirected=True)

    original = utils.to_networkx(original_test_graph, to_undirected=True)

    G.add_nodes_from(original.nodes, color='b')
    G.add_edges_from(original.edges, color='b')

    colors_node = ['b']*len(noisy.nodes)
    colors_edge = []#['b']*len(original.edges)

    noisy_nodes = list(set(list(noisy.nodes))-set(list(original.nodes)))
    noisy_edges = list(set(list(noisy.edges))-set(list(original.edges)))

    G.add_nodes_from(noisy_nodes, color='y')
    G.add_edges_from(noisy_edges, color='y')
    
    for edge in list(G.edges.data('color')):
        n1, n2, color = edge
        colors_edge.append(color)
        colors_node[n1]=color
        colors_node[n2]=color

    for id,e_idx in enumerate(G.edges):
        if edges_idx[id]:
            colors_edge[id] = options[0]
            n1, n2 = e_idx
            colors_node[n1]=options[0]
            colors_node[n2]=options[0]

            if e_idx in noisy_edges:
                colors_edge[id] = options[3]
                n1, n2 = e_idx
                colors_node[n1]=options[3]
                colors_node[n2]=options[3]

    pos_1 = test_graph.pos.numpy()

    nx.draw_networkx_nodes(G, node_color=colors_node, pos=pos_1, node_size=1)
    nx.draw_networkx_edges(G, edge_color=colors_edge, pos=pos_1)

    clinical_type = original_test_graph.clinical_type
    img_id = original_test_graph.img_id
    p_id = original_test_graph.p_id

    clinical_type = test_graph.clinical_type
    img_id = test_graph.img_id
    p_id = test_graph.p_id
    y_label = 'p_noise:{}'.format(args.prop_noise_nodes) 
    

    noisy_node_in_expl = 'g'
    d = Counter(colors_node)
    num_noisy_node_in_expl = d[noisy_node_in_expl]
    print('num_noisy_node_in_expl:', num_noisy_node_in_expl)

    noisy_node_in_graph = 'y'
    d = Counter(colors_node)
    num_noisy_node_in_graph = d[noisy_node_in_graph]
    print('num_noisy_node_in_graph:', num_noisy_node_in_graph)

    ################################################################

    noisy_edge_in_expl = 'g'
    d = Counter(colors_edge)
    num_noisy_edge_in_expl = d[noisy_edge_in_expl]
    print('num_noisy_edge_in_expl:', num_noisy_edge_in_expl)

    noisy_edge_in_graph = 'y'
    d = Counter(colors_edge)
    num_noisy_edge_in_graph = d[noisy_edge_in_graph]
    print('num_noisy_edge_in_graph:', num_noisy_edge_in_graph)

    print('Proportion of # of noisy node in Explanation to total # of noisy node {}%'.format((num_noisy_node_in_expl/(num_noisy_node_in_graph+num_noisy_node_in_expl))*100))
    
    print('Proportion of # of noisy node in Explanation to total # of noisy node {}%'.format((num_noisy_edge_in_expl/(num_noisy_edge_in_graph+num_noisy_edge_in_expl))*100))

    #######################################################################
    x_1 = 'There are {} noise neighbours found in the explanations of {} test samples, an average of {} per sample'.format(sum(pred_class_num_noise_neis), args_test_samples, sum(pred_class_num_noise_neis)/args_test_samples)
    x_2 = 'Proportion of explanations showing noisy neighbours: {:.2f}%'.format(
            100 * sum(pred_class_num_noise_neis) / sum(K))
    perc = 100 * sum(pred_class_num_noise_neis) / (sum(total_num_noisy_nei))
    perc2 = 100 * (sum(K) - sum(pred_class_num_noise_neis)) \
    / (sum(total_neigbours) - sum(total_num_noisy_nei))
    x_3 = 'Proportion of noisy neighbours found in explanations vs normal neighbours (in subgraph): {:.2f}% vs {:.2f}'.format(
            perc, perc2)
    x_4 = 'Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(
            100 * sum(total_num_noisy_nei) / sum(total_neigbours))
    x_5 = 'Proportion of noisy neighbours found in explanations (entire graph): {:.2f}%'.format(
            100 * sum(pred_class_num_noise_neis) / (args_test_samples * args_num_noise_nodes))
    columns = ('Results')
    data = [[x_1], [x_2], [x_3], [x_4], [x_5]]
    rows = ['%d' % x for x in (1, 2, 3, 4, 5)]
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4
    y_offset = np.zeros(len(columns))

    cell_text = []
    for row in range(n_rows):
        #plt.plot(index, data[row], color=colors[row])
        y_offset = data[row]
        cell_text.append([x for x in y_offset])

    cell_text.reverse()

    # Add a table at the bottom of the axes
    '''the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      #rowColours=colors,
                      colLabels=columns,
                      loc='bottom')'''
    #plt.xticks([])
    # Adjust layout to make room for the table:
    #plt.subplots_adjust(left=0.2, bottom=0.2)

    #plt.xlabel(x_label)
    plt.title('Clinincal Type: {}, Image Id: {}, Patient Id: {}'.format(clinical_type,img_id,p_id))
    plt.ylabel(y_label)

    plt.savefig(f'plot/subgraph_{clinical_type}_{img_id}_{p_id}_{args.exp_epoch}_{args.exp_lr}_{args.relevant_edges}_{args.factor}_{args.patience}_{args.noisy_data}_{args.prop_noise_nodes}.png', dpi=1000)

    return plt.close()


def filter_useless_nodes(args_hops=args.hops,
                                    args_num_samples=args.num_samples,
                                    args_test_samples=20,
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
        '''print('dataset:',dataset)
        print('i:',i)'''
        args_num_noise_nodes = int(args_prop_noise_nodes * i.x.size(0))
        if not node_indices:
            node_indices = extract_test_nodes(args_prop_noise_nodes, seed=seed)

        #for k in dataset[:num_of_train]:
            #print('k:', k)
        # Add noisy neighbours to the graph, with random features
        data = add_noise_neighbours(i, args_num_noise_nodes, node_indices,
                                    binary=False, p=0.5, connectedness='medium')
        noisy_dataset.append(data)
            #print('data:',type(data))
        #print('data3:', i)
        #print('new_data:', data)
    #print('noisy_dataset:', type(noisy_dataset))
    num_of_train = int(len(dataset)*0.80)
    num_of_val = int(len(dataset)*0.10)

    
    
    train_dataset = noisy_dataset[:num_of_train]
    validation_dataset = noisy_dataset[num_of_train:num_of_train+num_of_val]
    test_dataset = noisy_dataset[num_of_train+num_of_val:]

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    #plot_random_graph(dataset, test_dataset, idx = 1)
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
                num_node_features = data.x.size(1), ####### LOOOOOOOOK HEREEEEEEEEE
                num_gcn_layers=args.num_of_gcn_layers, 
                num_ff_layers=args.num_of_ff_layers, 
                gcn_hidden_neurons=args.gcn_h, 
                ff_hidden_neurons=args.fcl, 
                dropout=args.dropout,
                aggregators=args.aggregators,
                scalers=args.scalers,
                deg = deg # Comes from data not hyperparameter
                    ).to(device)

    print('Model:',model)
    print_at_each_epoch = True
    

    if args.training is True:
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

                '''if r2_score>0.7:

                    df2 = pd.concat([df_train, df_val, df_test])
                    df2.to_csv(f"{OUT_DATA_PATH}/{args_str}.csv", index=False)
                    # print(list_ct)
                    # plotting.plot_pred_vs_real_lst(df2, ['OS Month (log)']*3, ["Predicted"]*3, "Clinical Type", list_ct, args_str)
                    plotting.plot_pred_(df2, list_ct, args_str)'''


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
        
        if args.save_model is True:
            print('Saving trained model...')
            if args.model == 'GATConv':
                torch.save(model.state_dict(), '../data/models/gat_noise_node_model.pth')
            elif args.model == 'PNAConv':
                torch.save(model.state_dict(), '../data/models/pna_noise_node_model.pth')

            print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")

        else:
            print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")

        # Define dataset
    else:
        args_list = sorted(vars(args).keys())
        args_str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in args_list])
        print(args_str)
        
        if args.model == 'PNAConv':
            if args.noisy_data is True:
                print('Loading pretrained PNA model with noisy dataset...')
                checkpoint = torch.load('../data/models/pna_noise_node_model.pth')
            else:
                print('Loading pretrained PNA model with original dataset...')
                checkpoint = torch.load('../data/models/pna_model.pth')
        elif args.model == 'GATConv':
            if args.noisy_data is True:
                print('Loading pretrained GAT model with original dataset...')
                checkpoint = torch.load('../data/models/gat_noise_node_model.pth')  
            else:
                print('Loading pretrained GAT model with original dataset...')
                checkpoint = torch.load('../data/models/gat_model.pth')        
        model.load_state_dict(checkpoint)
        model = model.eval()

    # Study attention weights of noisy nodes in GAT model - compare attention with explanations
    

    if args.model == "GATConv":
        for i in range(len(test_dataset)):
            test_graph = test_dataset[i]
            batch = torch.zeros(test_graph.x.shape[0], dtype=int, device=test_graph.x.device).cuda()
            study_attention_weights(test_graph, model, args_test_samples, batch)
    
    # Adaptable K - top k explanations we look at for each node
    # Depends on number of existing features/neighbours considered for GraphSVX
    # if 'GraphSVX' in args_explainers:
    # 	K = []
    # else:
    # 	K = [5]*len(node_indices)

    # Do for several explainers
    #for c, explainer_name in enumerate(args_explainers):
    
    
    
    # Define the explainer
    #explainer = eval(explainer_gnnex)(data, model, args_gpu)
    
    pred_class_num_noise_neis_in_test_dataset = []

    total_num_noisy_nei_in_test_dataset =[]

    total_neigbours_in_test_dataset = []

    K_in_test_dataset = []

    t_total = time.time()
    len_test_dataset = 5#len(test_dataset)
    for i in range(len_test_dataset):#((len(test_dataset))//2):

        print('EXPLAINER: ', args.explainer_name)

        test_graph = test_dataset[i]
        original_test_graph = original_test_dataset[i]
        plot_original_graph(original_test_graph)

        print(test_graph)
        # Loop on each test sample and store how many times do noisy nodes appear among
        # K most influential features in our explanations
        # count number of noisy nodes in explanations 
        pred_class_num_noise_neis = []
        # count number of noisy nodes in subgraph
        total_num_noisy_nei = []
        # Number of neigbours of v in subgraph
        total_neigbours = []
        # Stores number of most important neighbours we look at, for each node 
        K = []  
        # To retrieve the predicted class
        j = 0

        t_small = time.time()
        
        for node_idx in tqdm(node_indices, desc='explain node', leave=False):

            # Look only at coefficients for nodes (not node features)

            if args.explainer_name == 'GNNExplainer':

                explainer = GNNExplainer(model, test_graph, epochs = args.exp_epoch, lr = args.exp_lr, weight_decay=args.weight_decay,
                                    factor = args.factor, patience = args.patience, min_lr = args.min_lr,
                                    edge_size = args.edge_size, node_feat_size= args.node_feat_size, edge_ent = args.edge_ent, node_feat_ent = args.node_feat_ent,
                                    return_type = args.retun_type, feat_mask_type = args.feat_mask_type, att=args.att).to(device)
            
                if args.att is True:
                    result = explainer.explain_graph(node_idx, test_graph.x.to(device), test_graph.edge_index.to(device))
                else:
                    result = explainer.explain_graph(node_idx, test_graph.x.to(device), test_graph.edge_index.to(device))

                (feature_mask, edge_mask) = result 

                edges_idx = edge_mask > args.relevant_edges

                #plot_subgraph(test_graph, original_test_graph, edges_idx)

                coefs = explainer.coefs

            
            elif args.explainer_name == 'SHAP':

                explainer = SHAP(test_graph, model)

                coefs = explainer.explain(node_idx, hops=2, num_samples=10, info=True, multiclass=False)
            
            elif args.explainer_name == 'LIME':

                explainer = LIME(test_graph, model)

                coefs = explainer.explain(node_index=0, hops=2, num_samples=10, info=True, multiclass=False)
                

   
            # Number of noisy nodes in the subgraph of node_idx
            num_noisy_nodes = len(
                    [n_idx for n_idx in explainer.neighbour if n_idx.item() >= test_graph.x.size(0)-args_num_noise_nodes])

            # Number of neighbours in the subgraph
            total_neigbours.append(len(explainer.neighbour))

            # Adaptable K - vary according to number of nodes in the subgraph
            if len(explainer.neighbour) > 100:
                K.append(int(args_K * 100))
            else:
                K.append( max(1, int(args_K * len(explainer.neighbour))) )

            # Store indexes of K most important features, for each class
            nei_indices = coefs.argsort()[-K[j]:].tolist()

             # Number of noisy features that appear in explanations - use index to spot them
            noise_nei = [idx for idx in nei_indices if idx > (explainer.neighbour.shape[0] - num_noisy_nodes)]

            num_noise_nei = len(noise_nei)
            pred_class_num_noise_neis.append(num_noise_nei)

            # Return number of noisy nodes adjacent to node of interest
            total_num_noisy_nei.append(num_noisy_nodes)

            j += 1

        plot_subgraph(test_graph, original_test_graph, edges_idx, pred_class_num_noise_neis, total_num_noisy_nei, total_neigbours, args_test_samples, K, args_num_noise_nodes)

        print('Noisy neighbours included in explanations: ',
                        pred_class_num_noise_neis)

        print('There are {} noise neighbours found in the explanations of {} test samples, an average of {} per sample'
                        .format(sum(pred_class_num_noise_neis), args_test_samples, sum(pred_class_num_noise_neis)/args_test_samples))

        print('Proportion of explanations showing noisy neighbours: {:.2f}%'.format(
            100 * sum(pred_class_num_noise_neis) / sum(K)))

        perc = 100 * sum(pred_class_num_noise_neis) / (sum(total_num_noisy_nei))
        perc2 = 100 * (sum(K) - sum(pred_class_num_noise_neis)) \
        / (sum(total_neigbours) - sum(total_num_noisy_nei))
        print('Proportion of noisy neighbours found in explanations vs normal neighbours (in subgraph): {:.2f}% vs {:.2f}'.format(
            perc, perc2))

        print('Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(
            100 * sum(total_num_noisy_nei) / sum(total_neigbours)))

        print('Proportion of noisy neighbours found in explanations (entire graph): {:.2f}%'.format(
            100 * sum(pred_class_num_noise_neis) / (args_test_samples * args_num_noise_nodes)))

        print('Time: {:.4f}s'.format(time.time() - t_small))
        
        print('------------------------------------')
        
        pred_class_num_noise_neis_in_test_dataset.append(sum(pred_class_num_noise_neis))
        total_num_noisy_nei_in_test_dataset.append(sum(total_num_noisy_nei))
        total_neigbours_in_test_dataset.append(sum(total_neigbours))
        K_in_test_dataset.append(sum(K))

    print('Noisy neighbours included in explanation: ',
                        pred_class_num_noise_neis_in_test_dataset)

    print('There are {} noise neighbours found in the explanations of {} test samples, an average of {} per sample'
                        .format(sum(pred_class_num_noise_neis_in_test_dataset), args_test_samples*len_test_dataset, sum(pred_class_num_noise_neis_in_test_dataset)/(args_test_samples*len_test_dataset)))
    print('Proportion of explanations showing noisy neighbours: {:.2f}%'.format(100 * sum(pred_class_num_noise_neis_in_test_dataset) / sum(K_in_test_dataset)))
    perc_test_dataset = 100 * sum(pred_class_num_noise_neis_in_test_dataset) / (sum(total_num_noisy_nei_in_test_dataset))
    perc2_test_dataset = 100 * (sum(K_in_test_dataset) - sum(pred_class_num_noise_neis_in_test_dataset)) \
        / (sum(total_neigbours_in_test_dataset) - sum(total_num_noisy_nei_in_test_dataset))
    print('Proportion of noisy neighbours found in explanations vs normal neighbours (in subgraph): {:.2f}% vs {:.2f}'.format(
            perc_test_dataset, perc2_test_dataset))
    print('Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(
            100 * sum(total_num_noisy_nei_in_test_dataset) / sum(total_neigbours_in_test_dataset)))

    print('Proportion of noisy neighbours found in explanations (entire graph): {:.2f}%'.format(
            100 * sum(pred_class_num_noise_neis_in_test_dataset) / (args_test_samples * args_num_noise_nodes * len_test_dataset)))
    #################################################################################################################
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print(#'Noisy neighbours included in explanation: {}'.format(pred_class_num_noise_neis_in_test_dataset),
              'Noise neighbours found in the explanations of {} test samples: {}'.format(args_test_samples*len_test_dataset, sum(pred_class_num_noise_neis_in_test_dataset)),
              'An average of per sample: {}'.format(sum(pred_class_num_noise_neis_in_test_dataset)/(args_test_samples*len_test_dataset)),
              'Proportion of explanations showing noisy neighbours: {:.2f}%'.format(100 * sum(pred_class_num_noise_neis_in_test_dataset) / sum(K_in_test_dataset)),
              'Proportion of noisy neighbours found in explanations vs normal neighbours (in subgraph): {:.2f}% vs {:.2f}'.format(perc_test_dataset, perc2_test_dataset),
              'Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(100 * sum(total_num_noisy_nei_in_test_dataset) / sum(total_neigbours_in_test_dataset)),
              'Proportion of noisy neighbours found in explanations (entire graph): {:.2f}%'.format(100 * sum(pred_class_num_noise_neis_in_test_dataset) / (args_test_samples * args_num_noise_nodes * len_test_dataset)))

    #print('There are {} noise neighbours found in test dataset'.format(sum(pred_class_num_noise_neis_in_test_dataset)))
    # Random explainer - plot estimated kernel density
    total_num_noise_neis = noise_nodes_for_random(
        noisy_dataset, model, K, node_indices, total_num_noisy_nei, total_neigbours)

    return total_num_noise_neis


if args.noise == 'Node':

    filter_useless_nodes(args_hops=args.hops,
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

else:

    filter_useless_features(args_hops=args.hops,
                                        args_num_samples=args.num_samples,
                                        args_test_samples=args.test_samples,
                                        args_prop_noise_feat=args.prop_noise_feat,
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
