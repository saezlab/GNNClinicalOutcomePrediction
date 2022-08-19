import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import pickle5 as pickle
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

import scipy
from copy import deepcopy
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
    default=15,
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

S_PATH = os.path.dirname(__file__)
pl.seed_everything(42)
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

device = "cuda"

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



class SHAP():
    """ KernelSHAP explainer - adapted to GNNs
    Explains only node features
    """
    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        # number of nonzero features - for each node index
        self.M = self.data.num_features
        self.neighbours = None
        self.F = self.M

        self.model.eval()

    def explain(self, node_index=0, hops=2, num_samples=10, info=True, multiclass=False, *unused):
        """
        :param node_index: index of the node of interest
        :param hops: number k of k-hop neighbours to consider in the subgraph around node_index
        :param num_samples: number of samples we want to form GraphSVX's new dataset 
        :return: shapley values for features that influence node v's pred
        """
        # Compute true prediction of model, for original instance
        with torch.no_grad():
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            batch = torch.zeros(self.data.x.shape[0], dtype=int, device=self.data.x.device).cuda()
            true_conf, true_pred = self.model(x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda(), batch=batch).exp()[node_index][0].max(dim=0)

        # Determine z => features whose importance is investigated
        # Decrease number of samples because nodes are not considered
        num_samples = num_samples//3

        # Consider all features (+ use expectation like below)
        # feat_idx = torch.unsqueeze(torch.arange(self.F), 1)

        # Sample z - binary vector of dimension (num_samples, M)
        z_ = torch.empty(num_samples, self.M).random_(2)
        # Compute |z| for each sample z
        s = (z_ != 0).sum(dim=1)

        # Define weights associated with each sample using shapley kernel formula
        weights = self.shapley_kernel(s)

        # Create dataset (z, f(z')), stored as (z_, fz)
        # Retrive z' from z and x_v, then compute f(z')
        fz = self.compute_pred(node_index, num_samples,
                            z_, multiclass, true_pred)

        # OLS estimator for weighted linear regression
        phi, base_value = self.OLS(z_, weights, fz)  # dim (M*num_classes)

        return phi

    def shapley_kernel(self, s):
        """
        :param s: dimension of z' (number of features + neighbours included)
        :return: [scalar] value of shapley value 
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1/self.M)
            else:
                shap_kernel.append(
                    (self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
        return torch.tensor(shap_kernel)

    def compute_pred(self, node_index, num_samples, z_, multiclass, true_pred):
        """
        Variables are exactly as defined in explainer function, where compute_pred is used
        This function aims to construct z' (from z and x_v) and then to compute f(z'), 
        meaning the prediction of the new instances with our original model. 
        In fact, it builds the dataset (z, f(z')), required to train the weighted linear model.
        :return fz: probability of belonging to each target classes, for all samples z'
        fz is of dimension N*C where N is num_samples and C num_classses. 
        """
        # This implies retrieving z from z' - wrt sampled neighbours and node features
        # We start this process here by storing new node features for v and neigbours to
        # isolate
        X_v = torch.zeros([num_samples, self.F])

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Do it for each sample
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Features where z_j == 1 are kept, others are set to 0
            for j in range(self.F):
                if z_[i, j].item() == 1:
                    X_v[i, j] = 1

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)
            X[node_index, :] = X_v[i, :]

            # Apply model on (X,A) as input.
            with torch.no_grad():
                
                batch = torch.zeros(X.shape[0], dtype=int, device=X.device).cuda()
                proba = self.model(x=X.cuda(), edge_index=self.data.edge_index.cuda(), batch=batch).exp()[
                        node_index]
            # Multiclass
            if not multiclass:
                fz[i] = proba[true_pred]
            else:
                fz[i] = proba

        return fz

    def OLS(self, z_, weights, fz):
        """
        :param z_: z - binary vector  
        :param weights: shapley kernel weights for z
        :param fz: f(z') where z is a new instance - formed from z and x
        :return: estimated coefficients of our weighted linear regression - on (z, f(z'))
        phi is of dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            '''
            numpy.linalg.LinAlgError: Singular matrix
            Use SVD or QR-decomposition to calculate exact solution in real or complex number fields:
            numpy.linalg.svd numpy.linalg.qr
            '''
            tmp = np.linalg.qr(np.dot(np.dot(z_.T, np.diag(weights)), z_))
            #tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(0.00001 * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.cpu().detach().numpy()))

        # Test accuracy
        # y_pred=z_.detach().numpy() @ phi
        #	print('r2: ', r2_score(fz, y_pred))
        #	print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]


def SHAPEx():
    '''
    Learns and returns a node feature mask and an edge mask that play a crucial role to explain the prediction made by the GNN for a graph.
    test_graph.x (torch.Tensor): The node feature matrix
    test_graph.edge_index (torch.Tensor): The edge indices.

    
    Additional hyper-parameters to override default settings in coeffs.
    coeffs = {'edge_ent': 1.0, 'edge_reduction': 'sum', 'edge_size': 0.005, 'node_feat_ent': 0.1, 'node_feat_reduction': 'mean', 'node_feat_size': 1.0}
    '''

    explainer = SHAP(test_graph, model)
    
    result = explainer.explain(node_index=0, hops=2, num_samples=10, info=True, multiclass=False)

    return result

results = SHAPEx()
print('test_graphs:', test_graph.x.shape)
print('results:', results.shape)
