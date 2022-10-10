import torch
from torchmetrics import Accuracy
from data_processing import OUT_DATA_PATH
from model import GCN, GCN2, GCN_NEW, GCN_MNIST_EXP
from dataset import TissueDataset
from torch_geometric.loader import DataLoader
from torch.nn import BatchNorm1d
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
import bin.plotting as plotting
import pandas as pd
import pickle
import os
import pytorch_lightning as pl
from torch_geometric import utils
from evaluation_metrics import r_squared_score, mse, rmse
from explain import saliency_map, grad_cam
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.cm as cm
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
from torchvision.datasets import MNIST
import os.path as osp
import random


S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

parser = argparse.ArgumentParser(description='GNN Arguments')
parser.add_argument(
    '--model',
    type=str,
    default="GCN2",
    metavar='mn',
    help='model name (default: GCN2)')

parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.001)')

parser.add_argument(
    # '--batch-size',
    '--bs',
    type=int,
    default=32,
    metavar='BS',
    help='batch size (default: 32)')

parser.add_argument(
    '--dropout',
    type=float,
    default=0.20, # 0.1 0.2 0.3 0.5
    metavar='DO',
    help='dropout rate (default: 0.20)')

parser.add_argument(
    '--epoch',
    type=int,
    default=50,
    metavar='EPC',
    help='Number of epochs (default: 50)')

parser.add_argument(
    '--num_of_gcn_layers',
    type=int,
    default=3,
    metavar='NGCNL',
    help='Number of GCN layers (default: 2)')

parser.add_argument(
    '--num_of_ff_layers',
    type=int,
    default=3,
    metavar='NFFL',
    help='Number of FF layers (default: 2)')
    
parser.add_argument(
    '--gcn_h',
    type=int,
    default=128,
    metavar='GCNH',
    help='GCN hidden channel (default: 128)')

parser.add_argument(
    '--fcl',
    type=int,
    default=128,
    metavar='FCL',
    help='Number of neurons in fully-connected layer (default: 128)')

parser.add_argument(
    '--en',
    type=str,
    default="my_experiment",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.001,
    metavar='WD',
    help='weight decay (default: 0.001)')

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

S_PATH = os.path.dirname(__file__)
pl.seed_everything(42)
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

device = "cpu"

if use_gpu:
    print("GPU is available on this device!")
    device = "cuda"
else:
    print("CPU is available on this device!")


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
# Load the MNISTSuperpixel dataset
dataset = MNISTSuperpixels(root=".")
print(len(dataset))
image_dataset = MNIST(root=path, download=True)
image_dataset = image_dataset
dataset

# Investigating the dataset
print("Dataset type: ", type(dataset))
print("Dataset features: ", dataset.num_features)
print("Dataset target: ", dataset.num_classes)
print("Dataset length: ", dataset.len)
print("Dataset sample: ", dataset[0])
print("Sample  nodes: ", dataset[0].num_nodes)
print("Sample  edges: ", dataset[0].num_edges)

# writer = SummaryWriter(log_dir=os.path.join(S_PATH,"../logs"))
# dataset = TissueDataset(os.path.join(S_PATH,"../data"))

# dataset = dataset.shuffle()
print(len(dataset))
num_of_train = int(len(dataset)*0.80)
num_of_val = int(len(dataset)*0.10)



train_dataset = dataset[:num_of_train]
validation_dataset = dataset[num_of_train:num_of_train+num_of_val]
test_dataset = dataset[num_of_train+num_of_val:]

"""print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of validation graphs: {len(validation_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f"Number of node features: {dataset.num_node_features}")
"""
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)


"""for step, data in enumerate(test_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
"""
print("NBuraya geldi")
model = GCN_MNIST_EXP(dataset.num_node_features, 
                num_of_gcn_layers=args.num_of_gcn_layers, 
                num_of_ff_layers=args.num_of_ff_layers, 
                gcn_hidden_neurons=args.gcn_h, 
                ff_hidden_neurons=args.fcl, 
                dropout=args.dropout).to(device)


"""if args.model =="GCN":
    model = GCN(dataset.num_node_features, hidden_channels=256).to(device)
elif args.model=="GCN2":
    model =GCN2(dataset.num_node_features, hidden_channels=args.gcn_h, fcl1=args.fcl, drop_rate=args.dropout).to(device)"""

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()


args_list = sorted(vars(args).keys())
args_str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in args_list])
print(args_str)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor= args.factor, patience=args.patience, min_lr=args.min_lr, verbose=True)


def train(input_model=None):
    global model
    if input_model:
        model = input_model
    model.train()
    total_loss = 0.0
    pred_list = []
    out_list = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        # print(data[1])
        # print(data.batch)
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
    
        loss.backward()  # Derive gradients.
        #print(data.x.shape)
        # print("BEFORE", model.input.grad.shape)
        
        if input_model:
            for id in range(len(data)):
                    #   print("========================SALIENCY MAP========================")

                """plt.subplot(1, 3, 1)
                plt.title("MNIST")
                image, label = image_dataset[2]
                np_image = np.array(image)
                plt.imshow(np_image)"""

                plt.subplot(1, 2, 1)
                node_saliency_map = saliency_map(model.input.grad)

                # construct networkx graph
                x, edge_index = data[id].x, data[id].edge_index
                df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
                G = nx.from_pandas_edgelist(df, 'from', 'to')    
                
                # flip over the axis of pos, this is because the default axis direction of networkx is different
                pos = {i: np.array([data[id].pos[i][0], 27 - data[id].pos[i][1]]) for i in range(data[id].num_nodes)}
                
                # get the current node index of G
                idx = list(G.nodes())

                # set the node sizes using node features
                size = x[idx] * 500 + 200
                
                # set the node colors using node features
                color = []
                for i in idx:
                    grey = x[i]
                    if grey == 0:
                        color.append('skyblue')
                    else:
                        color.append('red')
                scaled_saliency_map_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(node_saliency_map).reshape(-1, 1)).reshape(-1, )
                # print(data.batch)
                # print("data", data)
                # print("scaled ", scaled_saliency_map_weights.shape)
                idx = (data.batch == id).nonzero().flatten()
                # print(idx)
                
                # with open(os.path.join(RAW_DATA_PATH, f'{data[2].img_id}_{data[2].p_id}_coordinates.pickle'), 'rb') as handle:
                #     coordinates_arr = pickle.load(handle)
                # print(data[1])
                # print(coordinates_arr.shape)
                g = utils.to_networkx(data[id], to_undirected=True)
                nx.draw(g, pos=pos, node_color=scaled_saliency_map_weights[idx[0]:idx[-1]+1], node_size=80, cmap=plt.cm.afmhot)
                # plt.savefig(f"saliency_deneme_mnist_{random.randint(0,10)}_{data[id].y}.png")
                # plt.clf()
                plt.subplot(1, 2, 2)
                # print("++++++++++++++++++++++++GRAD CAM++++++++++++++++++++++++")
                # print(model.final_conv_acts.shape)
                final_conv_acts = model.final_conv_acts[idx[0]:idx[-1]+1]#.view(40, 512)
                final_conv_grads = model.final_conv_grads[idx[0]:idx[-1]+1]#.view(40, 512)
                grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)# [:mol.GetNumAtoms()]
                
                scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )
                # print("GRAD CAM WEIG", scaled_grad_cam_weights.shape)
                g = utils.to_networkx(data[id], to_undirected=True)
                nx.draw(g, pos=pos, node_color=scaled_grad_cam_weights, node_size=80, cmap=plt.cm.afmhot)
                plt.savefig(f"gradcam_deneme_mnist_{random.randint(0,10)}_{data[id].y}.png")
                plt.clf()
                # print("++++++++++++++++++++++++")
        
        """out_list.extend([val.item() for val in out.squeeze()])
        
        pred_list.extend([val.item() for val in data.y])"""

        total_loss += float(loss.item())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients
        # print("After", model.input.grad)

    return total_loss

def test(loader, label=None, fl_name=None, plot_pred=False):
    model.eval()

    total_loss = 0.0
    correct= 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        data.y =  data.y.to(device)
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
        # print("out", out)
        pred = out.data.max(1, keepdim=True)[1]
        # print("target", data.y)
        # print("pred", pred)
        correct += pred.eq(data.y.view_as(pred)).sum().to(device)
        
        # print(pred.eq(data.y.view_as(pred)).sum())
        total_loss += float(loss.item())
    
    acc = 100. * correct / len(loader.dataset)

    return total_loss, acc


best_val_loss = np.inf
best_train_loss = np.inf
best_test_loss = np.inf
best_train_acc, best_validation_acc, best_test_acc = -1.0, -1.0, -1.0

print_at_each_epoch = True
for epoch in range(1, args.epoch):
    
    train()
    
    train_loss, validation_loss, test_loss = np.inf, np.inf, np.inf
    train_acc, val_acc, test_acc = -1.0, -1.0, -1.0
    plot_at_last_epoch = True
    if epoch== args.epoch-1:

        train_loss, train_acc = test(train_loader, "train", "train", plot_at_last_epoch)
        validation_loss, val_acc= test(validation_loader, "validation", "validation", plot_at_last_epoch)
        test_loss, test_acc = test(test_loader, "test", "test", plot_at_last_epoch)

    else:
        train_loss, train_acc = test(train_loader)
        validation_loss, val_acc= test(validation_loader)
        test_loss, test_acc = test(test_loader)
    
    """for param_group in optimizer.param_groups:
        print(param_group['lr'])"""
    
    # scheduler.step(validation_loss)

    """writer.add_scalar("training/loss", train_loss, epoch)
    writer.add_scalar("validation/loss", validation_loss, epoch)
    writer.add_scalar("test/loss", test_loss, epoch)"""

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_train_loss = train_loss
        best_test_loss = test_loss
        best_train_acc = train_acc
        best_val_acc = val_acc
        best_test_acc = test_acc
        input_grads = model.input.grad
        torch.save(model.state_dict(), f"../models/{args_str}.pth")
        # print("GRADS:", input_grads)
    
    if print_at_each_epoch:

        

        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}, Test loss: {test_loss:.4f}, Train acc: {train_acc:.3f}, Validation acc: {val_acc:.3f}, Test acc: {test_acc:.3f}')

print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}, Best val acc: {best_val_acc}, Best test acc: {best_test_acc}")
# writer.close()

model.load_state_dict(torch.load(f"../models/{args_str}.pth"))
train(model)

"""
plt.subplot(1, 3, 1)
plt.title("MNIST")
image, label = image_dataset[2]
np_image = np.array(image)
plt.imshow(np_image)"""

"""plt.subplot(1, 2, 1)
node_saliency_map = saliency_map(model.input.grad)

# construct networkx graph
x, edge_index = data[2].x, data[2].edge_index
df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
G = nx.from_pandas_edgelist(df, 'from', 'to')    

# flip over the axis of pos, this is because the default axis direction of networkx is different
pos = {i: np.array([data[2].pos[i][0], 27 - data[2].pos[i][1]]) for i in range(data[2].num_nodes)}

# get the current node index of G
idx = list(G.nodes())

# set the node sizes using node features
size = x[idx] * 500 + 200

# set the node colors using node features
color = []
for i in idx:
    grey = x[i]
    if grey == 0:
        color.append('skyblue')
    else:
        color.append('red')
scaled_saliency_map_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(node_saliency_map).reshape(-1, 1)).reshape(-1, )
# print(data.batch)
# print("data", data)
# print("scaled ", scaled_saliency_map_weights.shape)
idx = (data.batch == 2).nonzero().flatten()
# print(idx)

# with open(os.path.join(RAW_DATA_PATH, f'{data[2].img_id}_{data[2].p_id}_coordinates.pickle'), 'rb') as handle:
#     coordinates_arr = pickle.load(handle)
# print(data[1])
# print(coordinates_arr.shape)
g = utils.to_networkx(data[2], to_undirected=True)
nx.draw(g, pos=pos, node_color=scaled_saliency_map_weights[idx[0]:idx[-1]+1], node_size=80, cmap=plt.cm.afmhot)
# plt.savefig(f"saliency_deneme_mnist_{data[2].y}.png")
# plt.clf()
plt.subplot(1, 2, 2)
# print("++++++++++++++++++++++++GRAD CAM++++++++++++++++++++++++")
# print(model.final_conv_acts.shape)
final_conv_acts = model.final_conv_acts[idx[0]:idx[-1]+1]#.view(40, 512)
final_conv_grads = model.final_conv_grads[idx[0]:idx[-1]+1]#.view(40, 512)
grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)# [:mol.GetNumAtoms()]

scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )
# print("GRAD CAM WEIG", scaled_grad_cam_weights.shape)
g = utils.to_networkx(data[2], to_undirected=True)
nx.draw(g, pos=pos, node_color=scaled_grad_cam_weights, node_size=80, cmap=plt.cm.afmhot)
plt.savefig(f"gradcam_deneme_mnist_{data[2].y}.png")
plt.clf()
"""