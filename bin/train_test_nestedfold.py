
# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

import torch
from data_processing import OUT_DATA_PATH
from model_dgermen import CustomGCN
from dataset import TissueDataset
from torch_geometric.loader import DataLoader
from torch.nn import BatchNorm1d
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotting
import pandas as pd
import pickle
import os
import pytorch_lightning as pl
from torch_geometric import utils
from explain import saliency_map
import networkx as nx
from torch_geometric.utils import degree
from evaluation_metrics import r_squared_score
import bin.custom_tools as custom_tools
import csv
import statistics

# region PATH SETTINGS
# ---------------------------------------------------------------------------- #
#                                 PATH SETTINGS                                #
# ---------------------------------------------------------------------------- #

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
#OUT_DATA_PATH = os.path.join(S_PATH, "../../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")
S_PATH = os.path.dirname(__file__)
# endregion


# region PARSER SETTINGS
parser = argparse.ArgumentParser(description='GNN Arguments')

parser.add_argument(
    '--id',
    type=int,
    default=99999,
    metavar='id',
    help='model id (default: 99999)')

parser.add_argument(
    '--model',
    type=str,
    default="TransformerConv",
    metavar='mn',
    help='model name (default: PNAConv)')

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
    default=2,
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

parser.add_argument(
    '--aggregators',
    # WARN This use of "+" doesn't go well with positional arguments
    nargs='+',
    # PROBLEM How to feed a list in CLI? Need to edit generator?
    # Take string split
    type = str,
    # ARBTR Change to something meaningful
    default= ["sum","mean"], # "sum", "mean", "min", "max", "var" and "std".
    metavar='AGR',
    help= "aggregator list for PNAConv"
)

parser.add_argument(
    '--scalers',
    # WARN This use of "+" doesn't go well with positional arguments
    nargs='+',
    # PROBLEM How to feed a list in CLI? Need to edit generator?
    type= str,
    default= ["amplification","identity"], # "identity", "amplification", "attenuation", "linear" and "inverse_linear"
    metavar='SCL',
    help='Set of scaling function identifiers,')
# endregion

pl.seed_everything(42)
args = parser.parse_args()

# region CPU -> GPU Switch
use_gpu = torch.cuda.is_available()

device = "cpu"

if use_gpu:
    print("GPU is available on this device!")
    device = "cuda"
else:
    print("CPU is available on this device!")
# endregion






def train():
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
        out_list.extend([val.item() for val in out.squeeze()])
        
        pred_list.extend([val.item() for val in data.y])

        total_loss += float(loss.item())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients
        # print("After", model.input.grad)

    return total_loss

def test(loader, label=None, fl_name=None, plot_pred=False):
    model.eval()

    total_loss = 0.0
    pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list = [], [], [], [], [], [], []

    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
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



# writer = SummaryWriter(log_dir=os.path.join(S_PATH,"../logs"))
dataset = TissueDataset(os.path.join(S_PATH,"../data"))

dataset = dataset.shuffle()

samplers = custom_tools.k_fold_TTV(dataset, T2VT_ratio=4, V2T_ratio=1)

results = []


for fold, train_sampler, validation_sampler, test_sampler in samplers:

    train_loader = DataLoader(dataset, batch_size=args.bs, sampler= train_sampler)
    validation_loader = DataLoader(dataset, batch_size=args.bs, sampler= validation_sampler)
    test_loader = DataLoader(dataset, batch_size=args.bs, sampler= test_sampler)

    train_dataset = dataset[train_sampler.indices]

    # region Setting up the model
    # Handling string inputs
    if type(args.aggregators) != list:
        args.aggregators = args.aggregators.split()

    if type(args.scalers) != list:
        args.scalers = args.scalers.split()

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
                    num_node_features = dataset.num_node_features, ####### LOOOOOOOOK HEREEEEEEEEE
                    num_gcn_layers=args.num_of_gcn_layers, 
                    num_ff_layers=args.num_of_ff_layers, 
                    gcn_hidden_neurons=args.gcn_h, 
                    ff_hidden_neurons=args.fcl, 
                    dropout=args.dropout,
                    aggregators=args.aggregators,
                    scalers=args.scalers,
                    deg = deg # Comes from data not hyperparameter
                        ).to(device)
    # endregion

    # region Training Hyperparameter Settings
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()


    args_list = sorted(vars(args).keys())
    args_str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in args_list])
    print(args_str)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor= args.factor, patience=args.patience, min_lr=args.min_lr, verbose=True)
    # endregion

    best_val_loss = np.inf
    best_train_loss = np.inf
    best_test_loss = np.inf

    print_at_each_epoch = True

    # Epoch loop
    for epoch in range(1, args.epoch):

        train()
        
        train_loss, validation_loss, test_loss = np.inf, np.inf, np.inf
        plot_at_last_epoch = True
        if epoch== args.epoch-1:

            train_loss, df_train = test(train_loader, "train", "train", plot_at_last_epoch)
            validation_loss, df_val= test(validation_loader, "validation", "validation", plot_at_last_epoch)
            test_loss, df_test = test(test_loader, "test", "test", plot_at_last_epoch)
            list_ct = list(set(df_train["Clinical Type"]))
            r2_score = r_squared_score(df_val['OS Month (log)'], df_val['Predicted'])

            if r2_score>0.7:

                df2 = pd.concat([df_train, df_val, df_test])
                df2.to_csv(f"{OUT_DATA_PATH}/{args_str}.csv", index=False)
                # print(list_ct)
                # plotting.plot_pred_vs_real_lst(df2, ['OS Month (log)']*3, ["Predicted"]*3, "Clinical Type", list_ct, args_str)
                plotting.plot_pred_(df2, list_ct, args_str)


        else:
            train_loss = test(train_loader)
            validation_loss= test(validation_loader)
            test_loss = test(test_loader)
        
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
        
        if print_at_each_epoch:
            print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}, Test loss: {test_loss:.4f}')

    print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")


    results.append([fold, best_train_loss, best_val_loss, best_test_loss])

header = ["fold_number","train","validation","test"]

train_results = []
valid_results = []
test_results = []

for _,train,valid,test in results:
    train_results.append(train)
    valid_results.append(valid)
    test_results.append(test)

means = [["Means",statistics.mean(train_results),statistics.mean(valid_results),statistics.mean(test_results)]]
variances = [["Variances",statistics.variance(train_results),statistics.variance(valid_results),statistics.variance(test_results)]]

with open('results/idedFiles/' + str(args.id) + '.csv', 'w', encoding="UTF8", newline='') as f:
    writer = csv.writer(f)

    writer.writerow(header)

    writer.writerows(results)

    writer.writerows(means)

    writer.writerows(variances)


    # writer.close()
