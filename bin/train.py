import torch
from model import GCN, GCN2, GCN_NEW
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
import os

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

args = parser.parse_args()

use_gpu = torch.cuda.is_available()

device = "cpu"

if use_gpu:
    print("GPU is available on this device!")
    device = "cuda"
else:
    print("CPU is available on this device!")


# writer = SummaryWriter(log_dir=os.path.join(S_PATH,"../logs"))
dataset = TissueDataset(os.path.join(S_PATH,"../data"))

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:170]
validation_dataset = dataset[170:205]
test_dataset = dataset[205:]

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

model = GCN_NEW(dataset.num_node_features, 
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
criterion = torch.nn.MSELoss()


args_list = sorted(vars(args).keys())
args_str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in args_list])
print(args_str)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor= args.factor, patience=args.patience, min_lr=args.min_lr, verbose=True)


def train():
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

def test(loader, fl_name=None, plot_pred=False):
    model.eval()

    total_loss = 0.0
    pred_list, out_list, tumor_grade_list, clinical_type_list, osmonth_list = [], [], [], [], []

    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
        total_loss += float(loss.item())

        out_list.extend([val.item() for val in out.squeeze()])
        pred_list.extend([val.item() for val in data.y])
        tumor_grade_list.extend([val for val in data.tumor_grade])
        clinical_type_list.extend([val for val in data.clinical_type])
        osmonth_list.extend([val for val in data.osmonth])
    
    df = pd.DataFrame(list(zip(out_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list)),
               columns =['OS Month (log)', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month"])
    if plot_pred:
        plotting.plot_pred_vs_real(df, 'OS Month (log)', 'Predicted', "Clinical Type", fl_name)


    
    return total_loss


best_val_loss = np.inf
best_train_loss = np.inf
best_test_loss = np.inf

print_at_each_epoch = False
for epoch in range(1, args.epoch):
    
    train()
    
    train_loss, validation_loss, test_loss = np.inf, np.inf, np.inf
    plot_at_last_epoch = False
    if epoch== args.epoch-1:
        train_loss = test(train_loader, "train", plot_at_last_epoch)
        validation_loss= test(validation_loader, "validation", plot_at_last_epoch)
        test_loss = test(test_loader, "test", plot_at_last_epoch)
    else:
        train_loss = test(train_loader)
        validation_loss= test(validation_loader)
        test_loss = test(test_loader)
    
    """for param_group in optimizer.param_groups:
        print(param_group['lr'])"""
    
    scheduler.step(validation_loss)

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
# writer.close()