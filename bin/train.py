import torch
from model import GCN, GCN2
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

parser = argparse.ArgumentParser(description='GNN Arguments')

parser.add_argument(
    '--model',
    type=str,
    default="GCN",
    metavar='mn',
    help='model name (default: GCN)')

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
    default=0.25,
    metavar='DO',
    help='dropout rate (default: 0.25)')

parser.add_argument(
    '--epoch',
    type=int,
    default=200,
    metavar='EPC',
    help='Number of epochs (default: 200)')

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


args = parser.parse_args()


use_gpu = torch.cuda.is_available()

device = "cpu"

if use_gpu:
    print("GPU is available on this device!")
    device = "cuda"
else:
    print("CPU is available on this device!")

print(device)

writer = SummaryWriter(log_dir="../logs")
dataset = TissueDataset("../data")
print(dataset.raw_file_names)
print(len(dataset))

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:170]
validation_dataset = dataset[170:205]
test_dataset = dataset[205:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of validation graphs: {len(validation_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f"Number of node features: {dataset.num_node_features}")

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)


for step, data in enumerate(test_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')


model = None

# model = GCN(dataset.num_node_features, hidden_channels=256).to(device)
model =GCN2(dataset.num_node_features, hidden_channels=args.gcn_h, fcl1=args.fcl, drop_rate=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)


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

for epoch in range(1, args.epoch):
    
    train()
    
    train_loss, validation_loss, test_loss = np.inf, np.inf, np.inf
    
    if epoch== args.epoch-1:
        train_loss = test(train_loader, "train", True)
        validation_loss= test(validation_loader, "validation", True)
        test_loss = test(test_loader, "test", True)
    else:
        train_loss = test(train_loader)
        validation_loss= test(validation_loader)
        test_loss = test(test_loader)
    
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    
    scheduler.step(validation_loss)

    writer.add_scalar("training/loss", train_loss, epoch)
    writer.add_scalar("validation/loss", validation_loss, epoch)
    writer.add_scalar("test/loss", test_loss, epoch)

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_train_loss = train_loss
        best_test_loss = test_loss

    print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}, Test loss: {test_loss:.4f}')

print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")
writer.close()