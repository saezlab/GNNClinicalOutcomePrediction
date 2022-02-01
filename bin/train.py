import torch
from model import GCN, GCN2
from dataset import TissueDataset
from torch_geometric.loader import DataLoader
from torch.nn import BatchNorm1d
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='GNN Arguments')

parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
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
"""


parser.add_argument(
    '--chln',
    type=str,
    default="512_512",
    metavar='CHLN',
    help='number of neurons in compound hidden layers (default: 512_512)')
"""
parser.add_argument(
    '--en',
    type=str,
    default="my_experiments",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')

parser.add_argument(
    '--model',
    type=str,
    default="CompFCNNTarCNNModuleInception",
    metavar='mn',
    help='model name (default: CompFCNNTarCNNModuleInception)')




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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


for step, data in enumerate(test_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')



# model = GCN(dataset.num_node_features, hidden_channels=256).to(device)
model =GCN2(dataset.num_node_features, hidden_channels=256, fcl1=128, drop_rate=0.25).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


def train():
    model.train()
    total_loss = 0.0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
        loss.backward()  # Derive gradients.
        
        total_loss += float(loss.item())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return total_loss

def test(loader):
    model.eval()
    
    correct = 0
    total_loss = 0.0
    
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).type(torch.DoubleTensor).to(device) # Perform a single forward pass.
        loss = criterion(out.squeeze(), data.y.to(device))  # Compute the loss.
        total_loss += float(loss.item())

    return total_loss


best_val_loss = 10000
best_train_loss = 10000
best_test_loss = 10000

for epoch in range(1, 30):
    
    train()

    train_loss = test(train_loader)
    validation_loss= test(validation_loader)
    test_loss = test(test_loader)
    
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