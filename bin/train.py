import torch
from model import GCN
from dataset import TissueDataset
from torch_geometric.loader import DataLoader

dataset = TissueDataset("../data")
print(dataset.raw_file_names)
print(len(dataset))

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:180]
validation_dataset = dataset[180:200]
test_dataset = dataset[200:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of validation graphs: {len(validation_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f"Number of node features: {dataset.num_node_features}")



train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')


    print()


model = GCN(dataset.num_node_features, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    total_loss = 0.0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch).type(torch.DoubleTensor) # Perform a single forward pass.
        # print("Burayı gecti")
        # print(out)
        loss = criterion(out.squeeze(), data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
    
        total_loss += float(loss.item())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    print(total_loss)

def test(loader):
     model.eval()

     correct = 0
     total_loss = 0.0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch).type(torch.DoubleTensor) # Perform a single forward pass.
         
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

"""
for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    # test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')#, Test Acc: {test_acc:.4f}')"""