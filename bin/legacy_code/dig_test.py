from dig.xgraph.dataset import MoleculeDataset
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import files_exist
import os.path as osp
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def split_dataset(dataset, dataset_split=[0.8, 0.1, 0.1]):
    dataset_len = len(dataset)
    dataset_split = [int(dataset_len * dataset_split[0]),
                     int(dataset_len * dataset_split[1]),
                     0]
    dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
    train_set, val_set, test_set = \
        random_split(dataset, dataset_split)

    return {'train': train_set, 'val': val_set, 'test': test_set}

dataset = MoleculeDataset('datasets', 'clintox')
dataset.data.x = dataset.data.x.to(torch.float32)
dataset.data.y = dataset.data.y[:, 0] 
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_targets = dataset.num_classes
num_classes = 2

splitted_dataset = split_dataset(dataset)
dataloader = DataLoader(splitted_dataset['test'], batch_size=1, shuffle=False)


from dig.xgraph.models import GIN_3l

def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)

model = GIN_3l(model_level='graph', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
model.to(device)
check_checkpoints()
ckpt_path = osp.join('checkpoints', 'clintox', 'GIN_3l', '0', 'GIN_3l_best.ckpt')
model.load_state_dict(torch.load(ckpt_path)['state_dict'])

data = list(dataloader)[0].to(device)
out = model(data.x, data.edge_index)
print(out)

from dig.xgraph.method import GradCAM
explainer = GradCAM(model, explain_graph=True)


# --- Set the Sparsity to 0.5 ---
sparsity = 0.5

# --- Create data collector and explanation processor ---
from dig.xgraph.evaluation import XCollector, ExplanationProcessor
x_collector = XCollector(sparsity)
# x_processor = ExplanationProcessor(model=model, device=device)


for index, data in enumerate(dataloader):
    print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
    data.to(device)

    if torch.isnan(data.y[0].squeeze()):
        continue

    walks, masks, related_preds = \
        explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)

    x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())

    # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
    # obtain the result: x_processor(data, masks, x_collector)

    if index >= 99:
        break
    


