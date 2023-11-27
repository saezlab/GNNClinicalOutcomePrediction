#https://gitcode.net/mirrors/rusty1s/pytorch_geometric/-/blob/0.3.1/examples/mnist_graclus.py
import matplotlib.pyplot as plt
import networkx as nx
import os.path as osp
from explain import saliency_map, grad_cam
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x,
                                global_mean_pool, GCNConv)
import random
from torch_geometric import utils
from sklearn.preprocessing import MinMaxScaler
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
train_dataset = MNISTSuperpixels(path, True, transform=T.Cartesian())
test_dataset = MNISTSuperpixels(path, False, transform=T.Cartesian())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
d = train_dataset.data
print(train_dataset)


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(d.num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 10) # d.num_classes)
        self.conv2_grads  = None
        self.conv2_acts = None

    
    def activations_hook(self, grad):
        self.conv2_grads = grad

    def forward(self, data):
        data.x.requires_grad = True
        self.input = data.x.requires_grad_()
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        
        data.x = self.conv2(data.x, data.edge_index, data.edge_attr)
        self.conv2_acts = data.x
        data.x.register_hook(self.activations_hook)

        data.x = F.elu(data.x)
        
        
        # print(self.conv2_grads)
        
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch, input_model=False):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data), data.y).backward()
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
                df = pd.DataFrame({'from': edge_index[0].cpu(), 'to': edge_index[1].cpu()})
                G = nx.from_pandas_edgelist(df, 'from', 'to')    
                
                # flip over the axis of pos, this is because the default axis direction of networkx is different
                pos = {i: np.array([data[id].pos[i][0].cpu(), 27 - data[id].pos[i][1].cpu()]) for i in range(data[id].num_nodes)}
                
                # get the current node index of G
                idx = list(G.nodes())

                # set the node sizes using node features
                size = x[idx] * 500 + 200
                
                # set the node colors using node features
                color = []
                """for i in idx:
                    grey = x[i]
                    if grey == 0:
                        color.append('skyblue')
                    else:
                        color.append('red')"""
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
                final_conv_acts = model.conv2_acts[idx[0]:idx[-1]+1]#.view(40, 512)
                final_conv_grads = model.conv2_grads[idx[0]:idx[-1]+1]#.view(40, 512)
                grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)# [:mol.GetNumAtoms()]
                
                scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )
                # print("GRAD CAM WEIG", scaled_grad_cam_weights.shape)
                g = utils.to_networkx(data[id], to_undirected=True)
                nx.draw(g, pos=pos, node_color=scaled_grad_cam_weights, node_size=80, cmap=plt.cm.afmhot)
                plt.savefig(f"gradcam_deneme_mnist_{random.randint(0,10)}_{data[id].y}.png")
                plt.clf()
                # print("++++++++++++++++++++++++")
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)


for epoch in range(1, 31):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))



train(epoch, True)