import os.path as osp
import numpy as np
import pandas as pd
from PIL import ImageColor
import seaborn as sns
import networkx as nx
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric import utils

# https://github.com/pyg-team/pytorch_geometric/blob/83db552aba6f8cc53672189d424daa6917a25510/examples/mnist_visualization.py

def visualize(image, data, example):
    plt.figure(figsize=(17, 8))
    
    # plot the mnist image
    plt.subplot(1, 3, 1)
    plt.title("MNIST")
    np_image = np.array(image)
    plt.imshow(np_image)
    
    # plot the super-pixel graph
    plt.subplot(1, 3, 2)
    x, edge_index = data.x, data.edge_index
    
    # construct networkx graph
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')    
    
    # flip over the axis of pos, this is because the default axis direction of networkx is different
    pos = {i: np.array([data.pos[i][0], 27 - data.pos[i][1]]) for i in range(data.num_nodes)}
    
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
    
    nx.draw(G, with_labels=True, node_color=color, pos=pos)
    plt.title("MNIST Superpixel")
    plt.subplot(1, 3, 3)
    g = utils.to_networkx(data, to_undirected=True)
    nx.draw(g)
    plt.savefig(f"mnist_vis_{example}.png")

if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
    image_dataset = MNIST(root=path, download=True)
    graph_dataset = MNISTSuperpixels(root=path)
    print(len(image_dataset))
    print(len(graph_dataset))

    for example in range(1,25):
        image, label = image_dataset[example]
        data = graph_dataset[example]
        print("Image Label:", label)
        visualize(image, data, example)