# import captum
from email.mime import base
import os
from dataset import TissueDataset
import bin.custom_tools as custom_tools
from captum.attr import *
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

device = custom_tools.get_device()

# Index of the to be inspected graph
idx = 10

cwd = os.getcwd()

dataset = TissueDataset(os.path.join(cwd, "data"))

inspected_graph = dataset[idx]

# Loading the trained model

model = customTools.load_model(fileName="TheModel_EM", path = (os.getcwd() + "/data/models/"))

# Loading model to saliency

# saliency = Saliency(model)

# Giving input to saliency

# attribution = saliency.attribute(inspected_graph.x.to(device))

# https://colab.research.google.com/drive/1fLJbFPz0yMCQg81DdCP5I8jXw9LoggKO?usp=sharing#scrollTo=Wz6B1NgorzAX

def model_forward(edge_mask, data):
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    out = model(data.x, data.edge_index, batch, edge_mask)
    return out


def explain(method, data):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    input_mask = input_mask[:,None] # This is done so that PNA's linear layer can work correctly!
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask,
                                  additional_forward_args=(data,))

    elif method == "dls":
        dls = DeepLiftShap(model_forward, multiply_by_inputs=True)
        mask = dls.attribute(input_mask,
                                  additional_forward_args=(data,),
                                  baselines=data)
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask

def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict
    
def draw_graph(g, edge_mask=None, draw_edge_labels=False):
    g = to_networkx(g)
    g = g.copy().to_undirected()
    # node_labels = {}
    pos = nx.planar_layout(g)
    # pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]

    edge_color = np.vstack(edge_color)
    widths = np.vstack(widths)
    
    nx.draw(g, pos=pos, width=widths,
            edge_color='black', edge_cmap=plt.cm.Blues,
            node_color='azure')
    
    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}    
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                    font_color='red')
    plt.show()


for title, method in [('Deep Lift Shap','dls'),('Saliency', 'saliency'), ('Integrated Gradients', 'ig')]:
    edge_mask = explain(method, inspected_graph)
    edge_mask_dict = aggregate_edge_directions(edge_mask, inspected_graph)
    draw_graph(inspected_graph, edge_mask_dict)
    pass
    

# Integrating the attribution to input


# Visualizing

print("done!")

# Get sub graphs

# Look for differently expressed protein abundance

