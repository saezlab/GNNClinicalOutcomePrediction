# import captum
import os
from dataset import TissueDataset
import bin.custom_tools as custom_tools
from captum.attr import *
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

device = custom_tools.get_device()

# Index of the to be inspected graph
idx = 10

cwd = os.getcwd()

dataset = TissueDataset(os.path.join(cwd, "data"))

inspected_graph = dataset[idx]

# Loading the trained model

model = custom_tools.load_model(fileName="TheModel_EM", path = (os.getcwd() + "/data/models/"))

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
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask,
                                  additional_forward_args=(data,))
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
    



for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
    edge_mask = explain(method, inspected_graph)
    edge_mask_dict = aggregate_edge_directions(edge_mask, inspected_graph)
    

# Integrating the attribution to input


# Visualizing

print("done!")

# Get sub graphs

# Look for differently expressed protein abundance

