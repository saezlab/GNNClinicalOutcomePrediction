import torch
import torch.nn.functional as F

def saliency_map(input_grads):

    print(input_grads.shape[0], input_grads.shape)
    # print('saliency_map')
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        # print(node_grads.shape)
        node_saliency = torch.norm(F.relu(node_grads)).item()
        # print(node_saliency)
        node_saliency_map.append(node_saliency)
    return node_saliency_map