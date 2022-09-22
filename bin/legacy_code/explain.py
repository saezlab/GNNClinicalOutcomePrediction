import torch
import torch.nn.functional as F

def saliency_map(input_grads):

    # print(input_grads.shape[0], input_grads.shape)
    # print('saliency_map')
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        # print(node_grads.shape)
        node_saliency = torch.norm(F.relu(node_grads)).item()
        # print(node_saliency)
        node_saliency_map.append(node_saliency)
    return node_saliency_map


def grad_cam(final_conv_acts, final_conv_grads):
    # print('grad_cam')
    node_heat_map = []
    # print(final_conv_grads.shape)
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    # print(alphas.shape)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    # print(node_heat_map)
    return node_heat_map


"""def ugrad_cam(mol, final_conv_acts, final_conv_grads):
    # print('new_grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)

    node_heat_map = np.array(node_heat_map[:mol.GetNumAtoms()]).reshape(-1, 1)
    pos_node_heat_map = MinMaxScaler(feature_range=(0,1)).fit_transform(node_heat_map*(node_heat_map >= 0)).reshape(-1,)
    neg_node_heat_map = MinMaxScaler(feature_range=(-1,0)).fit_transform(node_heat_map*(node_heat_map < 0)).reshape(-1,)
    return pos_node_heat_map + neg_node_heat_map"""


