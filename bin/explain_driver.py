import joblib
import torch
import sys
import numpy as np
import custom_tools
import os
import plotting
from dataset import TissueDataset
from explainer_cli_new import Custom_Explainer
import pytorch_lightning as pl
import pickle

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "JacksonFischer/raw")


"""
job_id = "sE3-V1wnbN4AgiUbKeP5oA"
args  = custom_tools.load_json(f"../models/best_full_training_22-11-2022/{job_id}.json")
deg = custom_tools.load_pickle(f"../models/best_full_training_22-11-2022/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = "../models/best_full_training_22-11-2022", model_type = "SD", args = args, deg=deg)
"""


job_id = "Hg20Z3phDR_YtQ7cehIVbA"
args  = custom_tools.load_json(f"../models/PNAConv_PNA_MSE_month_30-06-2023/{job_id}.json")
args["num_node_features"] = 33
deg = custom_tools.load_pickle(f"../models/PNAConv_PNA_MSE_month_30-06-2023/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = "../models/PNAConv_PNA_MSE_month_30-06-2023", model_type = "SD", args = args, deg=deg)






dataset = TissueDataset(os.path.join(S_PATH,"../data/JacksonFischer/basel_zurich_preprocessed_compact_dataset.csv"))

dataset = dataset.shuffle()

num_of_train = int(len(dataset)*0.80)
num_of_val = int(len(dataset)*0.10)

train_dataset = dataset[:num_of_train]
validation_dataset = dataset[num_of_train:num_of_train+num_of_val]
test_dataset = dataset[num_of_train+num_of_val:]

test_graph = test_dataset[0]
with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
    coordinates_arr = pickle.load(handle)

# plotting.plot_khop(test_dataset[0], "../plots/subgraphs", f"{test_graph.img_id}_{test_graph.p_id}", coordinates_arr,)


random_seed_list = [42] # , 21, 1, 12, 123, 1234, 2, 23, 234, 2345]
all_edges_list = []
for random_seed in random_seed_list:
    explainer = Custom_Explainer(model, dataset, seed=random_seed)

    pl.seed_everything(random_seed)
    for lr in [0.1]:# , 0.01, 0.001, 0.0001]:
        edge_idx = explainer.explain(epoch=100, lr=lr)
        # all_edges_list.append([val.item() for val in list(edge_idx.cpu())])"""
"""
import pickle
result = np.logical_and.reduce(all_edges_list)
test_graph = test_dataset[0]
with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
    coordinates_arr = pickle.load(handle)

plotting.plot_subgraph(test_graph, "../plots/subgraphs", f"{test_graph.img_id}_{test_graph.p_id}_intersection", coordinates_arr, result )
print(np.unique(result, return_counts=True))
"""