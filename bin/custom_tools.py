import os
import torch
import json
import pickle
import secrets
from random import random
from distutils.log import error
from model import CustomGCN
from sklearn.model_selection import KFold


# K-Fold cross validation index creator function
# Dataset idices and ratios must be supplied
# Return triplet of samplers for amount of wanted fold
def k_fold_ttv(dataset,T2VT_ratio,V2T_ratio):
    """Splits dataset into Train, Validation and Test sets

    Args:
        dataset (_type_): Data to be splitted
        T2VT_ratio (int): Train / (Valid + Test)
        V2T_ratio (int): Valid / Test
    """

    fold_T2VT=KFold(n_splits=T2VT_ratio+1)
    fold_V2T =KFold(n_splits=V2T_ratio+1)

    # List to save sampler triplet
    samplers = []

    # Fold count init
    fold_count = 0

    # Pulling indexes for sets
    for (train_idx, valid_test_idx) in fold_T2VT.split(dataset):
        for (valid_idx, test_idx) in fold_V2T.split(valid_test_idx):

            fold_count += 1

            valid_idx = valid_test_idx[valid_idx]
            test_idx = valid_test_idx[test_idx]


            samplers.append((
                (fold_count),
                (torch.utils.data.SubsetRandomSampler(train_idx)),
                (torch.utils.data.SubsetRandomSampler(test_idx)),
                (torch.utils.data.SubsetRandomSampler(valid_idx))))
                
    return samplers



def save_model(model: CustomGCN,fileName ,mode: str, path = os.path.join(os.curdir, "..", "models")):
    """
    Saves the pytoch model, has 3 modes.
    Models have extension ".mdl", 
    Mode SD: Just save the state dict., user needs to hold hyperparameters
    Mode SDH: State dict and hyperparameters are saved, in DIFFERENT files
    Mode EM: Entire model is saved
    For now only works with "model_dgermen", NOT generic

    Args:
        model (CustomGCN): Model to be saved
        fileName (str): Model's saved file name.
        mode (str): Mode to save the model
        path (_type_, optional): path that model will be saved to. Defaults to os.curdir().
    """
    path_model = os.path.join(path, fileName + "_" + mode + ".mdl") 
    path_hyp = os.path.join(path + fileName + "_" + mode  + ".hyp")

    if mode == "SD":
        torch.save(model.state_dict(), path_model)

    elif mode == "SDH":
        torch.save(model.state_dict(), path_model)
        
        try: 
            model_hyp = {
                "type" : model.model_type,
                "args" : model.pars
            }

        except KeyError:
            raise KeyError("Model doesn't have type attribute, add manually or use updated 'model_dgermen'.")

        pickle_out = open(path_hyp,"wb")
        pickle.dump(model_hyp, pickle_out)
        pickle_out.close()

    elif mode == "EM":
        torch.save(model, path_model)

    print(f"Model saved with session id: {fileName}!")

def load_model(fileName: str, path =  os.curdir, model_type: str = "NONE", args: dict = {}, deg = None):
    """Models the specified model and returns it

    Args:
        fileName (str): Name of the file without the extension
        path (str, optional): Path to the directory of the saved model. Defaults to os.curdir.
        model_type (str, optional): If saved with SD mode, need to supply model type. Defaults to "NONE".
        args (dict, optional): If saved with SD mode, need to supply model creation hyperparameter. Defaults to {}.

    Raises:
        ValueError: 
        ValueError: 

    Returns:
        CustomGCN: Ready model
    """

    path_model = os.path.join(path, fileName + ".mdl")
    path_hyp = os.path.join(path, fileName + ".hyp")

    mode = fileName.rsplit("_",1)[-1]

    if mode == "SD":
        if model_type == "NONE":
            raise ValueError("For SD mode, model_type must be properly defined!")
        if args == {}:
            raise ValueError("For SD mode, args must be supplied properly!")

        model = CustomGCN(
                    type = args["model"],
                    num_node_features = args["num_node_features"], 
                    num_gcn_layers=args["num_of_gcn_layers"], 
                    num_ff_layers=args["num_of_ff_layers"], 
                    gcn_hidden_neurons=args["gcn_h"], 
                    ff_hidden_neurons=args["fcl"], 
                    dropout=args["dropout"],
                    aggregators=args["aggregators"],
                    scalers=args["scalers"],
                    deg = deg
                        )
        
        model.load_state_dict(torch.load(path_model))
        model.eval()

        return model

    elif mode == "SDH":

        pickle_in = open(path_hyp,"rb")
        hyp = pickle.load(pickle_in)
        pickle_in.close()

        model_type = hyp["type"]
        args = hyp["args"]

        model = CustomGCN(
                    type = model_type,
                    num_node_features = args["num_node_features"], 
                    num_gcn_layers=args["num_gcn_layers"], 
                    num_ff_layers=args["num_ff_layers"], 
                    gcn_hidden_neurons=args["gcn_hidden_neurons"], 
                    ff_hidden_neurons=args["ff_hidden_neurons"], 
                    dropout=args["dropout"],
                    aggregators=args["aggregators"],
                    scalers=args["scalers"],
                    deg = args["deg"] 
                        )

        model.load_state_dict(torch.load(path_model))
        model.eval()

        return model

    
    elif mode == "EM":

        model = torch.load(path_model)
        model.eval()

        return model



def get_device():
    """Returns the available device, default priority is "cuda"

    Returns:
        string: cuda or cpu
    """

    use_gpu = torch.cuda.is_available()

    device = "cpu"

    if use_gpu:
        device = "cuda"
        print("GPU is available on this device!")
    else:
        print("CPU is available on this device!")

    return device

def save_dict_as_json(s_dict, file_name, path):
    """
    Save dictionary as a json file
    """

    with open(os.path.join(path,file_name+".json"), "w") as write_file:
        json.dump(s_dict, write_file, indent=4)

def load_json(file_path):
    """Loads the json file for given path

    Args:
        file_path (string): file path

    Returns:
        dict: dict of the json
    """
    
    with open(file_path, 'r') as fp:
        l_dict = json.load(fp)
    return l_dict

def save_pickle(obj, file_name: str, path =  os.curdir):

    pickle_out = open(os.path.join(path, file_name),"wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def load_pickle(path_obj):
    pickle_in = open(path_obj,"rb")
    obj = pickle.load(pickle_in)
    pickle_in.close()
    return obj


def generate_session_id():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16)  

import argparse

def general_parser() -> argparse.Namespace:
    """Used inside a file, makes the file able to parse CLI arguments.

    Returns:
        argparse.Namespace: argparse namespace that includes supplied argument values
    """

    parser = argparse.ArgumentParser(description='GNN Arguments')
    parser.add_argument(
        '--model',
        type=str,
        default="PNAConv",
        metavar='mn',
        help='model name (default: PNAConv)')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
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
        default=0.20, # 0.1 0.2 0.3 0.5
        metavar='DO',
        help='dropout rate (default: 0.20)')

    parser.add_argument(
        '--epoch',
        type=int,
        default=5,
        metavar='EPC',
        help='Number of epochs (default: 50)')

    parser.add_argument(
        '--num_of_gcn_layers',
        type=int,
        default=3,
        metavar='NGCNL',
        help='Number of GCN layers (default: 2)')

    parser.add_argument(
        '--num_of_ff_layers',
        type=int,
        default=3,
        metavar='NFFL',
        help='Number of FF layers (default: 2)')
        
    parser.add_argument(
        '--gcn_h',
        type=int,
        default=128,
        metavar='GCNH',
        help='GCN hidden channel (default: 128)')

    parser.add_argument(
        '--fcl',
        type=int,
        default=128,
        metavar='FCL',
        help='Number of neurons in fully-connected layer (default: 128)')

    parser.add_argument(
        '--en',
        type=str,
        default="my_experiment",
        metavar='EN',
        help='the name of the experiment (default: my_experiment)')

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.001,
        metavar='WD',
        help='weight decay (default: 0.001)')

    parser.add_argument(
        '--factor', # 0.5 0.8, 0.2
        type=float,
        default=0.5,
        metavar='FACTOR',
        help='learning rate reduce factor (default: 0.5)')

    parser.add_argument(
        '--patience', # 5, 10, 20
        type=int,
        default=5,
        metavar='PA',
        help='patience for learning rate scheduling (default: 5)')

    parser.add_argument(
        '--min_lr',
        type=float,
        default=0.00002,#0.0001
        metavar='MLR',
        help='minimum learning rate (default: 0.00002)')

    parser.add_argument(
        '--aggregators',
        # WARN This use of "+" doesn't go well with positional arguments
        nargs='+',
        # PROBLEM How to feed a list in CLI? Need to edit generator?
        # Take string split
        type = str,
        # ARBTR Change to something meaningful
        default= ["sum","mean"], # "sum", "mean", "min", "max", "var" and "std".
        metavar='AGR',
        help= "aggregator list for PNAConv"
    )

    parser.add_argument(
        '--scalers',
        # WARN This use of "+" doesn't go well with positional arguments
        nargs='+',
        # PROBLEM How to feed a list in CLI? Need to edit generator?
        type= str,
        default= ["amplification","identity"], # "identity", "amplification", "attenuation", "linear" and "inverse_linear"
        metavar='SCL',
        help='Set of scaling function identifiers,')

    


    args = parser.parse_args()

    # Handling string inputs
    if type(args.aggregators) != list:
        args.aggregators = args.aggregators.split()

    if type(args.scalers) != list:
        args.scalers = args.scalers.split()

    # This can be used to print all parser arguments
    parser_args_list = sorted(vars(args).keys())
    args.str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in parser_args_list])

    return args

