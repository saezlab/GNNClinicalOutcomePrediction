import os
import torch
import json
import pickle
import secrets
from random import random
from distutils.log import error
from model_dgermen import CustomGCN
from sklearn.model_selection import KFold


# K-Fold cross validation index creator function
# Dataset idices and ratios must be supplied
# Return triplet of samplers for amount of wanted fold
def k_fold_TTV(dataset,T2VT_ratio,V2T_ratio):
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

def load_model(fileName: str, path =  os.curdir, model_type: str = "NONE", args: dict = {}):
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

    path_model = path + fileName + ".mdl"
    path_hyp = path + fileName + ".hyp"

    mode = fileName.rsplit("_",1)[-1]

    if mode == "SD":
        if model_type == "NONE":
            raise ValueError("For SD mode, model_type must be properly defined!")
        if args == {}:
            raise ValueError("For SD mode, args must be supplied properly!")

        model = CustomGCN(
                    type = model_type,
                    num_node_features = args.num_node_features, 
                    num_gcn_layers=args.num_gcn_layers, 
                    num_ff_layers=args.num_ff_layers, 
                    gcn_hidden_neurons=args.gcn_hidden_neurons, 
                    ff_hidden_neurons=args.ff_hidden_neurons, 
                    dropout=args.dropout,
                    aggregators=args.aggregators,
                    scalers=args.scalers,
                    deg = args.deg 
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
    
    with open(file_path, 'r') as fp:
        l_dict = json.load(fp)
    return l_dict




def generate_session_id():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16)  