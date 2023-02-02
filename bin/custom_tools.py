import os
import torch
import json
import pickle
import secrets
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from pathlib import Path
from model import CustomGCN
from torch_geometric import utils
from data_preparation import get_basel_zurich_staining_panel
from sklearn.model_selection import KFold

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

# K-Fold cross validation index creator function
# Dataset idices and ratios must be supplied
# Return triplet of samplers for amount of wanted fold
def k_fold_ttv(dataset,T2VT_ratio,V2T_ratio, shuffle_VT = False):
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

            if not shuffle_VT:
                break

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

        model.load_state_dict(torch.load(path_model, map_location=get_device()))
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
        default="test_experiment",
        metavar='EN',
        help='the name of the experiment (default: test_experiment)')

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
        help= "aggregator list for PNAConv")

    parser.add_argument(
        '--scalers',
        # WARN This use of "+" doesn't go well with positional arguments
        nargs='+',
        # PROBLEM How to feed a list in CLI? Need to edit generator?
        type= str,
        default= ["amplification","identity"], # "identity", "amplification", "attenuation", "linear" and "inverse_linear"
        metavar='SCL',
        help='Set of scaling function identifiers,')

    parser.add_argument(
        '--heads',
        type = int,
        default=1,
        metavar="NOH",
        help='number of heads for GATConv'
    )

    parser.add_argument(
        '--fold',
        type= bool,
        action=argparse.BooleanOptionalAction,
        default= True,
        metavar='F',
        help='Perform train/val/test or n-fold x-val (--fold, --no-fold),')

    parser.add_argument(
        '--full_training',
        type= bool,
        action=argparse.BooleanOptionalAction,
        default= False,
        metavar='F',
        help='Perform full_training, default: --no-full_training (--full_training, --no-full_training),')

    parser.add_argument(
        '--label',
        type= str,
        default= "OSMonth", # "identity", "amplification", "attenuation", "linear" and "inverse_linear"
        metavar='LBL',
        help='Which property of the dataset will be used as label')

    


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


def create_directories(lst_path):
    """Create nested directories for the input paths
    
    Args:
        lst_path (list): file paths

    """
    for path in lst_path:
        Path(path).mkdir(parents=True, exist_ok=True)


def extract_LT(list):
    unique_elements = set(list)
    ToIndex = {}
    ToElement = {}
    for i, element in enumerate(unique_elements):
        ToIndex[str(element)] = i
        ToElement[str(i)] = element

    return (ToIndex,ToElement)


def convert_wLT(theList,LT) -> list:
    converted_list = list(map(lambda x:LT[str(x)], theList))
    return converted_list


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])


def get_khop_node_score(test_graph, node_id, edgeid_to_mask_dict, n_of_hops):
    subset_nodes, subset_edge_index, mapping, edge_mask = utils.k_hop_subgraph(node_id, n_of_hops, test_graph.edge_index)
    explained_edges = []
    
    total_score=0.0
    for ind, val in enumerate(edge_mask):
        if val.item():
            # print(original_edges[ind])
            n1, n2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
            # if abs(edgeid_to_mask_dict[(n1,n2)] - edgeid_to_mask_dict[(n2,n1)]) < 0.5:
            explained_edges.append((n1,n2))
            total_score += edgeid_to_mask_dict[(n1,n2)]
                # print((n1,n2), edgeid_to_mask_dict[(n1,n2)])
    
    if len(explained_edges)>0:
        total_score = total_score/len(explained_edges)
        return total_score
    else:
        return 0.0


def get_all_k_hop_node_scores(test_graph, edgeid_to_mask_dict, n_of_hops):
    original_graph = utils.to_networkx(test_graph)
    node_list= original_graph.nodes
    nodeid_score_dict = dict()
    # WARN: There is a strange error in image id 74ul_52
    for node_id in node_list:
        try:
            node_score = get_khop_node_score(test_graph, node_id, edgeid_to_mask_dict, n_of_hops)
            nodeid_score_dict[node_id] = node_score
        except:

            print("ERROR occured when finding node score!")
            nodeid_score_dict[node_id] = 0.0
        

    return nodeid_score_dict


def convert_graph_to_anndata(graph, node_id_to_importance_dict, imp_quant_thr=0.90):
    adata = None
    positions = np.array(graph.pos)
    features = np.array(graph.x)
    clinical_type = graph.clinical_type
    img_id= graph.img_id
    p_id= graph.p_id
    tumor_grade= graph.tumor_gradep
    osmonth= graph.osmonth
    
    obs = [str(val) for val in list(range(graph.x.shape[0]))]
    var = get_gene_list()
    node_importance = []
    for item in obs:
        node_importance.append(node_id_to_importance_dict[int(item)])

    node_importance = np.array(node_importance)
    node_imp_thr = np.quantile(node_importance, imp_quant_thr)

    adata = sc.AnnData(features)
    adata.obs_names = obs
    adata.var_names = var
    adata.obs["clinical_type"] = clinical_type
    adata.obs["img_id"] = str(img_id)
    adata.obs["p_id"] = p_id
    adata.obs["tumor_grade"] = str(tumor_grade)
    adata.obs["osmonth"] = float(osmonth)
    
    adata.obsm["pos"] = positions
    adata.obsm["importance"] = node_importance

    importances_hard = np.array(node_importance > node_imp_thr, dtype="str")
    # print(importances_hard)
    importances_hard = pd.Series(importances_hard, dtype="category")
    # print(importances_hard)
    adata.obs["importance_hard"] = importances_hard.values
    # print(adata.obs)
    sc.tl.rank_genes_groups(adata, groupby="importance_hard", method='wilcoxon', key_added = f"wilcoxon")
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"wilcoxon", show=True, groupby="importance_hard", save="important_vs_unimportant")

    # print(adata.obs)


    adata.write(os.path.join(OUT_DATA_PATH, "adatafiles", f"{graph.img_id}_{graph.p_id}.h5ad"))
    
    # return adata


def get_hvgs(adata):
    pass


def get_gene_list():

    get_id_to_gene_dict = get_basel_zurich_staining_panel()
    lst_features = ['Intensity_MeanIntensity_FullStack_c12',
       'Intensity_MeanIntensity_FullStack_c13',
       'Intensity_MeanIntensity_FullStack_c14',
       'Intensity_MeanIntensity_FullStack_c15',
       'Intensity_MeanIntensity_FullStack_c16',
       'Intensity_MeanIntensity_FullStack_c17',
       'Intensity_MeanIntensity_FullStack_c18',
       'Intensity_MeanIntensity_FullStack_c19',
       'Intensity_MeanIntensity_FullStack_c20',
       'Intensity_MeanIntensity_FullStack_c21',
       'Intensity_MeanIntensity_FullStack_c22',
       'Intensity_MeanIntensity_FullStack_c23',
       'Intensity_MeanIntensity_FullStack_c24',
       'Intensity_MeanIntensity_FullStack_c25',
       'Intensity_MeanIntensity_FullStack_c27',
       'Intensity_MeanIntensity_FullStack_c28',
       'Intensity_MeanIntensity_FullStack_c29',
       'Intensity_MeanIntensity_FullStack_c30',
       'Intensity_MeanIntensity_FullStack_c31',
       'Intensity_MeanIntensity_FullStack_c33',
       'Intensity_MeanIntensity_FullStack_c34',
       'Intensity_MeanIntensity_FullStack_c35',
       'Intensity_MeanIntensity_FullStack_c37',
       'Intensity_MeanIntensity_FullStack_c38',
       'Intensity_MeanIntensity_FullStack_c39',
       'Intensity_MeanIntensity_FullStack_c40',
       'Intensity_MeanIntensity_FullStack_c41',
       'Intensity_MeanIntensity_FullStack_c43',
       'Intensity_MeanIntensity_FullStack_c44',
       'Intensity_MeanIntensity_FullStack_c45',
       'Intensity_MeanIntensity_FullStack_c46',
       'Intensity_MeanIntensity_FullStack_c47',
       'Intensity_MeanIntensity_FullStack_c9']

    lst_genes = [ get_id_to_gene_dict[int(g_id.split("Intensity_MeanIntensity_FullStack_c")[1])] for g_id in lst_features ]
    
    return lst_genes


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)