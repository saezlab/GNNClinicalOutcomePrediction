import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from tqdm import tqdm

dataset = "METABRIC"
# dataset = "JacksonFischer"

RAW_DATA_PATH = os.path.join("../data",dataset,"raw")
OUT_DATA_PATH = os.path.join("../data", "out_data", dataset)
PLOT_PATH = os.path.join("../plots", dataset,"graph_voronoi_plots")
GRAPH_DIV_THR = 2500
CELL_COUNT_THR = 100




def get_dataset_from_csv(path="a"):
    """
    path = None
    if dataset == "JacksonFischer":
        path = "../data/JacksonFischer/raw/basel_zurich_preprocessed_compact_dataset.csv"
    elif dataset == "METABRIC":
        path = "../data/METABRIC/raw/merged_data.csv"
    """

    return pd.read_csv(path)


def get_cell_count_df(cell_count_thr,path):
    df_dataset = get_dataset_from_csv(path)
    df_cell_count = df_dataset.groupby(by=["ImageNumber"]).size().reset_index(name='Cell Counts')

    # first_quartile value is 705.75
    df_cell_count = df_cell_count.loc[(df_cell_count['Cell Counts'] >= cell_count_thr)]#  & (df_cell_count['Cell Counts'] <= 2000)]
    return df_cell_count

def generate_graphs_using_points(df_image, imgnum_edge_thr_dict, img_num,  pid,  pos=None, plot=False,PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    
    points = df_image[["Location_Center_X", "Location_Center_Y"]].to_numpy()
    
    # print(df_image.columns)
    
    # point_labels = list(df_image["ObjectNumber"].values)
    
    # divide the graph based on the mean x and mean y values if # of cells is greater than 75 percentile
    if pos:
        img_num_lbl = f"{img_num}{pos}"
    else:
        img_num_lbl = img_num
    
    point_to_lbl_dict = dict()
    for ind, pt in enumerate(points):
        point_to_lbl_dict[tuple(pt)]  = ind
    
    incidence_set = set()

    tri = Delaunay(points)
    small_edges = set()
    large_edges = set()
    for tr in tri.simplices:
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i+1)%3]

            if (edge_idx1, edge_idx0) in small_edges:
                continue  # already visited this edge from other side
            if (edge_idx1, edge_idx0) in large_edges:
                continue
            p0 = points[edge_idx0]
            p1 = points[edge_idx1]

            # print(p0, p1)
            edge_length = np.linalg.norm(p1 - p0)
            if  edge_length <  imgnum_edge_thr_dict[img_num]:
                small_edges.add((edge_idx0, edge_idx1))
                incidence_set.add((point_to_lbl_dict[tuple(p0)], point_to_lbl_dict[tuple(p1)], edge_length))
                incidence_set.add((point_to_lbl_dict[tuple(p1)], point_to_lbl_dict[tuple(p0)], edge_length))
            else:
                large_edges.add((edge_idx0, edge_idx1))

    if plot:
        plt.clf()
        """plt.figure(dpi=300)
        plt.plot(points[:, 0], points[:, 1], '.', markersize=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_onlycells.png")

        plt.clf()
        plt.figure(dpi=300)
        plt.triplot(points[:,0], points[:,1], tri.simplices, linewidth=1, markersize=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_delaunay.png")

        plt.clf()"""
        
        plt.figure(dpi=300)
        plt.plot(points[:, 0], points[:, 1], '.', markersize=2)
        for i, j in small_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], 'r', linewidth=1)
        for i, j in large_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], 'c--', alpha=0.2, linewidth=1)

        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_adapt_thr.pdf")
        
        # print(tri.simplices)

        # plt.triplot(points[:,0], points[:,1], tri.simplices)
        # plt.plot(points[:,0], points[:,1], '.')
        # fig = plt.get_figure()

        
        plt.clf()
        
        plt.figure(dpi=300)
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_{pid}_voronoi.pdf", dpi=300)
        plt.clf()
    
    
    edge_index_arr = np.array([list(edge)[:2] for edge in incidence_set], dtype=np.int32)
    edge_length_arr= np.array([list(edge)[-1] for edge in incidence_set])
    
    
    assert edge_index_arr.shape[0]==edge_length_arr.shape[0]

    clinical_info_dict = dict()

    # TODO: Refactor this part
    if "JacksonFischer" in RAW_DATA_PATH:
        clinical_info_dict["grade"] = df_image["grade"].values[0]
        clinical_info_dict["tumor_size"] = df_image["tumor_size"].values[0]
        clinical_info_dict["treatment"] = df_image["treatment"].values[0]
        clinical_info_dict["age"] = df_image["age"].values[0]
        clinical_info_dict["DiseaseStage"] = df_image["DiseaseStage"].values[0]
        clinical_info_dict["diseasestatus"] = df_image["diseasestatus"].values[0]
        clinical_info_dict["clinical_type"] = df_image["clinical_type"].values[0]
        clinical_info_dict["DFSmonth"] = df_image["DFSmonth"].values[0]
        clinical_info_dict["OSmonth"] = df_image["OSmonth"].values[0]
        clinical_info_dict["Patientstatus"] = df_image["Patientstatus"].values[0]
        # clinical_info_dict["cell_type"] = df_image["cell_type"].values[0]
        # clinical_info_dict["class"] = df_image["class"].values[0]
        clinical_info_dict["cell_count"] = len(df_image)
    elif "METABRIC" in RAW_DATA_PATH:
        metabric_clinical_feat = ['SOM_nodes', 'pg_cluster', 'description',
                                  'Study ID', 'Patient ID', 'PID', 'age', 'Type of Breast Surgery',
                                  'cell_type', 'Cancer Type Detailed', 'Cellularity', 'treatment',
                                  'Pam50 + Claudin-low subtype', 'Cohort', 'ER status measured by IHC',
                                  'ER Status', 'grade', 'HER2 status measured by SNP6', 'HER2 Status',
                                  'Tumor Other Histologic Subtype', 'Hormone Therapy',
                                  'Inferred Menopausal State', 'Integrative Cluster',
                                  'Primary Tumor Laterality', 'Lymph nodes examined positive',
                                  'Mutation Count', 'DiseaseStage', 'Oncotree Code', 'OSmonth',
                                  'Overall Survival Status', 'PR Status', 'Radio Therapy', 'DFSmonth',
                                  'Relapse Free Status', 'Number of Samples Per Patient', 'Sample Type',
                                  'Sex', '3-Gene classifier subtype', 'TMB (nonsynonymous)', 'tumor_size',
                                   'Tumor Stage', 'diseasestatus', 'clinical_type']
        for col in metabric_clinical_feat:
            clinical_info_dict[col] = df_image[col].values[0]
        
        clinical_info_dict["cell_count"] = len(df_image)


    # save the edge indices and as a list 
    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_edge_index_length.pickle'), 'wb') as handle:
        pickle.dump((edge_index_arr, edge_length_arr), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    nonfeat_cols = []
    for col in df_image.columns:
        if "MeanIntensity" not in col:
            nonfeat_cols.append(col) 
    # save the feature vector as numpy array, first column is the cell id
    # "ImageNumber", "ObjectNumber", "Location_Center_X", "Location_Center_Y", "PID", "grade", "tumor_size", "age", "treatment", "DiseaseStage", "diseasestatus", "clinical_type", "DFSmonth", "OSmonth"

    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_features.pickle'), 'wb') as handle:
        pickle.dump(np.array(df_image.drop(nonfeat_cols, axis=1)), handle, protocol=pickle.HIGHEST_PROTOCOL)

    if "METABRIC" in RAW_DATA_PATH:
        with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_ct_class.pickle'), 'wb') as handle:
            pickle.dump(df_image[["cell_type"]].to_numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_ct_class.pickle'), 'wb') as handle:
            pickle.dump(df_image[["cell_type", "class"]].to_numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # save the feature vector as numpy array, first column is the cell id
    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_coordinates.pickle'), 'wb') as handle:
        pickle.dump(points, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(RAW_DATA_PATH, f'{img_num_lbl}_{pid}_clinical_info.pickle'), 'wb') as handle:
        pickle.dump(clinical_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_edge_length_dist(data_path,cell_count_thr, quant, plot_dist=False, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    """
    Calculate the distribution of edge lengths for each image in the dataset.

    Args:
        cell_count_thr (int): Cell count threshold used to filter the dataset.
        quant (float): The quantile value to calculate the edge length threshold.
        plot_dist (bool, optional): Whether to plot and save the edge length distribution for each image.
                                    Defaults to False.

    Returns:
        dict: A dictionary containing the image number as key and its corresponding edge length threshold as value.
    """
    df_dataset = pd.read_csv(data_path)
    df_cell_count = get_cell_count_df(cell_count_thr,path=data_path)
    imgnum_edge_thr_dict = dict()
    all_edge_count_list = []
    
    for ind, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        
        img_num = row["ImageNumber"]
        cell_count = row["Cell Counts"]
        df_image = df_dataset[df_dataset["ImageNumber"]==img_num]
        points = [list(item) for item in list(df_image[["Location_Center_X", "Location_Center_Y"]].values)]
        # print(points)
        # point_labels = [list(item) for item in list(df_image["ObjectNumber"].values)]
        point_to_lbl_dict = dict()
        for ind, pt in enumerate(points):
            point_to_lbl_dict[tuple(pt)]  = ind
        
        tri = Delaunay(points)
        edge_set = set()
        edge_length = []
        # tri.simplices are array of points each element in the array has 3 points
        for tr in tri.simplices:
            for i in range(3):
                edge_idx0 = tr[i]
                edge_idx1 = tr[(i+1)%3]

                if (edge_idx1, edge_idx0) in edge_set:
                    continue  # already visited this edge from other side
                edge_set.add((edge_idx0, edge_idx1))
                p0 = points[edge_idx0]
                p1 = points[edge_idx1]
                
                # print(np. where(points ==p0))
                
                # print(p0, p1)

                edge_length.append(np.linalg.norm(np.array(p1) - np.array(p0)))
        

        edge_thr = np.quantile(edge_length, quant)
        imgnum_edge_thr_dict[img_num] = edge_thr
    
        if plot_dist:
            plt.clf()
            plt.figure(dpi=300)
            edge_dist_plot = sns.displot(data=edge_length, kde=True)
            plt.axvline(x=edge_thr, linestyle='--', color="black")
            # fig = edge_dist_plot.get_figure()
            plt.savefig(f"{PLOT_PATH}/{img_num}_edge_distribution.pdf")
            plt.clf()
        
        
        # edge_length
        all_edge_count_list.append(edge_length)
    
    with open(os.path.join(RAW_DATA_PATH, 'edge_thr.pickle'), 'wb') as handle:
        pickle.dump(imgnum_edge_thr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
                

def create_graphs_delauney_triangulation(data_path, cell_count_thr=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, plot=False, RAW_DATA_PATH = RAW_DATA_PATH, PLOT_PATH=PLOT_PATH):

    df_dataset = get_dataset_from_csv(data_path)
    df_cell_count = get_cell_count_df(cell_count_thr,data_path)
    imgnum_edge_thr_dict = dict()

    # print(df_dataset["PID"])

    with open(os.path.join(RAW_DATA_PATH, 'edge_thr.pickle'), 'rb') as handle:
            imgnum_edge_thr_dict = pickle.load(handle)

    for ind, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        # print(row)
        img_num = row["ImageNumber"]
        cell_count = row["Cell Counts"]
        df_image = df_dataset[df_dataset["ImageNumber"]==img_num]
        pid = df_image["PID"].values[0]
    
        new_cell_ids = list(range(len(df_image)))
        
        # TODO: make this parametric 
        if cell_count >= GRAPH_DIV_THR:
            # find the center of the cells
            x_center, y_center = df_image[["Location_Center_X", "Location_Center_Y"]].describe().loc["mean"]["Location_Center_X"], df_image[["Location_Center_X", "Location_Center_Y"]].describe().loc["mean"]["Location_Center_Y"]
            # ll lower-left  ul upper-left lr lower right points 
            """ll_points = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] <= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()
            ul_points = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] >= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()
            lr_points = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] <= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()
            ur_points = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] >= y_center))][["Location_Center_X", "Location_Center_Y"]].to_numpy()"""

            ll_df_image = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] <= y_center))]
            ul_df_image = df_image[((df_image['Location_Center_X'] <= x_center) & (df_image['Location_Center_Y'] >= y_center))]
            lr_df_image = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] <= y_center))]
            ur_df_image = df_image[((df_image['Location_Center_X'] >= x_center) & (df_image['Location_Center_Y'] >= y_center))]

            

            generate_graphs_using_points(ll_df_image, imgnum_edge_thr_dict, img_num, pid, "ll", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(ul_df_image, imgnum_edge_thr_dict, img_num, pid, "ul", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(lr_df_image, imgnum_edge_thr_dict, img_num, pid, "lr", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)
            generate_graphs_using_points(ur_df_image, imgnum_edge_thr_dict, img_num, pid, "ur", plot, PLOT_PATH=PLOT_PATH,RAW_DATA_PATH=RAW_DATA_PATH)

        else:
            # points = df_image[["Location_Center_X", "Location_Center_Y"]].to_numpy()
            generate_graphs_using_points(df_image, imgnum_edge_thr_dict, img_num, pid, pos=None, plot = plot, PLOT_PATH=PLOT_PATH)

       
def check_cell_ids_sequential():
    df_dataset = get_dataset_from_csv()
    df_cell_count = get_cell_count_df(0)


    for ind, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        # print(row)
        img_num = row["ImageNumber"]
        cell_count = row["Cell Counts"]
        df_image = df_dataset[df_dataset["ImageNumber"]==img_num]

        pid = df_image["PID"].values[0]
        points = df_image[["Location_Center_X", "Location_Center_Y"]].to_numpy()
        point_labels = list(df_image["ObjectNumber"].values)
        max_val = max(point_labels)
        # print(max_val)
        #print(list(range(1,max_val+1)))
        # print(point_labels)
        print(list(range(1,max_val+1)))
        assert point_labels==list(range(1,max_val+1))



# To generate the dataset run the below functions
# 1) get_edge_length_dist(CELL_COUNT_THR, 0.975, plot_dist=True)
# 2) get_cell_count_df(CELL_COUNT_THR)
# 3) create_graphs_delauney_triangulation(CELL_COUNT_THR, plot=True)

# create_graphs_delauney_triangulation(CELL_COUNT_THR, plot=True)

# check if all cell ids are available 
# check_cell_ids_sequential()

def data_processing_pipeline(data_path, CELL_COUNT_THR=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    # TODO make this parametric
    from data_preparation import create_preprocessed_sc_feature_fl
    print("Creating compact dataset...")
    # create_preprocessed_sc_feature_fl()
    print("Calculating edge length distribution....")
    get_edge_length_dist(data_path=data_path, cell_count_thr=CELL_COUNT_THR, quant= 0.975, plot_dist=False, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)
    print("Calculating cell count distribution....")
    df_cell_count = get_cell_count_df(CELL_COUNT_THR, path=data_path)
    quartile = int(np.quantile(df_cell_count["Cell Counts"], 0.50))

    print("Creating graphs...")
    create_graphs_delauney_triangulation(cell_count_thr=CELL_COUNT_THR, GRAPH_DIV_THR=quartile, plot=True,RAW_DATA_PATH = RAW_DATA_PATH, data_path=data_path, PLOT_PATH=PLOT_PATH)

# data_processing_pipeline("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/data/JacksonFischer/raw/merged_preprocessed_dataset.csv")

def data_processing_metabric_pipeline(data_path, CELL_COUNT_THR=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH):
    # TODO make this parametric
    from data_preparation import METABRIC_preprocess
    # print("Creating compact dataset...")
    # METABRIC_preprocess(visualize = False)
    print("Calculating edge length distribution....")
    get_edge_length_dist(data_path=data_path, cell_count_thr=CELL_COUNT_THR, quant= 0.975, plot_dist=False, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)
    
    print("Calculating cell count distribution....")
    df_cell_count = get_cell_count_df(CELL_COUNT_THR, path=data_path)
    quartile = int(np.quantile(df_cell_count["Cell Counts"], 0.50))
    print(RAW_DATA_PATH)
    print("Creating graphs...")
    create_graphs_delauney_triangulation(cell_count_thr=CELL_COUNT_THR, GRAPH_DIV_THR=quartile, plot=True,RAW_DATA_PATH = RAW_DATA_PATH, data_path=data_path, PLOT_PATH=PLOT_PATH)


# data_processing_metabric_pipeline("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/data/METABRIC/raw/merged_preprocessed_dataset.csv",CELL_COUNT_THR=CELL_COUNT_THR, GRAPH_DIV_THR=GRAPH_DIV_THR, PLOT_PATH=PLOT_PATH, RAW_DATA_PATH=RAW_DATA_PATH)