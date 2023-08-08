from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUT_DATA_PATH = os.path.join("../data", "out_data", "jackson")
PLOT_PATH = os.path.join("../plots", "jackson/graph_voronoi_plots")
GRAPH_DIV_THR = 2500
CELL_COUNT_THR = 100


def get_dataset_from_csv(path = "../data/JacksonFischer/raw/basel_zurich_preprocessed_compact_dataset.csv"):
    return pd.read_csv(path)


def get_cell_count_df(cell_count_thr,path):
    df_dataset = get_dataset_from_csv(path)
    df_cell_count = df_dataset.groupby(by=["ImageNumber"]).size().reset_index(name='Cell Counts')

    # first_quartile value is 705.75
    df_cell_count = df_cell_count.loc[(df_cell_count['Cell Counts'] >= cell_count_thr)]#  & (df_cell_count['Cell Counts'] <= 2000)]
    return df_cell_count

def generate_graphs_using_points(df_image, imgnum_edge_thr_dict, img_num,  pid,  pos=None, plot=False,PLOT_PATH=PLOT_PATH):
    
    points = df_image[["Location_Center_X", "Location_Center_Y"]].to_numpy()
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

        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_adapt_thr.pdf")
        
        # print(tri.simplices)

        # plt.triplot(points[:,0], points[:,1], tri.simplices)
        # plt.plot(points[:,0], points[:,1], '.')
        # fig = plt.get_figure()

        
        plt.clf()
        
        plt.figure(dpi=300)
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6, point_size=2)
        plt.savefig(f"{PLOT_PATH}/{img_num_lbl}_voronoi.pdf", dpi=300)
        plt.clf()
    
    
    edge_index_arr = np.array([list(edge)[:2] for edge in incidence_set], dtype=np.int32)
    edge_length_arr= np.array([list(edge)[-1] for edge in incidence_set])
    
    
    assert edge_index_arr.shape[0]==edge_length_arr.shape[0]


def get_edge_length_dist(data_path,cell_count_thr, quant, plot_dist=False, PLOT_PATH=PLOT_PATH, OUT_DATA_PATH=OUT_DATA_PATH):
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
    df_dataset = get_dataset_from_csv(data_path)
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
    
    with open(os.path.join(OUT_DATA_PATH, 'edge_thr.pickle'), 'wb') as handle:
        pickle.dump(imgnum_edge_thr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
                

def create_graphs_delauney_triangulation(cell_count_thr,data_path, plot=False, OUT_DATA_PATH = OUT_DATA_PATH, PLOT_PATH=PLOT_PATH):

    df_dataset = get_dataset_from_csv(data_path)
    df_cell_count = get_cell_count_df(cell_count_thr,data_path)
    imgnum_edge_thr_dict = dict()

    with open(os.path.join(OUT_DATA_PATH, 'edge_thr.pickle'), 'rb') as handle:
            imgnum_edge_thr_dict = pickle.load(handle)

    for ind, row in tqdm(df_cell_count.iterrows(), total=len(df_cell_count)):
        # print(row)
        img_num = row["ImageNumber"]
        cell_count = row["Cell Counts"]
        df_image = df_dataset[df_dataset["ImageNumber"]==img_num]
        pid = df_image["PID"].values[0]
        new_cell_ids = list(range(len(df_image)))
        
        # TODO: make this parametric 
        if cell_count > GRAPH_DIV_THR:
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

            

            generate_graphs_using_points(ll_df_image, imgnum_edge_thr_dict, img_num, pid, "ll", plot, PLOT_PATH=PLOT_PATH)
            generate_graphs_using_points(ul_df_image, imgnum_edge_thr_dict, img_num, pid, "ul", plot, PLOT_PATH=PLOT_PATH)
            generate_graphs_using_points(lr_df_image, imgnum_edge_thr_dict, img_num, pid, "lr", plot, PLOT_PATH=PLOT_PATH)
            generate_graphs_using_points(ur_df_image, imgnum_edge_thr_dict, img_num, pid, "ur", plot, PLOT_PATH=PLOT_PATH)

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
        print(point_labels)
        print(list(range(1,max_val+1)))
        assert point_labels==list(range(1,max_val+1))



# To generate the dataset run the below functions
# 1) get_edge_length_dist(CELL_COUNT_THR, 0.975, plot_dist=True)
# 2) get_cell_count_df(CELL_COUNT_THR)
# 3) create_graphs_delauney_triangulation(CELL_COUNT_THR, plot=True)

# create_graphs_delauney_triangulation(CELL_COUNT_THR, plot=True)

# check if all cell ids are available 
# check_cell_ids_sequential()

def data_processing_pipeline(data_path,CELL_COUNT_THR=CELL_COUNT_THR,GRAPH_DIV_THR=GRAPH_DIV_THR,PLOT_PATH=PLOT_PATH, OUT_DATA_PATH=OUT_DATA_PATH):
    # TODO make this parametric
    get_edge_length_dist(data_path=data_path,cell_count_thr=CELL_COUNT_THR,quant= 0.975, plot_dist=False,PLOT_PATH=PLOT_PATH, OUT_DATA_PATH=OUT_DATA_PATH)
    get_cell_count_df(CELL_COUNT_THR, path=data_path)
    create_graphs_delauney_triangulation(CELL_COUNT_THR, plot=True,OUT_DATA_PATH = OUT_DATA_PATH, data_path=data_path, PLOT_PATH=PLOT_PATH)