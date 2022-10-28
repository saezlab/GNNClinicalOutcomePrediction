import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
from pandas.core.dtypes.missing import notna
from data_processing import get_cell_count_df
from data_processing import GRAPH_DIV_THR, CELL_COUNT_THR
from torch_geometric.data import InMemoryDataset, download_url
import pytorch_lightning as pl


pl.seed_everything(42)


S_PATH = os.path.realpath(__file__)


# Overwriting the S_PATH, it doesnt seem to work
S_PATH = os.path.abspath(os.path.dirname(__file__))


# S_PATH = os.path.dirname(__file__)

RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
PLOT_PATH = os.path.join(S_PATH, "../plots")



class TissueDataset(InMemoryDataset):
    def __init__(self, root, wanted_label = "OSmonth", transform=None, pre_transform=None):
        """Creates the dataset for given "root" location and wanted label. Wanted label is currently limited to "treatment", "DiseasesStage", "OSmonth" and "grade."

        Args:
            root (str): _description_
            wanted_label (str, optional): _description_. Defaults to "regression".
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(root, transform, pre_transform)
        self.wanted_label = wanted_label
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["basel_zurich_preprocessed_compact_dataset.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        self.data = pd.read_csv(os.path.join(self.root, "raw", self.raw_file_names[0]))
        
        # GRAPH_DIV_THR


        img_pid_set = set([(item[0], item[1]) for item in self.data[["ImageNumber", "PID"]].values])
        data_list = []
        count = 0

        for fl in os.listdir(os.path.join(self.root, "raw")):
            if fl.endswith("features.pickle"):
                # print(fl)
                img, pid = fl.split("_")[:2]
                with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_clinical_info.pickle'), 'rb') as handle:
                    clinical_info_dict = pickle.load(handle)
                
                # the criteria to select the 
                if (clinical_info_dict["grade"]==3 or  clinical_info_dict["grade"]==2) and pd.notna(clinical_info_dict["OSmonth"]) and clinical_info_dict["diseasestatus"]=="tumor" and pd.notna(clinical_info_dict["clinical_type"]):
                    with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_features.pickle'), 'rb') as handle:
                        feature_arr = pickle.load(handle)
                        feature_arr = np.array(feature_arr)

                    with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_edge_index_length.pickle'), 'rb') as handle:
                        edge_index_arr, edge_length_arr = pickle.load(handle)
                        edge_index_arr = np.array(edge_index_arr)

                    with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_coordinates.pickle'), 'rb') as handle:
                        coordinates_arr = pickle.load(handle)
                        coordinates_arr = np.array(coordinates_arr)
                    if self.wanted_label == "OSmonth":
                        data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=np.log(clinical_info_dict["OSmonth"]+0.1), osmonth=clinical_info_dict["OSmonth"], clinical_type=clinical_info_dict["clinical_type"], tumor_grade=clinical_info_dict["grade"], img_id=img, p_id=pid)
                    elif self.wanted_label == "treatment":
                        data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=clinical_info_dict["treatment"], osmonth=clinical_info_dict["OSmonth"], clinical_type=clinical_info_dict["clinical_type"], tumor_grade=clinical_info_dict["grade"], img_id=img, p_id=pid)
                    elif self.wanted_label == "DiseaseStage":
                        data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=clinical_info_dict["DiseaseStage"], osmonth=clinical_info_dict["OSmonth"], clinical_type=clinical_info_dict["clinical_type"], tumor_grade=clinical_info_dict["grade"], img_id=img, p_id=pid)
                    elif self.wanted_label == "grade":
                        data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=clinical_info_dict["grade"], osmonth=clinical_info_dict["OSmonth"], clinical_type=clinical_info_dict["clinical_type"], tumor_grade=clinical_info_dict["grade"], img_id=img, p_id=pid)
                    data_list.append(data)
                    count+=1
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        """for img, pid in img_pid_set:
            try:
            
                with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_clinical_info.pickle'), 'rb') as handle:
                    clinical_info_dict = pickle.load(handle)
                
                # the criteria to select the 
                if (clinical_info_dict["grade"]==3 or  clinical_info_dict["grade"]==2) and pd.notna(clinical_info_dict["OSmonth"]) and clinical_info_dict["diseasestatus"]=="tumor" and pd.notna(clinical_info_dict["clinical_type"]):
                    with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_features.pickle'), 'rb') as handle:
                        feature_arr = pickle.load(handle)
                        feature_arr = np.array(feature_arr)

                    with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_edge_index_length.pickle'), 'rb') as handle:
                        edge_index_arr, edge_length_arr = pickle.load(handle)
                        edge_index_arr = np.array(edge_index_arr)

                    with open(os.path.join(RAW_DATA_PATH, f'{img}_{pid}_coordinates.pickle'), 'rb') as handle:
                        coordinates_arr = pickle.load(handle)
                        coordinates_arr = np.array(coordinates_arr)
                    
                    data = Data(x=torch.from_numpy(feature_arr).type(torch.FloatTensor), edge_index=torch.from_numpy(edge_index_arr).type(torch.LongTensor).t().contiguous(), pos=torch.from_numpy(coordinates_arr).type(torch.FloatTensor), y=np.log(clinical_info_dict["OSmonth"]+0.1), osmonth=clinical_info_dict["OSmonth"], clinical_type=clinical_info_dict["clinical_type"], tumor_grade=clinical_info_dict["grade"] )
                    data_list.append(data)
                    count+=1

            except:
                pass
        
        print(count)


        
        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])"""





"""with open(os.path.join(RAW_DATA_PATH, f'{img_num}_{pid}_edge_index_length.pickle'), 'wb') as handle:
            pickle.dump((edge_index_arr, edge_length_arr), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        nonfeat_cols = []
        for col in df_image.columns:
            if "MeanIntensity" not in col:
                nonfeat_cols.append(col) 
        # save the feature vector as numpy array, first column is the cell id
        # "ImageNumber", "ObjectNumber", "Location_Center_X", "Location_Center_Y", "PID", "grade", "tumor_size", "age", "treatment", "DiseaseStage", "diseasestatus", "clinical_type", "DFSmonth", "OSmonth"

        with open(os.path.join(RAW_DATA_PATH, f'{img_num}_{pid}_features.pickle'), 'wb') as handle:
            pickle.dump(np.array(df_image.drop(nonfeat_cols, axis=1)), handle, protocol=pickle.HIGHEST_PROTOCOL)

                # save the feature vector as numpy array, first column is the cell id
        with open(os.path.join(RAW_DATA_PATH, f'{img_num}_{pid}_coordinates.pickle'), 'wb') as handle:
            pickle.dump(points, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(RAW_DATA_PATH, f'{img_num}_{pid}_clinical_info.pickle'), 'wb') as handle:"""

"""S_PATH = os.path.dirname(__file__)
dataset = TissueDataset(os.path.join(S_PATH, "../data"))
print(len(dataset))"""