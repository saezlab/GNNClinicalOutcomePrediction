import os
import argparse
import numpy as np
import pandas as pd
from model import CustomGCN
from evaluation_metrics import r_squared_score, mse, rmse

def calculate_reg_scores_from_preds(folder_path, file_name):
    experiment_name = os.path.split(folder_path)[-1]
    df_results = pd.read_csv(os.path.join(folder_path, file_name), sep=",")
    job_id = file_name.split(".csv")[0]
    
    result_list = [experiment_name, job_id]
    for idx, val in enumerate(["train", "validation", "test"]):

        df_tvt = df_results.loc[(df_results['Fold#-Set'].str[2:] == val)]
        col_name = "True Value"
        if "OS Month (log)" in df_tvt.columns:
            col_name = "OS Month (log)"

        r2_score = r_squared_score(df_tvt[col_name], df_tvt['Predicted'])
        mser = mse(df_tvt[col_name], df_tvt['Predicted'])
        result_list.extend([r2_score, mser])
    
    return result_list

def calculate_all_reg_scores(folder_path_list):


    header = ["experiment name", "job id", "train_r2 score", "train_mse", "val_r2 score", "val_mse", "test_r2 score", "test_mse"]

    for folder_path in folder_path_list:
        all_results = []
        for fl in os.listdir(folder_path):
            if fl.endswith(".csv"):
                all_results.append(calculate_reg_scores_from_preds(folder_path, fl))
        
    df_results = pd.DataFrame (all_results, columns = header)
    df_results.sort_values(by=["val_r2 score", "test_r2 score"], inplace=True, ascending=False)

    return df_results


# calculate_all_reg_scores(["/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/data/out_data/PNA_training_12-10-2022"])