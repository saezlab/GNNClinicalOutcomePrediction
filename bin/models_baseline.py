# -*- coding: utf-8 -*-
import torch
import os, sys
import numpy as np
from sklearn.model_selection import RandomizedSearchCV,RepeatedKFold, PredefinedSplit
import pickle
from tqdm import tqdm
import pandas as pd
from dataset import TissueDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings
from torch_geometric.loader import DataLoader
import custom_tools
import evaluation_metrics
from sklearn.model_selection import ParameterSampler
import baseline_hyperparams
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")


class Regressors(object):


    def __init__(self,):
        
        """
        Description: 
            Six different machine learning methods for regression
            are introduced. Their hyperparameters are tuned by
            RandomizedSearchCV and all methods return only their hyperparameters 
            that give the best respect to cross-validation that is created by RepeatedKFold.

        Parameters:
            path: {string}, A destination point where model is saved.
            X_train: Feature matrix, {list, numpy array}
            y_train: (default = None), Label matrix, type = {list, numpy array}
            X_valid: (default = None), Validation Set, type = {list,numpy array}
            y_valid: (default = None), Validation Label, type = {list,numpy array}

        Returns:
            model: Parameters of fitted model
        """
        self.dataset = TissueDataset(os.path.join(S_PATH,"../data"))
        self.dataset = self.dataset.shuffle()
        print("init", self.dataset)
        self.parameters = None
        self.n_jobs = -1
        self.random_state = 0
      
    
    def init_folds(self):
        """Pulls data, creates samplers according to ratios, creates train, test and validation loaders for 
        each fold, saves them under the 'folds_dict' dictionary
        """
        # self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data"))
        
        print("Dataset:",  self.dataset)

        self.fold_dicts = []
        self.samplers = custom_tools.k_fold_ttv(self.dataset, 
                T2VT_ratio=4,
                V2T_ratio=1)
        
        for fold, train_sampler, validation_sampler, test_sampler in self.samplers:
            
            train_loader = DataLoader(self.dataset, sampler= train_sampler)
            validation_loader = DataLoader(self.dataset, sampler= validation_sampler)
            test_loader = DataLoader(self.dataset, sampler= test_sampler)
            
            train_features, train_labels = self.create_pseudobulk_datasets(train_loader)
            validation_features, validation_labels = self.create_pseudobulk_datasets(validation_loader)
            test_features, test_labels = self.create_pseudobulk_datasets(test_loader)

            fold_dict = {
                "fold": fold,
                "train_features": np.array(train_features),
                "train_labels": train_labels,
                "validation_features": validation_features,
                "validation_labels": validation_labels,
                "test_features": test_features,
                "test_labels": test_labels
            }
            
            self.fold_dicts.append(fold_dict)


    def create_pseudobulk_datasets(self, loader):
        features, labels  = [], []
        for data in loader:  # Iterate in batches over the training dataset.
                pseudo_bulk_features = np.array(torch.sum(data.x,0) / data.x.shape[0])
                labels.append(data.y.item())
                features.append(pseudo_bulk_features)

            
        return np.array(features), labels


    def regress_baseline_model(self, classifier):
        
        print(f"Performing cross validation using {classifier}...")
        if classifier=="SVR":
            from baseline_hyperparams import rgr_svm_params as hyperparams
        elif classifier =="RF":
            from baseline_hyperparams import rgr_random_forest_params as hyperparams
        elif classifier == "LR":
            from baseline_hyperparams import rgr_linear_regression_params as hyperparams
        elif classifier =="MLP":
            from baseline_hyperparams import rgr_mlp_params as hyperparams
        else:
            raise Exception("Undefined classifier! Please provide a valid classifier name")

                
        
        
        param_list = list(ParameterSampler(hyperparams, n_iter=100))
        # append default params
        param_list.append({})
        
        param_dict_result = dict()
        for param in tqdm(param_list):
            param_str = self.dict_to_str(param)
            param_dict_result[param_str] = dict()
            for fold in self.fold_dicts:
                regressor = None
                if classifier=="SVR":
                    regressor= SVR(**param)
                elif classifier =="RF":
                    regressor= RandomForestRegressor(**param)
                elif classifier == "LR":
                    regressor = LinearRegression(**param)
                elif classifier == "MLP":
                    regressor = MLPRegressor(**param)
                
                fold_num = fold["fold"]
                model = regressor.fit(fold["train_features"],fold["train_labels"])
                val_predictions = model.predict(fold["validation_features"])
                test_predictions = model.predict(fold["test_features"])
                val_r2_score = evaluation_metrics.r_squared_score(fold["validation_labels"], val_predictions)
                val_mse_score = evaluation_metrics.mse(fold["validation_labels"], val_predictions)
                test_r2_score = evaluation_metrics.r_squared_score(fold["test_labels"], test_predictions)
                test_mse_score = evaluation_metrics.mse(fold["test_labels"], test_predictions)
                result_list = [val_r2_score, val_mse_score, test_r2_score, test_mse_score]
                result_list = [round(val, 3) for val in result_list]
                param_dict_result[param_str][fold_num] = result_list
            
        best_result_param = ""
        best_average_r2 = -np.inf
        
        for key in param_dict_result:
            result_arr = []
            for fold in param_dict_result[key]:
                result_arr.append(param_dict_result[key][fold])
            result_arr = np.array(result_arr)
        
            mean_results = np.mean(result_arr, 0)
            # check val r2 score
            if mean_results[0] > best_average_r2:
                best_average_r2 = mean_results[0]
                best_result_param = key
        
        rows = []
        for fold in param_dict_result[best_result_param]:
            rows.append([fold])
            rows[-1].extend(param_dict_result[best_result_param][fold])
        df = pd.DataFrame(rows, columns = ["Fold #", "Validation R2", "Validation MSE", "Test R2", "Test MSE"])
        df.to_csv(os.path.join(OUT_DATA_PATH, "baseline_predictors", f"{classifier}_best_performance.csv" ))
        
    def run_k_fold_baseline(self):
        self.init_folds()
        self.regress_baseline_model("SVR")
        self.regress_baseline_model("RF")
        self.regress_baseline_model("LR")
        self.regress_baseline_model("MLP")

    def dict_to_str(self, i_dict):
        # an empty string
        converted = str()
        for key in i_dict:
            converted += key + ": " + str(i_dict[key]) + ", "
        return converted

    def get_best_model(self, model, X_train, y_train,X_valid, y_valid):

        if X_valid is None: 
            
            cv = RepeatedKFold(n_splits=10,n_repeats = 5,random_state= self.random_state)
            
        else:
            
            if y_valid is None:
                raise ValueError(f'True label data for validation set cannot be None')
            
            X_train = list(X_train)
            y_train = list(y_train)
            
            len_tra, len_val = len(X_train),len(X_valid)
            
            X_train.extend(list(X_valid))
            y_train.extend(list(y_valid))
            
            test_fold = [0 if x in np.arange(len_tra) else -1 for x in np.arange(len_tra+len_val)]
            cv = PredefinedSplit(test_fold)


        clf = RandomizedSearchCV(model,self.parameters,n_iter = 10,
                                     n_jobs=self.n_jobs, cv = cv,
                                     scoring="f1")

        
        if y_train is not None:
            clf.fit(X_train,y_train)
        else:
            clf.fit(X_train)
        best_model = clf.best_estimator_

        """if self.path is not None:

            with open(self.path, 'wb') as f:
                pickle.dump(best_model,f)"""

        return best_model


    def MLP(self,X_train,y_train,X_valid,y_valid):

        return self.get_best_model(model, X_train, y_train,X_valid, y_valid)




regressor_obj = Regressors()
regressor_obj.run_k_fold_baseline()






