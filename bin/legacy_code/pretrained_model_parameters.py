from cProfile import label
from sklearn.metrics import accuracy_score, f1_score, precision_score
import torch
from model import CustomGCN
from dataset import TissueDataset
from torch_geometric.loader import DataLoader
import numpy as np
import plotting
from tqdm import tqdm
import pandas as pd
import os
from torch_geometric.utils import degree
from evaluation_metrics import r_squared_score, mse, rmse
import custom_tools as custom_tools
import csv
import statistics



class trainer_tester:


    def __init__(self, parser_args, setup_args ) -> None:
        """Manager of processes under train_tester, main goel is to show the processes squentially

        Args:
            parser_args (Namespace): Holds the arguments that came from parsing CLI
            setup_args (Namespace): Holds the arguments that came from setup
        """
        self.parser_args = parser_args
        self.setup_args = setup_args
        self.set_device()

        self.init_folds()

        #self.train_test_loop()

        # self.save_results()

    def set_device(self):
        """Sets up the computation device for the class
        """
        self.device = custom_tools.get_device()

    def init_folds(self):
        """Pulls data, creates samplers according to ratios, creates train, test and validation loaders for 
        each fold, saves them under the 'folds_dict' dictionary
        """
        # self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data"))
        self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data"))

        if self.parser_args.label == "OSMonth":
            self.label_type = "regression"
            self.num_classes = 1

        elif self.parser_args.label == "treatment":
            self.label_type = "classification"
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()
            self.dataset.data.y = self.dataset.data.clinical_type
            self.unique_classes = set(self.dataset.data.clinical_type)
            self.num_classes = len(self.unique_classes)


        # WARN CURRENTLY DOESN'T PULL THE DATA NOT WORKING
        elif self.parser_args.label == "DiseaseStage":
            self.label_type = "classification"
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()

        elif self.parser_args.label == "grade":
            self.label_type = "classification"
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()
            self.dataset.data.y = self.dataset.data.tumor_grade
            self.unique_classes = torch.unique(self.dataset.data.tumor_grade)
            self.num_classes = len(self.unique_classes)

        self.label_data = self.dataset.data.y
        if self.label_type=="classification":
            self.LT_ToIndex, self.LT_FromIndex = custom_tools.extract_LT(self.dataset.data.y)
            self.dataset.data.y = custom_tools.convert_wLT(self.dataset.data.y,self.LT_ToIndex)

        self.dataset = self.dataset.shuffle()

        

        self.samplers = custom_tools.k_fold_ttv(self.dataset, 
            T2VT_ratio=self.setup_args.T2VT_ratio,
            V2T_ratio=self.setup_args.V2T_ratio)


        self.fold_dicts = []

        deg = -1

        for fold, train_sampler, validation_sampler, test_sampler in self.samplers:
            train_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= train_sampler)
            validation_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= validation_sampler)
            test_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= test_sampler)

            if self.parser_args.model == "PNAConv":
                deg = self.calculate_deg(train_sampler)
            
            model = self.set_model(deg)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.parser_args.lr, weight_decay=self.parser_args.weight_decay)

            fold_dict = {
                "fold": fold,
                "train_loader": train_loader,
                "validation_loader": validation_loader,
                "test_loader": test_loader,
                "deg": deg,
                "model": model,
                "optimizer": optimizer
            }

            self.fold_dicts.append(fold_dict)

            if not self.setup_args.use_fold:
                break

    def calculate_deg(self,train_sampler):
        """Calcualtes deg, which is necessary for some models

        Args:
            train_sampler (_type_): Training data sampler
        """
        train_dataset = self.dataset[train_sampler.indices]
        max_degree = -1
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        return deg

    def set_model(self, deg):
        """Sets the model according to parser parameters and deg

        Args:
            deg (_type_): degree data of the graph
        """
        model = CustomGCN(
                    type = self.parser_args.model,
                    num_node_features = self.dataset.num_node_features, ####### LOOOOOOOOK HEREEEEEEEEE
                    num_gcn_layers=self.parser_args.num_of_gcn_layers, 
                    num_ff_layers=self.parser_args.num_of_ff_layers, 
                    gcn_hidden_neurons=self.parser_args.gcn_h, 
                    ff_hidden_neurons=self.parser_args.fcl, 
                    dropout=self.parser_args.dropout,
                    aggregators=self.parser_args.aggregators,
                    scalers=self.parser_args.scalers,
                    deg = deg, # Comes from data not hyperparameter
                    num_classes = self.num_classes,
                    label_type = self.label_type
                        ).to(self.device)
        checkpoint = torch.load('../data/models/pna_model.pth')
        model.load_state_dict(checkpoint)

        params = model.state_dict()
        #batch = data.y.size(0)
            
        #for bs in range(batch):
        #    data_each = data[bs]
        #    print(data_each)
        #print(params)
        param = params['convs.0.pre_nns.0.0.weight']
        print('convs.0.pre_nns.0.0.weight:', param.shape)
        param_list = param.tolist()
        print(param_list)
        print("--------------------------------------")
        param = params['convs.0.post_nns.0.0.weight']
        print('convs.0.post_nns.0.0.weight:', param.shape)
        param_list = param.tolist()
        print(param_list)
        #print(params)
        print("--------------------------------------")
        param = params['convs.0.lin.weight']
        print('convs.0.lin.weight:', param.shape)
        param_list = param.tolist()
        print(param_list)
        #print(params)
        print("--------------------------------------")
        param = params['convs.1.pre_nns.0.0.weight']
        print('convs.1.pre_nns.0.0.weight:', param.shape)
        param_list = param.tolist()
        print(param_list)
        #print(params)
        print("--------------------------------------")
        param = params['convs.1.post_nns.0.0.weight']
        print('convs.1.post_nns.0.0.weight:', param.shape)
        param_list = param.tolist()
        print(param_list)
        #print(params)
        print("--------------------------------------")
        param = params['convs.1.lin.weight']
        print('convs.1.lin.weight:', param.shape)
        param_list = param.tolist()
        print(param_list)
        #print(params)
        return model


import torch
from data_processing import OUT_DATA_PATH
import os
import pytorch_lightning as pl
import custom_tools as custom_tools
from types import SimpleNamespace
#from trainer_tester import trainer_tester




pl.seed_everything(42)
parser_args = custom_tools.general_parser()
setup_args = SimpleNamespace()

setup_args.id = custom_tools.generate_session_id()

print(f"Session id: {setup_args.id}")

setup_args.S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
setup_args.OUT_DATA_PATH = os.path.join(setup_args.S_PATH, "../data", "out_data", parser_args.en)
setup_args.RESULT_PATH = os.path.join(setup_args.S_PATH, "../results", "idedFiles", parser_args.en)
setup_args.PLOT_PATH = os.path.join(setup_args.S_PATH, "../plots", parser_args.en)
setup_args.MODEL_PATH = os.path.join(setup_args.S_PATH, "../models", parser_args.en)

custom_tools.create_directories([setup_args.OUT_DATA_PATH, setup_args.RESULT_PATH, setup_args.PLOT_PATH, setup_args.MODEL_PATH])

setup_args.T2VT_ratio = 4
setup_args.V2T_ratio = 1
setup_args.use_fold = parser_args.fold

print(setup_args.use_fold )


# This is NOT for sure, loss can change inside the class
setup_args.criterion = torch.nn.MSELoss()
setup_args.print_every_epoch = 10
setup_args.plot_result = True

device = custom_tools.get_device()


# Object can be saved if wanted
trainer_tester(parser_args, setup_args)