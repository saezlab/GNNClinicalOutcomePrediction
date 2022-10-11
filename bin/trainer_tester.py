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
from evaluation_metrics import r_squared_score
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
        self.set_device()

        self.parser_args = parser_args
        self.setup_args = setup_args

        self.init_folds()

        self.train_test_loop()

        self.save_results()

    def set_device(self):
        """Sets up the computation device for the class
        """
        self.device = custom_tools.get_device()

    def init_folds(self):
        """Pulls data, creates samplers according to ratios, creates train, test and validation loaders for 
        each fold, saves them under the 'folds_dict' dictionary
        """

        self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data"))
        print(len(self.dataset))
        self.dataset = self.dataset.shuffle()
        print(len(self.dataset))
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
                    deg = deg # Comes from data not hyperparameter
                        ).to(self.device)

        return model

    def train(self, fold_dict):
        """Trains the network

        Args:
            fold_dict (dict): Holds data about the used fold

        Returns:
            float: Total loss
        """
        fold_dict["model"].train()
        total_loss = 0.0
        pred_list = []
        out_list = []
        
        for data in fold_dict["train_loader"]:  # Iterate in batches over the training dataset.
            out = fold_dict["model"](data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.

            loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.
        
            loss.backward()  # Derive gradients.
            out_list.extend([val.item() for val in out.squeeze()])
            
            pred_list.extend([val.item() for val in data.y])

            total_loss += float(loss.item())
            fold_dict["optimizer"].step()  # Update parameters based on gradients.
            fold_dict["optimizer"].zero_grad()  # Clear gradients

        return total_loss

    def test(self, fold_dict, test_on: str,label=None, fl_name=None, plot_pred=False):
        """Tests the model on wanted loader

        Args:
            fold_dict (dict): Holds data about the used fold
            test_on (str): Which loader to test on (train_loader, test_loader, valid_loader)
            label (_type_, optional): Label of the loader. Defaults to None.
            fl_name (_type_, optional): Name of the loader. Defaults to None.
            plot_pred (bool, optional): Should there be a plot. Defaults to False.

        Returns:
            float: total loss
        """

        loader = fold_dict[test_on]
        fold_dict["model"].eval()

        total_loss = 0.0
        pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list = [], [], [], [], [], [], []
        
        for data in loader:  # Iterate in batches over the training/test dataset.
            
            if data.y.shape[0]>1:
                out = fold_dict["model"](data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.
                loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.
                
                total_loss += float(loss.item())

                true_list.extend([val.item() for val in data.y])
                pred_list.extend([val.item() for val in out.squeeze()])
                #pred_list.extend([val.item() for val in data.y])
                tumor_grade_list.extend([val for val in data.tumor_grade])
                clinical_type_list.extend([val for val in data.clinical_type])
                osmonth_list.extend([val for val in data.osmonth])
                pid_list.extend([val for val in data.p_id])
                img_list.extend([val for val in data.img_id])
            else:
                pass
        
        if plot_pred:
            #plotting.plot_pred_vs_real(df, 'OS Month (log)', 'Predicted', "Clinical Type", fl_name)
            
            label_list = [label]*len(clinical_type_list)
            df = pd.DataFrame(list(zip(pid_list, img_list, true_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list, label_list)),
                columns =["Patient ID","Image Number", 'OS Month (log)', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month", "Train Val Test"])
            
            return total_loss, df
        else:
            return total_loss

    def train_test_loop(self):
        """Training and testing occurs under this function. 
        """

        self.results =[] 

        for fold_dict in self.fold_dicts:

            best_val_loss = np.inf


            for epoch in tqdm(range(self.parser_args.epoch)):

                self.train(fold_dict)

                train_loss = self.test(fold_dict, "train_loader")
                print("Train loss: ", train_loss)
                validation_loss= self.test(fold_dict, "validation_loader")
                test_loss = self.test(fold_dict, "test_loader")

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_train_loss = train_loss
                    best_test_loss = test_loss

            train_loss, df_train = self.test(fold_dict, "train_loader", "train", "train", self.setup_args.plot_result)
            validation_loss, df_val= self.test(fold_dict, "validation_loader", "validation", "validation", self.setup_args.plot_result)
            test_loss, df_test = self.test(fold_dict, "test_loader", "test", "test", self.setup_args.plot_result)
            list_ct = list(set(df_train["Clinical Type"]))
            r2_score = r_squared_score(df_val['OS Month (log)'], df_val['Predicted'])

            if r2_score>0.7:

                df2 = pd.concat([df_train, df_val, df_test])
                df2.to_csv(f"{self.setup_args.OUT_DATA_PATH}/{self.parser_args_str}.csv", index=False)
                # print(list_ct)
                # plotting.plot_pred_vs_real_lst(df2, ['OS Month (log)']*3, ["Predicted"]*3, "Clinical Type", list_ct, parser_args_str)
                plotting.plot_pred_(df2, list_ct, self.parser_args_str)

            if (epoch % self.setup_args.print_every_epoch) == 0:
                print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}, Test loss: {test_loss:.4f}')

            print(f"For fold:  {fold_dict['fold']}")
            print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")

            self.results.append([fold_dict['fold'], best_train_loss, best_val_loss, best_test_loss])

    def save_results(self):
        """Found results are saved into CSV file
        """
        header = ["fold_number","train","validation","test"]

        train_results = []
        valid_results = []
        test_results = []

        for _,train,valid,test in self.results:
            train_results.append(train)
            valid_results.append(valid)
            test_results.append(test)

        if self.setup_args.use_fold:
            means = [["Means",statistics.mean(train_results),statistics.mean(valid_results),statistics.mean(test_results)]]
            variances = [["Variances",statistics.variance(train_results),statistics.variance(valid_results),statistics.variance(test_results)]]

        else:

            means = [["Means",train_results[0],valid_results[0],test_results[0]]]
            variances = [["Variances",train_results[0],valid_results[0],test_results[0]]]

        ff = open(self.setup_args.RESULT_PATH + str(self.setup_args.id) + '.csv', 'w')
        ff.close()

        with open(self.setup_args.RESULT_PATH + str(self.setup_args.id) + '.csv', 'w', encoding="UTF8", newline='') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            writer.writerows(self.results)

            writer.writerows(means)

            writer.writerows(variances)