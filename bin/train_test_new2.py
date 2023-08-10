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
from evaluation_metrics import r_squared_score, mse, rmse, mae
import custom_tools as custom_tools
import csv
import statistics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import shutil
from sweep_config import sweep_configuration
from data_processing import OUT_DATA_PATH
import pytorch_lightning as pl
from types import SimpleNamespace

import warnings
warnings.filterwarnings("ignore")

wandb.login()

def sweep(config = None, project_name = None):
    r"""
    Runs a hyperparameter sweep for a given configuration.

    Args:
        - `config` (dict, optional): A dictionary containing hyperparameters 
            and their values to be swept. Defaults to `"None"`.

        - `project_name` (str, optional): The name of the project to log 
            the sweep under. Defaults to `"None"`.

    Returns:
        `wandb.sweep object`: A sweep object that can be used to start a 
            sweep run with the specified configuration.
    """
    return wandb.sweep(sweep=config, project=project_name)


class trainer_tester:


    def __init__(self, setup_args) -> None:
        """Manager of processes under train_tester, main goel is to show the processes squentially

        Args:
            parser_args (Namespace): Holds the arguments that came from parsing CLI
            setup_args (Namespace): Holds the arguments that came from setup
        """
        self.setup_args = setup_args
        self.set_device()

        

        # self.save_results()

        # Create a new sweep with the specified configuration and project name.
        sweep_id = sweep(config = sweep_configuration, project_name = 'FUTON')

        # Set the WANDB_CACHE_DIR environment variable to the specified directory path.
        wand_dir = 'cache/'
        os.environ['WANDB_CACHE_DIR'] = wand_dir

        # Set the name of the run using the specified convolutional layer and sweep ID.
        run_name = "{}".format(sweep_id)

        # Initialize a new WandB run with the specified run name and project name, and save the configuration.
        wand_run = wandb.init(name=run_name, project='FUTON', config=sweep_configuration, dir=wand_dir)

        # Set the path to the directory where the results for the current run will be saved.
        self.run_folder_name = wandb.run.dir

        # Get the hyperparameters from the configuration.

        self.unit = wand_run.config["parameters"]["unit"]["values"][0]
        self.label = wand_run.config["parameters"]["label"]["values"][0]
        self.model = wand_run.config["parameters"]["model"]["values"][0]
        self.bs = wand_run.config["parameters"]["bs"]["values"][0]
        self.lr = wand_run.config["parameters"]["lr"]["values"][0]
        self.weight_decay = wand_run.config["parameters"]["weight_decay"]["values"][0]
        self.num_of_gcn_layers = wand_run.config["parameters"]["num_of_gcn_layers"]["values"][0]
        self.num_of_ff_layers = wand_run.config["parameters"]["num_of_ff_layers"]["values"][0]
        self.gcn_h = wand_run.config["parameters"]["gcn_h"]["values"][0]
        self.fcl = wand_run.config["parameters"]["fcl"]["values"][0]
        self.dropout = wand_run.config["parameters"]["dropout"]["values"][0]
        self.aggregators = wand_run.config["parameters"]["aggregators"]["values"][0]
        self.scalers = wand_run.config["parameters"]["scalers"]["values"][0]
        self.heads = wand_run.config["parameters"]["heads"]["values"][0]
        self.epoch = wand_run.config["parameters"]["epoch"]["values"][0]
        self.en = wand_run.config["parameters"]["en"]["values"][0]
        self.full_training = wand_run.config["parameters"]["full_training"]["values"][0]
        self.fold = wand_run.config["parameters"]["fold"]["values"][0]
        self.loss = wand_run.config["parameters"]["loss"]["values"][0]

        self.init_folds()

        if self.full_training:
            self.full_train_loop()
        else:
            self.train_test_loop()
    


    def set_device(self):
        """Sets up the computation device for the class
        """
        self.device = custom_tools.get_device()

    def convert_to_month(self, df_col):
        if self.unit=="week":
            # convert to month
            return df_col/4.0
        elif self.unit=="month":
            return df_col
        elif self.unit=="week_lognorm":
            return np.exp(df_col)/4.0
        else:
            raise Exception("Invalid target unit... Should be week,  month, or week_lognorm")

    def init_folds(self):
        """Pulls data, creates samplers according to ratios, creates train, test and validation loaders for 
        each fold, saves them under the 'folds_dict' dictionary
        """
        # self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data"))
        # self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data/JacksonFischer/week"), "week")
        print('unit:',self.unit)
        self.dataset = TissueDataset(os.path.join(getattr(self.setup_args, 'S_PATH', ''), "../data/JacksonFischer", self.unit), self.unit)
        print("Number of samples:", len(self.dataset))

        if self.label == "OSMonth":
            self.label_type = "regression"
            self.num_classes = 1

        elif self.label == "treatment":
            self.label_type = "classification"
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()
            self.dataset.data.y = self.dataset.data.clinical_type
            self.unique_classes = set(self.dataset.data.clinical_type)
            self.num_classes = len(self.unique_classes)


        # WARN CURRENTLY DOESN'T PULL THE DATA NOT WORKING
        elif self.label == "DiseaseStage":
            self.label_type = "classification"
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()

        elif self.label == "grade":
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

        self.fold_dicts = []
        deg = -1

        if self.full_training:
            # TODO: Consider bootstrapping 
            train_sampler = torch.utils.data.SubsetRandomSampler(list(range(len(self.dataset))))
            train_loader = DataLoader(self.dataset, batch_size=self.bs, shuffle=True)
            validation_loader, test_loader = None, None
            if self.model == "PNAConv":
                    deg = self.calculate_deg(train_sampler)

            model = self.set_model(deg)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


            fold_dict = {
                    "fold": 1,
                    "train_loader": train_loader,
                    "validation_loader": validation_loader,
                    "test_loader": test_loader,
                    "deg": deg,
                    "model": model,
                    "optimizer": optimizer
                }

            self.fold_dicts.append(fold_dict)

        
        else:
            print('setup:', self.setup_args)
            self.samplers = custom_tools.k_fold_ttv(self.dataset, 
                T2VT_ratio=self.setup_args.T2VT_ratio,
                V2T_ratio=self.setup_args.V2T_ratio)


            deg = -1

            for fold, train_sampler, validation_sampler, test_sampler in self.samplers:
                train_loader = DataLoader(self.dataset, batch_size=self.bs, sampler= train_sampler)
                validation_loader = DataLoader(self.dataset, batch_size=self.bs, sampler= validation_sampler)
                test_loader = DataLoader(self.dataset, batch_size=self.bs, sampler= test_sampler)

                if self.model == "PNAConv":
                    deg = self.calculate_deg(train_sampler)
                
                model = self.set_model(deg)

                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                # scheduler = ReduceLROnPlateau(optimizer, 'min', factor= self.parser_args.factor, patience=self.parser_args.patience, min_lr=self.parser_args.min_lr, verbose=True)

                fold_dict = {
                    "fold": fold,
                    "train_loader": train_loader,
                    "validation_loader": validation_loader,
                    "test_loader": test_loader,
                    "deg": deg,
                    "model": model,
                    "optimizer": optimizer,
                    # "schedular": scheduler
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
                    type = self.model,
                    num_node_features = self.dataset.num_node_features, ####### LOOOOOOOOK HEREEEEEEEEE
                    num_gcn_layers=self.num_of_gcn_layers, 
                    num_ff_layers=self.num_of_ff_layers, 
                    gcn_hidden_neurons=self.gcn_h, 
                    ff_hidden_neurons=self.fcl, 
                    dropout=self.dropout,
                    aggregators=self.aggregators,
                    scalers=self.scalers,
                    deg = deg, # Comes from data not hyperparameter
                    num_classes = self.num_classes,
                    heads = self.heads,
                    label_type = self.label_type
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
        # WARN Disabled it but IDK what it does
        #out_list = []
        for data in fold_dict["train_loader"]:  # Iterate in batches over the training dataset.
            
            print('fold_dict["model"]:', fold_dict["model"])
            print('(data.x:', data.x.shape)


            out = fold_dict["model"](data.x.to(self.device), data.edge_index.to(self.device), 
                                     data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.
            # print(out)
            loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.
        
            loss.backward()  # Derive gradients.
            #out_list.extend([val.item() for val in out.squeeze()])
            
            pred_list.extend([val.item() for val in data.y])

            total_loss += float(loss.item())
            fold_dict["optimizer"].step()  # Update parameters based on gradients.
            fold_dict["optimizer"].zero_grad()  # Clear gradients

        return total_loss

    def test(self, fold_dict, test_on: str,label=None, plot_pred=False):
        """Tests the model on wanted loader

        Args:
            fold_dict (dict): Holds data about the used fold
            test_on (str): Which loader to test on (train_loader, test_loader, valid_loader)
            label (_type_, optional): Label of the loader. Defaults to None.
            plot_pred (bool, optional): Should there be a plot. Defaults to False.

        Returns:
            float: total loss
        """

        loader = fold_dict[test_on]
        fold_dict["model"].eval()

        total_loss = 0.0
        pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list = [], [], [], [], [], [], []
        
        for data in loader:  # Iterate in batches over the training/test dataset.
            """print(data)
            print(data.y)
            print(data.x)
            print(data.clinical_type)"""
            if data.y.shape[0]>1:
                out = fold_dict["model"](data.x.to(self.device), data.edge_index.to(self.device), 
                                         data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.
                loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.
                
                total_loss += float(loss.item())
                # print("test", data.y)
                # print("test out", out.squeeze())
                true_list.extend([round(val.item(),6) for val in data.y])
                # WARN Correct usage of "max" ?
                if self.label_type == "regression":
                    pred_list.extend([val.item() for val in out.squeeze()])
                else:
                    pred_list.extend([custom_tools.argmax(val) for val in out.squeeze()])
                #pred_list.extend([val.item() for val in data.y])
                tumor_grade_list.extend([val.item() for val in data.tumor_grade])
                clinical_type_list.extend([val for val in data.clinical_type])
                osmonth_list.extend([val.item() for val in data.osmonth])
                pid_list.extend([val for val in data.p_id])
                img_list.extend([val for val in data.img_id])
            else:
                pass
        
        if plot_pred:
            #plotting.plot_pred_vs_real(df, 'OS Month (log)', 'Predicted', "Clinical Type", fl_name)
            
            label_list = [str(fold_dict["fold"]) + "-" + label]*len(clinical_type_list)
            df = pd.DataFrame(list(zip(pid_list, img_list, true_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list, label_list)),
                columns =["Patient ID","Image Number", 'True Value', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month", "Fold#-Set"])
            
            return total_loss, df
        else:
            return total_loss

    def train_test_loop(self, dataset="FUTON_1"):
        """Training and testing occurs under this function. 
        """

        self.results =[] 
        # collect train/val/test predictions of all folds in all_preds_df
        all_preds_df = []

        best_thresholds = []


        for fold_dict in self.fold_dicts:
        
            best_val_loss = np.inf

            print(f"########## Fold :  {fold_dict['fold']} ########## ")
            for epoch in (pbar := tqdm(range(self.epoch), disable=True)):

                self.train(fold_dict)

                train_loss = self.test(fold_dict, "train_loader")
                pbar.set_description(f"Train loss: {train_loss}")
    
                validation_loss= self.test(fold_dict, "validation_loader")
                test_loss = self.test(fold_dict, "test_loader")

                wandb.log({"train_loss": train_loss}, step=epoch)
                wandb.log({"validation_loss": validation_loss}, step=epoch)
                wandb.log({"test_loss": test_loss}, step=epoch)
                wandb.log({"epoch": epoch})

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_train_loss = train_loss
                    best_test_loss = test_loss
                
                if (epoch % self.setup_args.print_every_epoch) == 0:
                    print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}, Test loss: {test_loss:.4f}')


            train_loss, df_train = self.test(fold_dict, "train_loader", "train", self.setup_args.plot_result)
            validation_loss, df_val= self.test(fold_dict, "validation_loader", "validation", self.setup_args.plot_result)
            test_loss, df_test = self.test(fold_dict, "test_loader", "test", self.setup_args.plot_result)
            list_ct = list(set(df_train["Clinical Type"]))

            artifact = wandb.Artifact("best_model_{}".format(self.model),
                                    type="model", metadata=dict(model_name=self.model,
                                                                dataset="FUTON_1",
                                                                ))

            if self.label_type == "regression":
                fold_val_r2_score = r_squared_score(df_val['True Value'], df_val['Predicted'])
                df_train['True Value'] = self.convert_to_month(df_train['True Value'])
                df_train['Predicted'] = self.convert_to_month(df_train['Predicted'])
                df_val['True Value'] = self.convert_to_month(df_val['True Value'])
                df_val['Predicted'] = self.convert_to_month(df_val['Predicted'])
                df_test['True Value'] = self.convert_to_month(df_test['True Value'])
                df_test['Predicted'] = self.convert_to_month(df_test['Predicted'])

                fold_val_mse_score = mse(df_val['True Value'], df_val['Predicted'])
                fold_val_rmse_score = rmse(df_val['True Value'], df_val['Predicted'])

                results_best = {"model": self.model, "fold_val_r2_score": fold_val_r2_score, 
                                "fold_val_mse_score":fold_val_mse_score,
                                "fold_val_rmse_score":fold_val_rmse_score}

            elif self.label_type == "classification":
                accuracy_Score = accuracy_score(df_val["True Value"], df_val['Predicted'])
                precision_Score = precision_score(df_val["True Value"], df_val['Predicted'],average="micro")   
                f1_Score = f1_score(df_val["True Value"], df_val['Predicted'],average="micro")     

                results_best = {"model": self.model, "accuracy_Score": accuracy_Score, 
                                "precision_Score":precision_Score, "f1_Score": f1_Score}         
                
            fold_tvt_preds_df = pd.concat([df_train, df_val, df_test])

            # populate all_preds_df with the first fold predictions
            if fold_dict["fold"] == 1:
                all_preds_df = fold_tvt_preds_df
            else:
                all_preds_df = pd.concat([all_preds_df, fold_tvt_preds_df])


            # print(all_preds_df)
            print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")
            
            if self.label_type == "regression": 
                # self.results.append([fold_dict['fold'], best_train_loss, best_val_loss, best_test_loss, r2_score, mse_score, rmse_score])
                self.results.append([fold_dict['fold'], round(best_train_loss, 4), round(best_val_loss, 4), round(best_test_loss, 4), fold_val_r2_score, fold_val_mse_score, fold_val_rmse_score])
                # val_r2_score = r_squared_score(all_folds_val_df['True Value'], val_df['Predicted'])
            

            elif self.label_type == "classification":
                self.results.append([fold_dict['fold'], best_train_loss, best_val_loss, best_test_loss, accuracy_Score, precision_Score, f1_Score])
                val_r2_score = 0


            
        
        all_folds_val_df = all_preds_df.loc[(all_preds_df['Fold#-Set'].str[2:] == "validation")]
        all_fold_val_r2_score = r_squared_score(all_folds_val_df['True Value'], all_folds_val_df['Predicted'])
        all_fold_val_mse_score = mse(all_folds_val_df['True Value'], all_folds_val_df['Predicted'])
        all_fold_val_mae_score = mae(all_folds_val_df['True Value'], all_folds_val_df['Predicted'])
    
        print(f"All folds val - R2 score: {all_fold_val_r2_score}\tMSE: {all_fold_val_mse_score}\tMAE: {all_fold_val_mae_score}")
    
        df_results = pd.DataFrame.from_dict(results_best)
        if not os.path.exists("results"):
            os.makedirs("results")
        df_results.to_csv("results/best_result_{}.csv".format(self.model))
        artifact.add_file("results/best_result_{}.csv".format(self.model))

        table = wandb.Table(dataframe=df_results)
        artifact.add(table, "results_{}".format(self.model))

        table2 = wandb.Table(columns=["thresholds"])
        for i in range(len(best_thresholds)):
            table2.add_data(best_thresholds[i])
        artifact.add(table2, "thresholds_{}".format(self.model))
        artifact.add_file("saved_models/best_model_{}.pt".format(self.model))

        wandb.log_artifact(artifact)
        wandb.finish()

        original_path = self.run_folder_name
        dir_name = os.path.dirname(original_path)

        new_path = dir_name + "/"
        if  (self.label_type == "regression" and all_fold_val_r2_score<0.6):
            print('Removing bad result logs...')
            shutil.rmtree(new_path)

        
        if  (self.label_type == "regression" and all_fold_val_r2_score>0.6):
            plotting.plot_pred_vs_real(all_preds_df, self.en, self.setup_args.id)
            all_preds_df.to_csv(os.path.join(self.setup_args.OUT_DATA_PATH, f"{self.setup_args.id}.csv"), index=False)
            self.save_results()
            custom_tools.save_dict_as_json(self.setup_args.id, self.setup_args.MODEL_PATH)
            if not self.fold:
                custom_tools.save_model(model=self.fold_dicts[0]["model"], fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
                if self.model == "PNAConv":
                    custom_tools.save_pickle(self.fold_dicts[0]["deg"], f"{self.setup_args.id}_deg.pckl", self.setup_args.MODEL_PATH)
    
        
    
    def full_train_loop(self):
        """Training and testing occurs under this function. 
        """

        self.results =[] 
        # collect train/val/test predictions of all folds in all_preds_df
        all_preds_df = []
        fold_dict = self.fold_dicts[0]    
        best_val_loss = np.inf

        print(f"Performing full training ...")
        for epoch in (pbar := tqdm(range(self.epoch), disable=True)):

            self.train(fold_dict)

            train_loss = self.test(fold_dict, "train_loader")
            pbar.set_description(f"Train loss: {train_loss}")

            if (epoch % self.setup_args.print_every_epoch) == 0:
                print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}')


        train_loss, df_train = self.test(fold_dict, "train_loader", "train", self.setup_args.plot_result)

        if self.label_type == "regression":
            r2_score = r_squared_score(df_train['True Value'], df_train['Predicted'])
            mse_score = mse(df_train['True Value'], df_train['Predicted'])
            rmse_score = rmse(df_train['True Value'], df_train['Predicted'])

        elif self.label_type == "classification":
            accuracy_Score = accuracy_score(df_train["True Value"], df_train['Predicted'])
            precision_Score = precision_score(df_train["True Value"], df_train['Predicted'],average="micro")   
            f1_Score = f1_score(df_train["True Value"], df_train['Predicted'],average="micro")    

        all_preds_df = df_train
        
        if self.label_type == "regression": 
            self.results.append([fold_dict['fold'], round(100000, 4), round(100000, 4), round(100000, 4), r2_score, mse_score, rmse_score])
        
        elif self.label_type == "classification":
            self.results.append([fold_dict['fold'], best_train_loss, best_val_loss, best_test_loss, accuracy_Score, precision_Score, f1_Score])


        # print("All folds val r2 score:", all_fold_val_r2_score)
    
        if  (self.label_type == "regression"):
            plotting.plot_pred_vs_real(all_preds_df, self.en, self.setup_args.id, full_training=True)
            all_preds_df.to_csv(os.path.join(self.setup_args.OUT_DATA_PATH, f"{self.setup_args.id}.csv"), index=False)
            self.save_results()
            custom_tools.save_dict_as_json(self.setup_args.id, self.setup_args.MODEL_PATH)
            if not self.fold:
                custom_tools.save_model(model=self.fold_dicts[0]["model"], fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
                if self.model == "PNAConv":
                    custom_tools.save_pickle(self.fold_dicts[0]["deg"], f"{self.setup_args.id}_deg.pckl", self.setup_args.MODEL_PATH)
                
    
    def save_results(self):
        """Found results are saved into CSV file
        """
        header = ["fold number", "best train loss", "best val loss", "best test loss", "fold val r2 score", "fold val mse", "fold val rmse"]
        

        train_results = []
        valid_results = []
        test_results = []
        r2_results = []
        mse_results = []
        rmse_results = []
        accuracy_results =[] 
        precision_results =[] 
        f1_results =[] 

        if self.label_type == "regression":
            for _,train,valid,test,r2,mse,rmse in self.results:
                train_results.append(train)
                valid_results.append(valid)
                test_results.append(test)

                r2_results.append(r2)
                mse_results.append(mse)
                rmse_results.append(rmse)

        elif self.label_type == "classification":
            for _,train,valid,test,accuracy,precision,f1 in self.results:
                train_results.append(train)
                valid_results.append(valid)
                test_results.append(test)

                accuracy_results.append(accuracy)
                precision_results.append(precision)
                f1_results.append(f1)

        if self.label_type == "regression":

            header = ["fold_number","train","validation","test","r2","mse","rmse"]

            if self.setup_args.use_fold:
                means = [["Mean", round(statistics.mean(train_results), 4), round(statistics.mean(valid_results), 4), round(statistics.mean(test_results), 4), statistics.mean(r2_results), statistics.mean(mse_results),statistics.mean(rmse_results)]]
                variances = [["Variance", round(statistics.variance(train_results), 4) ,round(statistics.variance(valid_results), 4), round(statistics.variance(test_results), 4), statistics.variance(r2_results),statistics.variance(mse_results),statistics.variance(rmse_results)]]

            else:

                means = [["Mean", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4),r2_results[0],mse_results[0],rmse_results[0]]]
                variances = [["Variance", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4), r2_results[0],mse_results[0],rmse_results[0]]]
        
        elif self.label_type == "classification":

            header = ["fold_number","train","validation","accuracy","precision","f1"]
            if self.setup_args.use_fold:
                means = [["Mean", round(statistics.mean(train_results), 4), round(statistics.mean(valid_results), 4), round(statistics.mean(test_results), 4), statistics.mean(accuracy_results), statistics.mean(precision_results), statistics.mean(f1_results)]]
                variances = [["Variance", round(statistics.variance(train_results), 4) ,round(statistics.variance(valid_results), 4), round(statistics.variance(test_results), 4), statistics.variance(accuracy_results), statistics.mean(precision_results), statistics.mean(f1_results)]]

            else:

                means = [["Mean", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4),accuracy_results[0],precision_results[0],f1_results[0]]]
                variances = [["Variance", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4), accuracy_results[0],precision_results[0],f1_results[0]]]

        ff = open(os.path.join(self.setup_args.RESULT_PATH, f"{str(self.setup_args.id)}.csv"), 'w')
        ff.close()

        with open(os.path.join(self.setup_args.RESULT_PATH, f"{str(self.setup_args.id)}.csv"), 'w', encoding="UTF8", newline='') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            writer.writerows(self.results)

            writer.writerows(means)

            writer.writerows(variances)



pl.seed_everything(42)
setup_args = SimpleNamespace()

setup_args.id = custom_tools.generate_session_id()

print(f"Session id: {setup_args.id}")


tt_instance = trainer_tester(setup_args)

setup_args.S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
setup_args.OUT_DATA_PATH = os.path.join(setup_args.S_PATH, "../data", "out_data", tt_instance.en)
setup_args.RESULT_PATH = os.path.join(setup_args.S_PATH, "../results", "idedFiles", tt_instance.en)
setup_args.PLOT_PATH = os.path.join(setup_args.S_PATH, "../plots", tt_instance.en)
setup_args.MODEL_PATH = os.path.join(setup_args.S_PATH, "../models", tt_instance.en)

custom_tools.create_directories([setup_args.OUT_DATA_PATH, setup_args.RESULT_PATH, setup_args.PLOT_PATH, setup_args.MODEL_PATH])

setup_args.T2VT_ratio = 4
setup_args.V2T_ratio = 1
setup_args.use_fold = tt_instance.fold


# This is NOT for sure, loss can change inside the class
setup_args.criterion = None
if tt_instance.loss=="MSE":
    setup_args.criterion = torch.nn.MSELoss()
elif tt_instance.loss=="Huber":
    setup_args.criterion = torch.nn.HuberLoss()
else:
    setup_args.criterion = torch.nn.MSELoss()

setup_args.print_every_epoch = 10
setup_args.plot_result = True

device = custom_tools.get_device()


# Object can be saved if wanted
#trainer_tester()


sweep_id = sweep(config = sweep_configuration)

#trainer_tester_instance = trainer_tester()
#run_function = trainer_tester_instance.train_test_loop(dataset="FUTON_1")

def run_function():
    
    return trainer_tester(setup_args)


wandb.agent(sweep_id, function=run_function, count=1)

