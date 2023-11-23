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
from eval import concordance_index
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import PNAConv 
from early_stopping import EarlyStopping

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

        print(self.device)

        self.init_folds()

        if self.parser_args.full_training:
            self.full_train_loop()
        else:
            self.train_test_loop()

        # self.save_results()

    def set_device(self):
        """Sets up the computation device for the class
        """
        self.device = custom_tools.get_device(self.parser_args.gpu_id)

    def convert_to_month(self, df_col):
        if self.parser_args.unit=="week":
            # convert to month
            return df_col/4.0
        elif self.parser_args.unit=="month":
            return df_col
        elif self.parser_args.unit=="week_lognorm":
            return np.exp(df_col)/4.0
        else:
            raise Exception("Invalid target unit... Should be week,  month, or week_lognorm")

    def init_folds(self):
        """Pulls data, creates samplers according to ratios, creates train, test and validation loaders for 
        each fold, saves them under the 'folds_dict' dictionary
        """
        # self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data"))
        # self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data/JacksonFischer/week"), "week")
        self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH,"../data/JacksonFischer", self.parser_args.unit),  self.parser_args.unit)
        print("Number of samples:", len(self.dataset))

        if self.parser_args.label == "OSMonth" or self.parser_args.loss == "CoxPHLoss":
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
        # print("Dataset:",  self.dataset)

        self.fold_dicts = []
        deg = -1

        if self.parser_args.full_training:
            # TODO: Consider bootstrapping 
            train_sampler = torch.utils.data.SubsetRandomSampler(list(range(len(self.dataset))))
            train_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, shuffle=True)
            validation_loader, test_loader = None, None
            if self.parser_args.model == "PNAConv" or "MMAConv" or "GMNConv":
                    deg = self.calculate_deg(train_sampler)

            model = self.set_model(deg)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.parser_args.lr, weight_decay=self.parser_args.weight_decay)


            fold_dict = {
                    "fold": 1,
                    "train_loader": train_loader,
                    "validation_loader": validation_loader,
                    "test_loader": test_loader,
                    "deg": deg,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler
                }

            self.fold_dicts.append(fold_dict)

        
        else:
            self.samplers = custom_tools.k_fold_ttv(self.dataset, 
                T2VT_ratio=self.setup_args.T2VT_ratio,
                V2T_ratio=self.setup_args.V2T_ratio)

            deg = -1

            for fold, train_sampler, validation_sampler, test_sampler in self.samplers:
                train_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= train_sampler)
                validation_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= validation_sampler)
                test_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= test_sampler)

                if self.parser_args.model == "PNAConv" or "MMAConv" or "GMNConv":
                    deg = self.calculate_deg(train_sampler)
                
                model = self.set_model(deg)

                optimizer = torch.optim.Adam(model.parameters(), lr=self.parser_args.lr, weight_decay=self.parser_args.weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, 'min', factor= self.parser_args.factor, patience=self.parser_args.patience, min_lr=self.parser_args.min_lr, verbose=True)

                fold_dict = {
                    "fold": fold,
                    "train_loader": train_loader,
                    "validation_loader": validation_loader,
                    "test_loader": test_loader,
                    "deg": deg,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler
                }

                self.fold_dicts.append(fold_dict)

                if not self.setup_args.use_fold:
                    break

    def calculate_deg(self, train_sampler):
        """Calcualtes deg, which is necessary for some models

        Args:
            train_sampler (_type_): Training data sampler
        """
        train_dataset = self.dataset[train_sampler.indices]
        train_loader = DataLoader(train_dataset, batch_size=self.parser_args.bs, shuffle=True)
        deg = PNAConv.get_degree_histogram(train_loader)

        return deg

    def set_model(self, deg):
        """Sets the model according to parser parameters and deg

        Args:
            deg (_type_): degree data of the graph
        """
        # print(vars(self.parser_args))#  = self.dataset.num_node_features
        self.parser_args.num_node_features = self.dataset.num_node_features
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
                    heads = self.parser_args.heads,
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
        # print(fold_dict["model"])
        for data in fold_dict["train_loader"]:  # Iterate in batches over the training dataset.

            out = fold_dict["model"](data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.

            loss = None
            if self.parser_args.loss == "CoxPHLoss":
                loss = self.setup_args.criterion(out, data.y.to(self.device), data.is_censored.to(self.device))  # Compute the loss.    
            else:
                loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.
            
            loss.backward()  # Derive gradients.
            #out_list.extend([val.item() for val in out.squeeze()])
            
            pred_list.extend([val.item() for val in out.squeeze()])
            total_loss += float(loss.item())
            fold_dict["optimizer"].step()  # Update parameters based on gradients.
            fold_dict["optimizer"].zero_grad()  # Clear gradients

        return total_loss

    def test(self, model, loader, fold, label=None, plot_pred=False):
        """Tests the model on wanted loader


        Args:
            fold_dict (dict): Holds data about the used fold
            test_on (str): Which loader to test on (train_loader, test_loader, valid_loader)
            label (_type_, optional): Label of the loader. Defaults to None.
            plot_pred (bool, optional): Should there be a plot. Defaults to False.

        Returns:
            float: total loss
        """

        # loader = fold_dict[test_on]
        model.eval()

        total_loss = 0.0
        pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list, censorship_list = [], [], [], [], [], [], [], []
        
        for data in loader:  # Iterate in batches over the training/test dataset.
            if data.y.shape[0]>1:
                out = model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.

                loss = None
                if self.parser_args.loss == "CoxPHLoss":
                    loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device), data.is_censored.to(self.device))  # Compute the loss.    
                   
                else:
                    loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.

                total_loss += float(loss.item())

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
                censorship_list.extend([val for val in data.is_censored])
                pid_list.extend([val for val in data.p_id])
                img_list.extend([val for val in data.img_id])
            else:
                pass
        
        
        if plot_pred:
            
            # label_list = [str(fold_dict["fold"]) + "-" + label]*len(clinical_type_list)
            label_list = [str(fold) + "-" + label]*len(clinical_type_list) # 
            df = pd.DataFrame(list(zip(pid_list, img_list, true_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list, censorship_list, label_list)),
                columns =["Patient ID","Image Number", 'True Value', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month", "Censored", "Fold#-Set"])
            
            return total_loss, df
        else:
            return total_loss

    def train_test_loop(self):
        """Training and testing occurs under this function. 
        """
        self.results =[] 
        # collect train/val/test predictions of all folds in all_preds_df
        all_preds_df = []
    
        for fold_dict in self.fold_dicts:
        
            best_val_loss = np.inf
            early_stopping = EarlyStopping(patience=25, verbose=True, model_path=self.setup_args.MODEL_PATH)

            print(f"########## Fold :  {fold_dict['fold']} ########## ")
            for epoch in (pbar := tqdm(range(self.parser_args.epoch), disable=False)):

                self.train(fold_dict)
                train_loss = self.test(fold_dict["model"], fold_dict["train_loader"],  fold_dict["fold"])
                validation_loss = self.test(fold_dict["model"], fold_dict["validation_loader"],  fold_dict["fold"])
                test_loss = self.test(fold_dict["model"], fold_dict["test_loader"],  fold_dict["fold"])
                """validation_loss= self.test(fold_dict, "validation_loader")
                test_loss = self.test(fold_dict, "test_loader")"""
                fold_dict["scheduler"].step(validation_loss)
                early_stopping(validation_loss, fold_dict["model"], id_file_name=self.setup_args.id, deg=self.fold_dicts[0]["deg"] if self.parser_args.model == "PNAConv" else None)
                
                pbar.set_description(f"Train loss: {train_loss:.2f} Val. loss: {validation_loss:.2f} Test loss: {test_loss:.2f} Patience: {early_stopping.counter}")

                if early_stopping.early_stop:
                    print("Early stopping the training...")
                    # break

                """if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_train_loss = train_loss
                    best_test_loss = test_loss"""
                
                # if (epoch % self.setup_args.print_every_epoch) == 0:
                #    print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}, Test loss: {test_loss:.4f}')
            # l(f"{job_id}_SD", path = "../models/best_full_training_22-11-2022", model_type = "SD", args = args, deg=deg)
            
            
            best_model = custom_tools.load_model(f"{self.setup_args.id}_SD", path = self.setup_args.MODEL_PATH, model_type = "SD", args = vars(self.parser_args), deg=self.fold_dicts[0]["deg"] if self.parser_args.model == "PNAConv" else None).to(self.device)
            
            best_train_loss, df_train = self.test(best_model, fold_dict["train_loader"], fold_dict["fold"], "train", self.setup_args.plot_result)  # type: ignore
            best_val_loss, df_val= self.test(best_model, fold_dict["validation_loader"], fold_dict["fold"], "validation", self.setup_args.plot_result)  # type: ignore
            best_test_loss, df_test = self.test(best_model, fold_dict["test_loader"], fold_dict["fold"], "test", self.setup_args.plot_result) # type: ignore

            

            if self.label_type == "regression" and self.parser_args.loss=="CoxPHLoss":
                fold_train_ci_score = concordance_index(df_train['OS Month'], df_train['Predicted'], df_train["Censored"])
                fold_val_ci_score = concordance_index(df_val['OS Month'], df_val['Predicted'], df_val["Censored"])
                fold_test_ci_score = concordance_index(df_test['OS Month'], df_test['Predicted'], df_test["Censored"])
                print(fold_train_ci_score, fold_val_ci_score, fold_test_ci_score)

            fold_tvt_preds_df = pd.concat([df_train, df_val, df_test])
            
            all_preds_df = None
            # populate all_preds_df with the first fold predictions
            if fold_dict["fold"] == 1:
                all_preds_df = fold_tvt_preds_df
            else:
                all_preds_df = pd.concat([all_preds_df, fold_tvt_preds_df])
            all_preds_df.to_csv(os.path.join(self.setup_args.RESULT_PATH, f"{self.setup_args.id}.csv"), index=False)
            
            if fold_val_ci_score > 0.60:
                all_preds_df.to_csv(os.path.join(self.setup_args.RESULT_PATH, f"{self.setup_args.id}.csv"), index=False)
            else:
                # clean the bad performing models  
                custom_tools.clean_session_files(result_fold_path=self.setup_args.RESULT_PATH, model_fold_path = self.setup_args.MODEL_PATH, model_id =self.setup_args.id, gnn_layer=self.parser_args.model)
                
            
                                
            """elif self.label_type == "regression":
                fold_val_r2_score = r_squared_score(df_val['True Value'], df_val['Predicted'])
                fold_val_mse_score = mse(df_val['True Value'], df_val['Predicted'])
                fold_val_rmse_score = rmse(df_val['True Value'], df_val['Predicted'])
                df_train['True Value'], df_train['Predicted'] = self.convert_to_month(df_train['True Value']), self.convert_to_month(df_train['Predicted'])
                df_val['True Value'],  df_val['Predicted'] = self.convert_to_month(df_val['True Value']), self.convert_to_month(df_val['Predicted'])
                df_test['True Value'], df_test['Predicted'] = self.convert_to_month(df_test['True Value']), self.convert_to_month(df_test['Predicted'])

            elif self.label_type == "classification":
                accuracy_Score = accuracy_score(df_val["True Value"], df_val['Predicted'])
                precision_Score = precision_score(df_val["True Value"], df_val['Predicted'],average="micro")   
                f1_Score = f1_score(df_val["True Value"], df_val['Predicted'],average="micro")
                
            

            
            # print(all_preds_df)
            print(f"Best val loss: {best_val_loss}, Best test loss: {best_test_loss}")
            
            if self.label_type == "regression" and self.parser_args.loss=="CoxPHLoss":
                self.results.append([fold_dict['fold'], round(best_train_loss, 4), round(best_val_loss, 4), round(best_test_loss, 4), fold_val_ci_score])

            elif self.label_type == "regression":
                self.results.append([fold_dict['fold'], round(best_train_loss, 4), round(best_val_loss, 4), round(best_test_loss, 4), fold_val_r2_score, fold_val_mse_score, fold_val_rmse_score])

            elif self.label_type == "classification":
                self.results.append([fold_dict['fold'], best_train_loss, best_val_loss, best_test_loss, accuracy_Score, precision_Score, f1_Score])"""
        
        """all_folds_val_df = all_preds_df.loc[(all_preds_df['Fold#-Set'].str[2:] == "validation")]

        
        if self.label_type == "regression" and self.parser_args.loss=="CoxPHLoss":
            all_fold_val_ci_score = concordance_index(all_folds_val_df['OS Month'], all_folds_val_df['Predicted'], all_folds_val_df["Censored"])
            print(f"All folds C index: {all_fold_val_ci_score}")

        elif self.label_type == "regression":
            all_fold_val_r2_score = r_squared_score(all_folds_val_df['True Value'], all_folds_val_df['Predicted'])
            all_fold_val_mse_score = mse(all_folds_val_df['True Value'], all_folds_val_df['Predicted'])
            all_fold_val_mae_score = mae(all_folds_val_df['True Value'], all_folds_val_df['Predicted'])
            print(f"All folds val - R2 score: {all_fold_val_r2_score}\tMSE: {all_fold_val_mse_score}\tMAE: {all_fold_val_mae_score}")
        
        if self.label_type == "regression" and self.parser_args.loss=="CoxPHLoss":
            all_preds_df.to_csv(os.path.join(self.setup_args.OUT_DATA_PATH, f"{self.setup_args.id}.csv"), index=False)

        elif  (self.label_type == "regression" and all_fold_val_r2_score>0.6):
            # plotting.plot_pred_vs_real(all_preds_df, self.parser_args.en, self.setup_args.id)
            all_preds_df.to_csv(os.path.join(self.setup_args.OUT_DATA_PATH, f"{self.setup_args.id}.csv"), index=False)
            self.save_results()
            custom_tools.save_dict_as_json(vars(self.parser_args), self.setup_args.id, self.setup_args.MODEL_PATH)
            if not self.parser_args.fold:
                custom_tools.save_model(model=self.fold_dicts[0]["model"], fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
                if self.parser_args.model == "PNAConv":
                    custom_tools.save_pickle(self.fold_dicts[0]["deg"], f"{self.setup_args.id}_deg.pckl", self.setup_args.MODEL_PATH)"""
    
    def full_train_loop(self):
        """Training and testing occurs under this function. 
        """

        self.results =[] 
        # collect train/val/test predictions of all folds in all_preds_df
        all_preds_df = []
        fold_dict = self.fold_dicts[0]    
        best_val_loss = np.inf

        print(f"Performing full training ...")
        for epoch in (pbar := tqdm(range(self.parser_args.epoch), disable=True)):

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
            plotting.plot_pred_vs_real(all_preds_df, self.parser_args.en, self.setup_args.id, full_training=True)
            all_preds_df.to_csv(os.path.join(self.setup_args.OUT_DATA_PATH, f"{self.setup_args.id}.csv"), index=False)
            self.save_results()
            custom_tools.save_dict_as_json(vars(self.parser_args), self.setup_args.id, self.setup_args.MODEL_PATH)
            if not self.parser_args.fold:
                custom_tools.save_model(model=self.fold_dicts[0]["model"], fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
                if self.parser_args.model == "PNAConv":
                    custom_tools.save_pickle(self.fold_dicts[0]["deg"], f"{self.setup_args.id}_deg.pckl", self.setup_args.MODEL_PATH)
                
    
    def save_results(self):
        """Found results are saved into CSV file
        """
        # header = ["fold number", "best train loss", "best val loss", "best test loss", "fold val r2 score", "fold val mse", "fold val rmse"]
        

        train_results = []
        valid_results = []
        test_results = []
        ci_results = []
        r2_results = []
        mse_results = []
        rmse_results = []
        accuracy_results =[] 
        precision_results =[] 
        f1_results =[] 

        
        if self.label_type == "regression" and self.parser_args.loss=="CoxPHLoss":
            for _,train,valid,test, cindex in self.results:
                train_results.append(train)
                valid_results.append(valid)
                test_results.append(test)
                ci_results.append(cindex)

        elif self.label_type == "regression":
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


    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 32 --dropout 0.0 --epoch 20 --num_of_gcn_layers 2 --num_of_ff_layers 1 --gcn_h 128 --fcl 256 --en best_n_fold_17-11-2022 --weight_decay 0.0001 --factor 0.8 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --label OSMonth --loss CoxPHLoss

    # full_training
    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 16 --dropout 0.0 --epoch 200 --num_of_gcn_layers 3 --num_of_ff_layers 2 --gcn_h 32 --fcl 512 --en best_full_training_week_15-12-2022 --weight_decay 0.0001 --factor 0.5 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --full_training --label OSMonth