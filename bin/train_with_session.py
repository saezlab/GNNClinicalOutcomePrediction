
import torch
from data_processing import OUT_DATA_PATH
import os
import argparse
import pytorch_lightning as pl
import custom_tools as custom_tools
from types import SimpleNamespace
from loss import CoxPHLoss, NegativeLogLikelihood
from trainer_tester import trainer_tester
import json


parser = argparse.ArgumentParser()

custom_tools.set_seeds(seed=42)
"""parser_args = custom_tools.general_parser()
setup_args = SimpleNamespace()
args = parser.parse_args()"""

# WCSTZwu7iVXlw-_9NvOqIw
t_args = argparse.Namespace()
json_fl = open("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/models/GATV2_NegativeLogLikelihood_month_04-12-2023/YyroGgMa_H4xn_ctP3C5Zw.json")
json_fl = open("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/models/GATV2_NegativeLogLikelihood_month_04-12-2023/FaAGmroaSN1zzkKthzk-cQ.json")
json_fl = open("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/models/GATV2_NegativeLogLikelihood_month_04-12-2023/WCSTZwu7iVXlw-_9NvOqIw.json")
json_fl = open("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/models/GATV2_NegativeLogLikelihood_fixed_dataset_13-12-2023/V05lYbfqzxjRjenrPbsplg.json")
json_fl = open("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/models/GATV2_NegativeLogLikelihood_month_04-12-2023/YyroGgMa_H4xn_ctP3C5Zw.json")
json_fl = open("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/models/GATV2_NegativeLogLikelihood_month_04-12-2023/YyroGgMa_H4xn_ctP3C5Zw.json")

t_args.__dict__.update(json.load(json_fl))
parser_args = parser.parse_args(namespace=t_args)

setup_args = SimpleNamespace()

setup_args.id = custom_tools.generate_session_id()

print(f"Session id: {setup_args.id}")

setup_args.S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
# setup_args.OUT_DATA_PATH = os.path.join(setup_args.S_PATH, "../data", "out_data", parser_args.en)
setup_args.RESULT_PATH = os.path.join(setup_args.S_PATH, "../results", "idedFiles", parser_args.en)
setup_args.PLOT_PATH = os.path.join(setup_args.S_PATH, "../plots", parser_args.en)
setup_args.MODEL_PATH = os.path.join(setup_args.S_PATH, "../models", parser_args.en)

custom_tools.create_directories([setup_args.RESULT_PATH, setup_args.PLOT_PATH, setup_args.MODEL_PATH])

setup_args.T2VT_ratio = 4
setup_args.V2T_ratio = 1
setup_args.use_fold = parser_args.fold

# This is NOT for sure, loss can change inside the class
setup_args.criterion = None
if parser_args.loss=="MSE":
    setup_args.criterion = torch.nn.MSELoss()
elif parser_args.loss=="Huber":
    setup_args.criterion = torch.nn.HuberLoss()
elif parser_args.loss == "CoxPHLoss":
    setup_args.criterion = CoxPHLoss()
elif parser_args.loss== "NegativeLogLikelihood":
    print("Loss is NegativeLogLikelihood")
    setup_args.criterion = NegativeLogLikelihood()
else:
    raise Exception("Loss function should be MSE, Huber, CoxPHLoss or NegativeLogLikelihood...")


setup_args.criterion = CoxPHLoss()
# setup_args.criterion = NegativeLogLikelihood()
setup_args.print_every_epoch = 10
setup_args.plot_result = True

# parser_args.full_training = True
parser_args.patience = 10
parser_args.dataset_name = "JacksonFischer"
# parser_args.epoch = 36
print(parser_args)
# Object can be saved if wanted
trainer_tester(parser_args, setup_args)