
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
import warnings

warnings.filterwarnings("ignore")

custom_tools.set_seeds(seed=42, deterministic=True)
parser = argparse.ArgumentParser()
parser.add_argument(
        '--dataset_name',
        type= str,
        metavar='F',
        required=True,
        help='Dataset name')
parser.add_argument(
        '--full_training',
        type= bool,
        action=argparse.BooleanOptionalAction,
        default= False,
        metavar='F',
        help='Perform full_training, default: --no-full_training (--full_training, --no-full_training),')
parser.add_argument(
        '--model_name',
        type= str,
        default= None,
        metavar='N',
        help='Use this name instead of random session id')

args = vars(parser.parse_args())
# args = vars(parser.parse_args())

dataset_name = args["dataset_name"]

# WCSTZwu7iVXlw-_9NvOqIw
t_args = argparse.Namespace()
"""json_fl = open("../models/GATV2_NegativeLogLikelihood_month_04-12-2023/YyroGgMa_H4xn_ctP3C5Zw.json")
json_fl = open("../models/GATV2_NegativeLogLikelihood_month_04-12-2023/FaAGmroaSN1zzkKthzk-cQ.json")
json_fl = open("../models/GATV2_NegativeLogLikelihood_month_04-12-2023/WCSTZwu7iVXlw-_9NvOqIw.json")
json_fl = open("../models/GATV2_NegativeLogLikelihood_fixed_dataset_13-12-2023/V05lYbfqzxjRjenrPbsplg.json")"""
# json_fl = open("../models/GATV2_NegativeLogLikelihood_month_04-12-2023/YyroGgMa_H4xn_ctP3C5Zw.json")
# gjqwQWOr25LssZF06ATdvg
# json_fl = open("../models/METABRIC_GATV2_CoxPHLoss_10_fold_gpu_14-04-2024/FuX5ML1-b9DagMwP7wWncQ.json")
# json_fl = open(f"../models/METABRIC_GATV2_CoxPHLoss_10_fold_gpusaez_14-04-2024/7r2YyfyWaEvKkZedb7E-hw.json")
json_fl = open(f"../models/{dataset_name}/model_hyperparams.json")
t_args.__dict__.update(json.load(json_fl))
# print("t_args", t_args)
parser_args = parser.parse_args(namespace=t_args)

# json_fl = open(f"../models/{dataset_name}/model_hyperparams.json")
# json_fl = open("../models/JacksonFischer_Final/FaX6TFVflduRtbuFdWSkYA.json")
# json_fl = open("../models/JacksonFischer_Final/O3CFsgYHQzTq__95Ozw_dQ.json")



args = vars(parser.parse_args())
full_training = args['full_training']
model_name = args['model_name']

setup_args = SimpleNamespace()
setup_args.id = custom_tools.generate_session_id()
if model_name:
    parser_args.id = model_name
    setup_args.id = model_name
    
setup_args.S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
setup_args.RESULT_PATH = os.path.join(setup_args.S_PATH, "../results", "idedFiles", parser_args.en)
setup_args.PLOT_PATH = os.path.join(setup_args.S_PATH, "../plots", parser_args.en)
setup_args.MODEL_PATH = os.path.join(setup_args.S_PATH, "../models", parser_args.en)
custom_tools.create_directories([setup_args.RESULT_PATH, setup_args.PLOT_PATH, setup_args.MODEL_PATH])
setup_args.use_fold = parser_args.fold

print(f"Session id: {setup_args.id}")

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

# parser_args.en = "JacksonFischer"

setup_args.criterion = CoxPHLoss()
parser_args.loss = "CoxPHLoss"
# setup_args.criterion = NegativeLogLikelihood()
setup_args.print_every_epoch = 10
setup_args.plot_result = True

parser_args.full_training = full_training
parser_args.patience = 10
# parser_args.dataset_name = "METABRIC"
parser_args.epoch = 100
print(parser_args)
print(setup_args)
# Object can be saved if wanted
trainer_tester(parser_args, setup_args)

# python train_with_session.py --dataset_name JacksonFischer --full_training --model_name JF