

import torch
from data_processing import OUT_DATA_PATH
import os
import pytorch_lightning as pl
import custom_tools as custom_tools
from types import SimpleNamespace
from trainer_tester import trainer_tester




pl.seed_everything(42)
parser_args = custom_tools.general_parser()
setup_args = SimpleNamespace()

# WARN is this correct usage?
setup_args.id = custom_tools.generate_session_id

setup_args.S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
setup_args.OUT_DATA_PATH = os.path.join(setup_args.S_PATH, "../data", "out_data")
setup_args.S_PATH = os.path.dirname(__file__)

setup_args.T2VT_ratio = 4
setup_args.V2T_ratio = 1
setup_args.use_fold = False

setup_args.criterion = torch.nn.MSELoss()
setup_args.print_every_epoch = 5
setup_args.plot_result = True

device = custom_tools.get_device()


# Object can be saved if wanted
trainer_tester(parser_args, setup_args)








