import custom_tools
import os
from dataset import TissueDataset
from explainer_cli import Explainer



S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])

args  = custom_tools.load_json("../models/Y26JhVHyenOKnAO8ljdngA.json")
deg = custom_tools.load_pickle("../models/Y26JhVHyenOKnAO8ljdngA_deg")
model = custom_tools.load_model("Y26JhVHyenOKnAO8ljdngA_SD", path = "../models", model_type = "SD", args = args, deg=deg)

dataset = TissueDataset(os.path.join(S_PATH,"../data"))


dataset = dataset.shuffle()

num_of_train = int(len(dataset)*0.80)
num_of_val = int(len(dataset)*0.10)

train_dataset = dataset[:num_of_train]
validation_dataset = dataset[num_of_train:num_of_train+num_of_val]
test_dataset = dataset[num_of_train+num_of_val:]

explainer = Explainer(model, test_dataset)

explainer.explain_by_gnnexplainer()