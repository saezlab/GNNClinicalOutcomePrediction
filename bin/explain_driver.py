import custom_tools
import os




args  = custom_tools.load_json("../models/Y26JhVHyenOKnAO8ljdngA.json")
deg = custom_tools.load_pickle("../models/Y26JhVHyenOKnAO8ljdngA_deg")
model = custom_tools.load_model("Y26JhVHyenOKnAO8ljdngA_SD", path = "../models", model_type = "SD", args = args, deg=deg)