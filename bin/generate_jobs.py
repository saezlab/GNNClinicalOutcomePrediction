import itertools as it
from analysis_scripts import calculate_all_reg_scores
from pathlib import Path
from sklearn.utils import shuffle
import random
from datetime import date
import os
import csv
from custom_tools import load_json
today = date.today()
d1 = today.strftime("%d-%m-%Y")


def generate_generic_job_commands():
    job_id = f"PNA_training_{d1}"


    config = {
        # "model": ["GCN", "GATConv", "TransformerConv", "PNAConv"],
        "model": ["PNAConv"],
        "lr": [0.1, 0.01, 0.001, 0.0001],
        "bs": [16, 32, 64],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "epoch": [50, 100, 200],
        "num_of_gcn_layers": [2,3], # 
        "num_of_ff_layers": [1,2], # 
        "gcn_h": [16, 32, 64, 128],
        "fcl": [64, 128, 256, 512],
        "weight_decay": [0.1, 0.001, 0.0001, 3e-6, 1e-5],
        "factor": [0.5, 0.8, 0.2],
        "patience": [5, 10, 20],
        "min_lr": [0.00002, 0.0001],
        "aggregators": ["min", "max", "sum","mean", "sum max"], # ARBTR Find references
        "scalers": ["identity","amplification"], # ARBTR Find references
        "en": [job_id],
        "no-fold": [""]
    }



    out_path = f"./jobs/{job_id}"
    result_path = f"{out_path}/results"


    Path(out_path).mkdir(parents=True, exist_ok=True)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    all_jobs_f = open(f"{out_path}/all_runs.sh", "w")

    random.seed = 42
    # my_dict={'A':['D','E'],'B':['F','G','H'],'C':['I','J']}
    allNames = sorted(config)
    print(allNames)
    combinations = list(it.product(*(config[Name] for Name in allNames))) # CREATIGN COMBINATIONS
    # print(list(combinations)[0])
    num_of_combs = len(combinations)
    print(num_of_combs)
    shuffled_experiments = list(range(num_of_combs)) # SHUFFLE THE COMBINATIONS
    random.shuffle(shuffled_experiments)


    number_of_runs = 1000
    count=1

    all_jobs_f.write(f"sbatch --job-name={job_id}_{count} -p gpu --gres=gpu:1 --mem=2g  -n 1 --time=7-00:00:00 --output=results/output_{count} \"{count}_{number_of_runs}.sh\"\nsleep 1\n")
    job_f = open(f"{out_path}/{count}_{number_of_runs}.sh", "w")
    job_f.writelines("#!/bin/sh\n")

    # #!/bin/sh
    for i in range(1, 100001):
        hyper_param_ind = shuffled_experiments[i]
        command_line = "python ../../train_test_controller.py "+ " ".join([f"--{param} "+ str(combinations[hyper_param_ind][allNames.index(param)]) for param in allNames]) # JUST DO THIS
        
        if i%number_of_runs == 0:
            count+=1
            job_f.close()
            all_jobs_f.write(f"sbatch --job-name={job_id}_{count} -p gpu --gres=gpu:1 --mem=2g -n 1 --time=7-00:00:00 --output=results/output_{count} \"{count}_{number_of_runs}.sh\"\nsleep 1\n")
            job_f = open(f"{out_path}/{count}_{number_of_runs}.sh", "w")
            job_f.write("#!/bin/sh\n")
            job_f.write(command_line+"\n")
        else:
            job_f.write(command_line+"\n")
            

    job_f.close()
    all_jobs_f.close()

def perform_k_fold_on_best_models(experiment_name, top_n=10):
    experiment_name = f"{experiment_name}_{d1}"

    out_path = f"./jobs/{experiment_name}"
    result_path = f"{out_path}/results"

    Path(out_path).mkdir(parents=True, exist_ok=True)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    all_jobs_f = open(f"{out_path}/all_runs.sh", "w")

    random.seed = 42

    df_results = calculate_all_reg_scores(["/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/data/out_data/PNA_training_12-10-2022"])

    number_of_runs = 10
    count=1

    all_jobs_f.write(f"sbatch --job-name={experiment_name}_{count} -p gpu --gres=gpu:1 --mem=2g  -n 1 --time=7-00:00:00 --output=results/output_{count} \"{count}_{number_of_runs}.sh\"\nsleep 1\n")
    job_f = open(f"{out_path}/{count}_{number_of_runs}.sh", "w")
    job_f.writelines("#!/bin/sh\n")

    # #!/bin/sh
    i = 0
    for ind, row in df_results[:top_n].iterrows():
        i += 1
        exp_name = row["experiment name"]
        j_id = row["job id"]
        
        try:
            hyper_param_dict = load_json(os.path.join(f"../models/{exp_name}/{j_id}.json"))
            # print(j_id, hyper_param_dict["scalers"], type(hyper_param_dict["scalers"]))
        
        except FileNotFoundError:
            print(f"{j_id}.json file does not exist!")
            pass
        hyper_param_dict["en"] = experiment_name
        exluded_params = {"aggregators", "scalers", "str", "fold", "num_node_features"}
        command_line = "python ../../train_test_controller.py "+ " ".join([f"--{param} "+ str(hyper_param_dict[param]) for param in (hyper_param_dict.keys()-exluded_params)]) # JUST DO THIS
        command_line += " --fold"
        if hyper_param_dict["model"]=="PNAConv":
            command_line += " --aggregators "+ " ".join(hyper_param_dict["aggregators"])
            command_line += " --scalers "+ " ".join(hyper_param_dict["scalers"])
        
        if i%number_of_runs == 0:
            count+=1
            job_f.close()
            all_jobs_f.write(f"sbatch --job-name={experiment_name}_{count} -p gpu --gres=gpu:1 --mem=2g -n 1 --time=7-00:00:00 --output=results/output_{count} \"{count}_{number_of_runs}.sh\"\nsleep 1\n")
            job_f = open(f"{out_path}/{count}_{number_of_runs}.sh", "w")
            job_f.write("#!/bin/sh\n")
            job_f.write(command_line+"\n")
        else:
            job_f.write(command_line+"\n")
            

    job_f.close()
    all_jobs_f.close()

perform_k_fold_on_best_models("best_n_fold", top_n=200)