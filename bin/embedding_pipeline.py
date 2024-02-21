
import os
import plotting
import embeddings
import custom_tools
from dataset import TissueDataset


def generate_embedding_plots(experiment_name, job_id, mode="FC"):
    # Read json file
    # job_id = "5rBzqSUcY3fkBsmCtnU3Mw"
    # experiment_name = "PNAConv_CoxPHLoss_month_24-11-2023"
    # job_id = "WQ-xPbLcP0BqY6wmWF1VPQ"
    device =  custom_tools.get_device()
    args  = custom_tools.load_json(f"../models/{experiment_name}/{job_id}.json")
    # args["num_node_features"] = 33
    deg= None
    if "PNA" in experiment_name:
        deg = custom_tools.load_pickle(f"../models/{experiment_name}/{job_id}_deg.pckl")
    model = custom_tools.load_model(f"{job_id}_SD", path = f"../models/{experiment_name}", model_type = "SD", args = args, deg=deg, device=device)



    

    dataset = TissueDataset(os.path.join("../data/JacksonFischer", "month"),  "month")
    # print(dataset)

    emd, related_data = embeddings.get_intermediate_embeddings_for_dataset(model, dataset, batch_size=1, mode=mode)
    # emd512 = emd[1]

    # print(related_data[0])

    for idx, embed in enumerate(emd):
        plotting.UMAP_plot(embed, idx, related_data, 'tumor_grade', experiment_name, job_id, mode)
        plotting.UMAP_plot(embed, idx, related_data, 'osmonth', experiment_name, job_id, mode)
        plotting.UMAP_plot(embed, idx, related_data, 'is_censored', experiment_name, job_id, mode)



# generate_embedding_plots("GATV2_NegativeLogLikelihood_month_04-12-2023", "YyroGgMa_H4xn_ctP3C5Zw", mode="FC")
# generate_embedding_plots("GATV2_NegativeLogLikelihood_month_04-12-2023", "YyroGgMa_H4xn_ctP3C5Zw", mode="CNV")

generate_embedding_plots("GATV2_NegativeLogLikelihood_fixed_dataset_13-12-2023", "V05lYbfqzxjRjenrPbsplg", mode="FC")
generate_embedding_plots("GATV2_NegativeLogLikelihood_fixed_dataset_13-12-2023", "V05lYbfqzxjRjenrPbsplg", mode="CNV")

# generate_embedding_plots("PNAConv_CoxPHLoss_month_24-11-2023", "e_mnebzfkFBnTMbd65DoKg", mode="CNV")