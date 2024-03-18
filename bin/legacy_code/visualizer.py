# Visualization functions kept here
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clinical_data(c_data):
    c_data  = pd.read_csv("../data/METABRIC/brca_metabric_clinical_data.tsv", sep="\t", index_col=False)
    c_data.columns = c_data.columns.str.strip()

    single_cell_data = pd.read_csv("../data/METABRIC/single_cell_data.csv", index_col=False)
    # Print columns
    print("Clinical data columns: ", c_data.columns)
    print("Single cell data columns: ", single_cell_data.columns)

    # Keep rows in c_data with PIDs in single_cell_data
    c_data = c_data[c_data["Patient ID"].isin(single_cell_data["metabricId"])]

    def generate_clinical_type(row):
        ER = row['ER Status']
        HER2 = row['HER2 Status']
        if ER == "Positive" and HER2 == "Positive":
            return "HR+HER2+"
        if ER == "Negative" and HER2 == "Positive":
            return "HR-HER2+"
        if ER == "Positive" and HER2 == "Negative":
            return "HR+HER2-"
        if ER == "Negative" and HER2 == "Negative":
            return "TripleNeg"
        # Check if HER2 is nan
        if HER2 != "Positive" and HER2 != "Negative":
            return "HER2-NAN"
        
    c_data['clinical_type'] = c_data.apply(generate_clinical_type, axis=1)

    # Define custom order for the plot
    custom_order = ["TripleNeg", "HR-HER2+", "HR+HER2-", "HR+HER2+", "HER2-NAN"]

    # Clinical_type vs Survival candle plot using sns and order the clinical_type
    clinical_type = c_data["clinical_type"]
    survival = c_data["Overall Survival (Months)"]
    ax = sns.boxplot(x=clinical_type, y=survival, order=custom_order)
    ax.set(xlabel="Clinical Type", ylabel="Overall Survival (Months)")
    plt.show()

    clinical_type = c_data["clinical_type"]
    age = c_data["Age at Diagnosis"]
    ax = sns.boxplot(x=clinical_type, y=age, order=custom_order)
    ax.set(xlabel="Clinical Type", ylabel="Age at Diagnosis")
    plt.show()

    # Calculate pearson correlation
    print("Pearson correlation between age and survival: ", c_data["Age at Diagnosis"].corr(c_data["Overall Survival (Months)"]))

    # Age vs Survival scatter plot,

    
    ax = sns.scatterplot(x=age, y=survival, hue=clinical_type, hue_order=custom_order)
    ax.set(xlabel="Age at Diagnosis", ylabel="Overall Survival (Months)")
    plt.show()

    
    # HER2-NAN count
    print("HER2-NAN count: ", len(c_data[c_data["clinical_type"] == "HER2-NAN"]))
    # Total count   
    print("Total count: ", len(c_data))