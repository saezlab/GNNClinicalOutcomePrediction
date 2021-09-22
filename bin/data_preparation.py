import pandas as pd
import os
import re

cwd = os. getcwd()
RAW_DATA_PATH = os.path.join(cwd, "..", "data", "JacksonFischer")


def read_basel_patient_meta_data():
    """
    Create a pattern dictionary for mapping the patients, images and features

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FileName_FullStack" from "Basel_PatientMetadata" and values
        are tuples in the form of (<the patterns to be caught>, <patient_id>)
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_PatientMetadata.csv")
    df_meta_file = pd.read_csv(f_path, index_col=False)
    
    # print(df_meta_file["FileName_FullStack"])
    # keys are "FileName_FullStack" values are patterns
    dict_basel_patterns = dict()
    for _, row in df_meta_file.iterrows():

        #get patient id
        p_id = row["PID"]

        # get the required fields from "FileName_FullStack" column
        fl_name = row["FileName_FullStack"]
        fields = fl_name.split("_")
        first_part = "_".join(fields[:7])

        # get the required fields from "core" column
        core = row["core"]
        core_X_ind = core.index("X")
        second_part = core[core_X_ind:]
        
        # create pattern
        pattern = re.compile(f"^({first_part})_[0-9]+_({second_part})_*")
        dict_basel_patterns[fl_name] = (pattern, p_id)
        

    return dict_basel_patterns

def read_zurich_patient_meta_data():
    """
    Create a pattern dictionary for mapping the patients, images and features

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FileName_FullStack" from "Zuri_PatientMetadata" and values
        are tuples in the form of (<the patterns to be caught>, <patient_id>)
    """
    f_path = os.path.join(RAW_DATA_PATH, "Zuri_PatientMetadata.csv")
    df_meta_file = pd.read_csv(f_path, index_col=False)
    
    # print(df_meta_file["FileName_FullStack"])
    # keys are "FileName_FullStack" values are patterns
    dict_zurich_patterns = dict()
    for _, row in df_meta_file.iterrows():

        #get patient id
        p_id = row["PID"]

        # get the required fields from "FileName_FullStack" column
        fl_name = row["FileName_FullStack"]
        fields = fl_name.split("_")
        first_part = "_".join(fields[:7])
        

        # get the required fields from "core" column
        core = row["core"].split("slide")[1]
        # pattern = re.compile("[ABC]y[0-9]+x[0-9]+[_]*[0-9]*")
        core_X_ind = -1
        if "A" in core: core_X_ind = core.index("A")
        elif "B" in core: core_X_ind = core.index("B")
        elif "C" in core: core_X_ind = core.index("C")
        else: pass
        
        second_part = core[core_X_ind:]
        print(second_part)
        # "B[0-9]{2}\\.[0-9]+_
        # create pattern
        pattern = re.compile(f"^({first_part})_B[0-9][0-9]\.[0-9]+_({second_part})_*")
        dict_zurich_patterns[fl_name] = (pattern, p_id)
        

    return dict_zurich_patterns


#  "Basel_Zuri_WholeImage file" to map the image number
def read_image_file():

    return 0

