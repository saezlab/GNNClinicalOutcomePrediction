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
    df_meta_df = pd.read_csv(f_path, index_col=False)
    
    # print(df_meta_file["FileName_FullStack"])
    # keys are "FileName_FullStack" values are patterns
    dict_zurich_patterns = dict()
    for _, row in df_meta_df.iterrows():

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
        # "B[0-9]{2}\\.[0-9]+_
        # create pattern
        pattern = re.compile(f"^({first_part})_B[0-9][0-9]\.[0-9]+_({second_part})_*")
        dict_zurich_patterns[fl_name] = (pattern, p_id)
        

    return dict_zurich_patterns


#  "Basel_Zuri_WholeImage file" to map the image number
def get_img_number_from_whole_image_file():
    """
    Create a dictionary for mapping file names and img ids

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FileName_FullStack" from "Zuri-Basel_PatientMetadata" and values
        are tuples in the form of (<file name from whole image flt>, <img number>)
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_Zuri_WholeImage.csv")
    whole_image_df = pd.read_csv(f_path, index_col=False)
    whole_image_cols = whole_image_df.columns
    
    dict_basel_patterns = read_basel_patient_meta_data()
    dict_zurich_patterns = read_zurich_patient_meta_data()

    dict_fl_name_img_number = dict()
    for _, row in whole_image_df.iterrows():
        # print(row)
        fl_name = row["FileName_FullStack"]
        img_number = row["ImageNumber"]
        pattern_dict = dict_basel_patterns if fl_name.startswith("Basel") else dict_zurich_patterns
        
        # get image ids for Zurich
        # if fl_name.startswith("ZTMA"):
        pattern_found = False
        for cohort_fl_name, pattern_pid in pattern_dict.items():
            pattern, pid = pattern_pid
            is_match = pattern.findall(fl_name)
            # print(fl_name, is_match)
            if is_match!=[]:
                if pattern_found:
                    print("Something is weird! Pattern found more than once!")
                pattern_found = True
                dict_fl_name_img_number[cohort_fl_name] = (fl_name, img_number)
        
        # print warning message if pattern cannot be found at all!    
        if not pattern_found:
            print("No matching pattern!")
            print("Whole image fl_name:", fl_name)

    return dict_fl_name_img_number

def get_basel_zurich_staining_panel():
    """
    Create a dictionary for fullstackids column to gene names

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FullStack" ids and values are gene names
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_Zuri_StainingPanel.csv")
    staining_panel_df = pd.read_csv(f_path, index_col=False)

    # ids to be eliminated
    unwanted_ids = list(range(1,9)) + list(range(48, 50)) + [26, 32, 36, 42]
    dict_fullstackid_gene = dict()


    for _, row in staining_panel_df.iterrows():
        target = row["Target"]
        full_stack = row["FullStack"]
        if full_stack not in unwanted_ids:
            dict_fullstackid_gene[full_stack] = target

    return dict_fullstackid_gene
