from numpy import integer
import pandas as pd
import os
import re

cwd = os. getcwd()
RAW_DATA_PATH = os.path.join(cwd, "data", "JacksonFischer")


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
def get_imgnumber_pid_flname_mapping_dict():
    """
    Create a dictionary for mapping file names, patient ids and img ids

    Parameters:
        None

    Returns:
        dict: A dictionary where the keys are "FileName_FullStack" from "Zuri-Basel_PatientMetadata" and values
        are tuples in the form of (<file name from whole image flt>, <img number>), <patient_id>
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_Zuri_WholeImage.csv")
    whole_image_df = pd.read_csv(f_path, index_col=False)
    # whole_image_cols = whole_image_df.columns

    img_numbers_to_be_ignored =[]
    
    dict_basel_patterns = read_basel_patient_meta_data()
    dict_zurich_patterns = read_zurich_patient_meta_data()

    dict_flname_imgnumber_pid = dict()
    dict_imgnumber_pid = dict()
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
                dict_flname_imgnumber_pid[cohort_fl_name] = (fl_name, img_number, pid)
                dict_imgnumber_pid[img_number] = pid
        
        ignore_warnings = True
        # print warning message if pattern cannot be found at all!    
        if not pattern_found:
            img_numbers_to_be_ignored.append(img_number)
            if not ignore_warnings: 
                print("No matching pattern! \n  Whole image fl_name:", fl_name)
            

    return dict_flname_imgnumber_pid, dict_imgnumber_pid, img_numbers_to_be_ignored

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


# TODO: read the single cell image file and get the required features and create the final dataset
def create_preprocessed_sc_feature_fl():
    """
    Perform mapping among files, select features, drop rows with NAs, remove the cells from unwanted images and save the compact file

    Parameters:
        None
    Returns:
        None  
    """
    f_path = os.path.join(RAW_DATA_PATH, "Basel_Zuri_SingleCell.csv")
    cols = pd.read_csv(f_path, index_col=False, nrows=0).columns.tolist()

    cols_to_be_selected = ["ImageNumber",  "ObjectNumber", "Location_Center_X", "Location_Center_Y"]
    
    # get channel ids to be kept
    dict_fullstackid_gene = get_basel_zurich_staining_panel()
    for col in cols:
         if col.startswith("Intensity_MeanIntensity_FullStack_c"):
             channel_id = int(col.split("Intensity_MeanIntensity_FullStack_c")[1])
             if channel_id in dict_fullstackid_gene.keys():
                 cols_to_be_selected.append(col)
                 

    # print(cols_to_be_selected)

    dict_flname_imgnumber_pid, dict_imgnumber_pid, img_numbers_to_be_ignored = get_imgnumber_pid_flname_mapping_dict()

    i=1
    chunks = []
    ch_size = 100000
    with  pd.read_csv(f_path, chunksize=ch_size, index_col=False) as reader:   
        for chunk in reader:
            print(f"Prossesing chunk #{i}... Chunk size: {ch_size}")
            chunks.append(chunk[cols_to_be_selected])
            # remove the rows with cells from unwanted images
            chunks[-1]  = chunks[-1][~chunks[-1].ImageNumber.isin(img_numbers_to_be_ignored)]
            pid_lst = []
            for _, row in chunks[-1].iterrows():
                pid = dict_imgnumber_pid[int(row["ImageNumber"])]
                pid_lst.append(pid)
            
            chunks[-1]["PID"] = pid_lst

            i += 1
    
    # concatenate chunks and remove the rows with NA values
    df_new_dataset = pd.concat(chunks)
    df_new_dataset.dropna(inplace=True)
    df_new_dataset[['PID', 'ObjectNumber', "ImageNumber"]] = df_new_dataset[['PID', 'ObjectNumber', "ImageNumber"]].astype(integer)




    # print(new_dataset.count())
    # print(new_dataset.size)  
    # print(new_dataset.columns)
    # print(new_dataset)
    df_new_dataset.to_csv(os.path.join(RAW_DATA_PATH,  "basel_zurich_preprocessed_compact_dataset.csv"), index=False)