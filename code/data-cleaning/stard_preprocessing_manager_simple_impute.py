import os
import sys
import pandas as pd
#import numpy as np
#from collections import namedtuple
from stard_preprocessing_manager import select_rows, select_columns, one_hot_encode_scales, convert_values, aggregate_rows,  generate_y ,select_subjects, replace_with_median, DIR_PROCESSED_DATA, DIR_AGGREGATED_ROWS, DIR_IMPUTED, IMPUTED_PREFIX, CSV_SUFFIX, DIR_SUBJECT_SELECTED, DIR_Y_MATRIX, add_new_imputed_features
import warnings

#from utils import *
#from stard_preprocessing_globals import ORIGINAL_SCALE_NAMES, BLACK_LIST_SCALES, SCALES, VALUE_CONVERSION_MAP, \
#    VALUE_CONVERSION_MAP_IMPUTE, NEW_FEATURES

""" 
As in the original and imported function stard_preprocessing_manager, this will take in multiple text files (representing psychiatric scales) and output multiple CSV files, at least for each scale read in.
However, this function does a "simple imputation", where all blanks are en masse replaced with median. Not used for the paper previous results. 
"""
LINE_BREAK = "*************************************************************"

def impute(root_data_dir_path):
    warnings.filterwarnings("ignore", category=FutureWarning)
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_aggregated_rows_dir_path = output_dir_path + "/" + DIR_AGGREGATED_ROWS + "/"
    output_imputed_dir_path = output_dir_path + "/" + DIR_IMPUTED + "/"

    # The input directory path will be that from the previous step (#5), row aggregation.
    input_dir_path = output_aggregated_rows_dir_path

    print("\n--------------------------------6. IMPUTATION-----------------------------------\n")

    final_data_matrix = pd.DataFrame()

    for i, filename in enumerate(os.listdir(input_dir_path)):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_imputed_dir_path):
            os.mkdir(output_imputed_dir_path)

        if "rs__cs__ohe__vc__ag__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        print(LINE_BREAK)
        print("Handling full data matrix =", scale_name, ", filename =", filename)

        # Read in the txt file
        agg_df = pd.read_csv(input_dir_path + "/" + filename)
                

        
        # For simple imputation, simple replace all blanks with median
        agg_df = replace_with_median(agg_df, list(agg_df.columns)[1:])
        
        # Re-iterate through agg_df for imputation of the new features, so that "row" contains updated values, to fix imput_qidscpccg bug
        for i, row in agg_df.iterrows():
            agg_df = add_new_imputed_features(agg_df, row, i)
        
        # Drop columns
        agg_df = agg_df.drop(columns=['wsas01__wsastot'])
        
        # Print warning if any empty cells in dataframe
        if agg_df.isnull().sum().sum() != 0:
            print("Warning, an entry in dataframe is null after imputation")
        
        final_data_matrix = agg_df

    output_file_name = IMPUTED_PREFIX + "stard_data_matrix"
    output_path = output_imputed_dir_path + output_file_name + CSV_SUFFIX
    final_data_matrix.to_csv(output_path, index=False)
    print("File has been written to:", output_path)


def rename_files(root_data_dir_path):
    """Simple function to rename files that were affected by changes in this imputation """
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    
    
    # Rename imputed files
    output_imputed_dir_path = output_dir_path + "/" + DIR_IMPUTED + "/" 
    output_file_name = IMPUTED_PREFIX + "stard_data_matrix" 
    os.rename(output_imputed_dir_path + output_file_name + CSV_SUFFIX, output_imputed_dir_path + output_file_name + "_simple_imputation" + CSV_SUFFIX )
    
    # Rename y generation
    output_y_dir_path = output_dir_path + "/" + DIR_Y_MATRIX + "/"
    os.rename(output_y_dir_path + "y_lvl2_rem_qids01" + CSV_SUFFIX, output_y_dir_path + "y_lvl2_rem_qids01" + "_simple_imputation" + CSV_SUFFIX)
    os.rename(output_y_dir_path + "y_wk8_response_qids01" + CSV_SUFFIX, output_y_dir_path + "y_wk8_response_qids01" + "_simple_imputation" + CSV_SUFFIX)

    # Rename final files
    output_subject_selected_path = output_dir_path + "/" + DIR_SUBJECT_SELECTED + "/"
    
    os.rename(output_subject_selected_path + "X_lvl2_rem_qids01__final" + CSV_SUFFIX, output_subject_selected_path + "X_lvl2_rem_qids01__final_simple_imputation" + CSV_SUFFIX)
    os.rename(output_subject_selected_path + "X_wk8_response_qids01__final" + CSV_SUFFIX, output_subject_selected_path + "X_wk8_response_qids01__final_simple_imputation" + CSV_SUFFIX)
    os.rename(output_subject_selected_path + "y_lvl2_rem_qids01__final" + CSV_SUFFIX, output_subject_selected_path + "y_lvl2_rem_qids01__final_simple_imputation" + CSV_SUFFIX)
    os.rename(output_subject_selected_path + "y_wk8_response_qids01__final" + CSV_SUFFIX, output_subject_selected_path + "y_wk8_response_qids01__final_simple_imputation" + CSV_SUFFIX)

if __name__ == "__main__":
    data_dir_path = sys.argv[1]
    option = sys.argv[2]
    is_valid = len(sys.argv) == 3 and os.path.isdir(data_dir_path)

    if is_valid and option in ["--row-select", "-rs"]:
        select_rows(data_dir_path)

    elif is_valid and option in ["--column-select", "-cs"]:
        select_columns(data_dir_path)

    elif is_valid and option in ["--one-hot-encode", "-ohe"]:
        one_hot_encode_scales(data_dir_path)

    elif is_valid and option in ["--value-convert", "-vc"]:
        convert_values(data_dir_path)

    elif is_valid and option in ["--aggregate-rows", "-ag"]:
        aggregate_rows(data_dir_path)

    elif is_valid and option in ["--impute", "-im"]:
        impute(data_dir_path)

    elif is_valid and option in ["--y-generation", "-y"]:
        generate_y(data_dir_path)

    elif is_valid and option in ["--subject-select", "-ss"]:
        select_subjects(data_dir_path)
        
    elif is_valid and option in ["--rename-files", "-rf"]:
        rename_files(data_dir_path)

    elif is_valid and option in ["--run-all", "-a"]:
        select_rows(data_dir_path)
        select_columns(data_dir_path)
        one_hot_encode_scales(data_dir_path)
        convert_values(data_dir_path)
        aggregate_rows(data_dir_path)
        impute(data_dir_path)
        generate_y(data_dir_path)
        select_subjects(data_dir_path)
        rename_files(data_dir_path)

        print("\nSteps complete:\n" +
              "\t Row selection\n" +
              "\t Column selection\n" +
              "\t One-hot encoding\n" +
              "\t Value conversion\n" +
              "\t Row aggregation (generate a single matrix)\n" +
              "\t Imputation of missing values\n" +
              "\t Generation of y matrices\n" +
              "\t Subject selection\n"
              "\t File Renaming\n"
              )

    else:
        raise Exception("Enter valid arguments\n"
              "\t path: the path to a real directory\n"
              "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3")
