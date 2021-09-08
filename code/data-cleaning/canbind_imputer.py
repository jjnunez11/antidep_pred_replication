import os
import sys
import pandas as pd
import numpy as np

from canbind_globals import VALUE_CONVERSION_MAP_IMPUTE
from stard_preprocessing_manager import replace_with_median, replace 
"""
Takes in can_bind data matrix containing blanks (produced by canbind_data_processor), imputes the blanks, and then imputes new features

Takes as an argument the preprocessed CAN-BIND data, or if no arguments uses
the default path hard-coded below
"""
def impute(root_dir):
    # Setup directories to match STAR*D processing
    data_dir = os.path.join(root_dir, "processed_data")
    input_dir = os.path.join(data_dir, "clean_aggregated")
    out_dir_final_xy_data_matrices = os.path.join(data_dir, "final_xy_data_matrices")
    if not os.path.exists(out_dir_final_xy_data_matrices):
        os.mkdir(out_dir_final_xy_data_matrices)

    input_file_name = os.path.join(input_dir, 'canbind_clean_aggregated.csv')

    # Read in the csv file
    try:
        df = pd.read_csv(input_file_name)
    except:
        print(f"Expected to find canbind_clean_aggregated.csv in {input_dir}")

    # Handle replace with mode or median
    df = replace_with_median(df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_median"]["col_names"]))
    # df = replace_with_mode(df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_mode"]["col_names"]))
    
    # Handle direct value conversions (NaN to a specific number)
    # blank_to_one_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_one"]
    # blank_to_twenty_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_twenty"]
    blank_to_zero_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_zero"]
    df = replace(df, list(blank_to_zero_config["col_names"]), blank_to_zero_config["conversion_map"])
    # agg_df = replace(agg_df, list(blank_to_one_config["col_names"]), blank_to_one_config["conversion_map"])
    # agg_df = replace(agg_df, list(blank_to_twenty_config["col_names"]), blank_to_twenty_config["conversion_map"])
    
    # Handle imputation based on cross-column conditions
    for i, row in df.iterrows():
        if 'SDS_TOT_baseline' in row:
                    col_names = ['SDS_1_1_baseline',
                                 'SDS_2_1_baseline',
                                 'SDS_3_1_baseline',]
                    if np.isnan(row['SDS_TOT_baseline']):
                        df.at[i, 'SDS_TOT_baseline'] =  np.sum(row[col_names])
        if 'SDS_FUNC_RESP_baseline' in row:
                    col_names = ['SDS_1_1_baseline',
                                 'SDS_2_1_baseline',
                                 'SDS_3_1_baseline',]
                    if np.isnan(row['SDS_FUNC_RESP_baseline']):
                        if (row['SDS_TOT_baseline'] <= 12) and (row[col_names] <= 4):
                            df.at[i, 'SDS_FUNC_RESP_baseline'] = 1
                        else:
                            df.at[i, 'SDS_FUNC_RESP_baseline'] = 0
        if 'SDS_FUNC_REMISS_baseline' in row:
                    col_names = ['SDS_1_1_baseline',
                                 'SDS_2_1_baseline',
                                 'SDS_3_1_baseline',]
                    if np.isnan(row['SDS_FUNC_REMISS_baseline']):
                        if (row['SDS_TOT_baseline'] <= 6) and (row[col_names] <= 2):
                            df.at[i, 'SDS_FUNC_REMISS_baseline'] = 1
                        else:
                            df.at[i, 'SDS_FUNC_REMISS_baseline'] = 0
        if 'MADRS_TOT_PRO_RATED_baseline' in row:
                    col_names = ['MADRS_APRNT_SDNS_baseline',
                                 'MADRS_CONC_DFCTY_baseline',
                                 'MADRS_INBLTY_TO_FEEL_baseline',
                                 'MADRS_INN_TNSN_baseline',
                                 'MADRS_LASS_baseline',
                                 'MADRS_PESS_THTS_baseline',
                                 'MADRS_RDCD_APTIT_baseline',
                                 'MADRS_RDCD_SLP_baseline',
                                 'MADRS_RPTRD_SDNS_baseline',
                                 'MADRS_SUICDL_THTS_baseline',]
                    if np.isnan(row['MADRS_TOT_PRO_RATED_baseline']):
                        df.at[i, 'MADRS_TOT_PRO_RATED_baseline'] = np.sum(row[col_names])
                    	
    # Write output file
    output_file_name = os.path.join(out_dir_final_xy_data_matrices, "X_tillwk4_canbind.csv")
    df.to_csv(output_file_name, index=False)
    

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        impute(sys.argv[1])
    elif len(sys.argv) == 0:
        impute(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\canbind_data_full_auto')
    else:
        print("Enter valid path to the dir with canbind input file \n")
