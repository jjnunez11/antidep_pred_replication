import os
import sys
import pandas as pd
import numpy as np
from collections import namedtuple

import warnings

from utils import *
from canbind_globals import VALUE_CONVERSION_MAP_IMPUTE, NEW_FEATURES
from stard_preprocessing_manager import replace_with_median, replace 
#from stard_preprocessing_globals import ORIGINAL_SCALE_NAMES, BLACK_LIST_SCALES, SCALES, VALUE_CONVERSION_MAP, \
#    VALUE_CONVERSION_MAP_IMPUTE, NEW_FEATURES

"""
Takes in can_bind data matrix containing blanks, imputes the blanks, and then imputes new features
"""
def impute(data_dir):
    #input_file_name = data_dir + "/" + 'canbind-clean-aggregated-data.with-id.contains-blanks-with-qidssr.csv'
    input_file_name = data_dir + "/" + 'canbind-clean-aggregated-data.with-id.contains-blanks.csv'
    
    
    # Read in the csv file
    df = pd.read_csv(input_file_name)
    
    # Handle replace with mode or median
    df = replace_with_median(df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_median"]["col_names"]))
    #df = replace_with_mode(df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_mode"]["col_names"]))
    
    # Handle direct value conversions (NaN to a specific number)
    ##blank_to_one_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_one"]
    ##blank_to_twenty_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_twenty"]
    blank_to_zero_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_zero"]
    df = replace(df, list(blank_to_zero_config["col_names"]), blank_to_zero_config["conversion_map"])
    ##agg_df = replace(agg_df, list(blank_to_one_config["col_names"]), blank_to_one_config["conversion_map"])
    ##agg_df = replace(agg_df, list(blank_to_twenty_config["col_names"]), blank_to_twenty_config["conversion_map"])
    
    # Handle imputation based on cross-column conditions
    for i, row in df.iterrows():
        if 'SDS_TOT_baseline' in row:
                    col_names = ['SDS_1_1_baseline',
                                 'SDS_2_1_baseline',
                                 'SDS_3_1_baseline',]
                    if np.isnan(row['SDS_TOT_baseline']):
                        df.set_value(i, 'SDS_TOT_baseline', np.sum(row[col_names]))
        if 'SDS_FUNC_RESP_baseline' in row:
                    col_names = ['SDS_1_1_baseline',
                                 'SDS_2_1_baseline',
                                 'SDS_3_1_baseline',]
                    if np.isnan(row['SDS_FUNC_RESP_baseline']):
                        if (row['SDS_TOT_baseline'] <= 12) and (row[col_names] <= 4):
                            df.set_value(i, 'SDS_FUNC_RESP_baseline', 1)
                        else:
                            df.set_value(i, 'SDS_FUNC_RESP_baseline', 0)
        if 'SDS_FUNC_REMISS_baseline' in row:
                    col_names = ['SDS_1_1_baseline',
                                 'SDS_2_1_baseline',
                                 'SDS_3_1_baseline',]
                    if np.isnan(row['SDS_FUNC_REMISS_baseline']):
                        if (row['SDS_TOT_baseline'] <= 6) and (row[col_names] <= 2):
                            df.set_value(i, 'SDS_FUNC_REMISS_baseline', 1)
                        else:
                            df.set_value(i, 'SDS_FUNC_REMISS_baseline', 0)
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
                        df.set_value(i, 'MADRS_TOT_PRO_RATED_baseline', np.sum(row[col_names]))
        if 
        QIDS_OVERL_SEVTY_baseline
        QIDS_OVERL_SEVTY_week2
                    	

    
    # Write output file
    output_file_name = data_dir + "/" + 'canbind_imputed.csv'
    df.to_csv(output_file_name)
    

impute(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\canbind_data_full_auto')

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        impute(sys.argv[1])
    else:
        print("Enter valid path to the dir with canbind input file \n")
