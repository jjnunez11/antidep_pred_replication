import os
import csv
import pandas as pd
import numpy as np
import sys

from canbind_globals import COL_NAME_PATIENT_ID,COL_NAME_EVENTNAME, COL_NAME_GROUP, GROUP_WHITELIST, YGEN_EVENTNAME_WHITELIST,  TARGET_MAP, YGEN_COL_NAMES_TO_CONVERT
from canbind_globals import YGEN_SCALE_FILENAMES, COL_NAMES_BLACKLIST_COMMON, COL_NAMES_BLACKLIST_QIDS
from canbind_utils import get_event_based_value, aggregate_rows
from canbind_utils import is_number, replace_target_col_values, collect_columns_to_extend
from utils import get_valid_subjects
""" 
Generates a y-matrix from the CAN-BIND data. Similiar code to canbind_data_processor, separated to ensure no week8 contamination to data matrix

Example usages

    Basic:
        python canbind_ygen.py /path/to/data/folders

This will output a single CSV file containing the y-matrix

The method expects CSV files to be contained within their own subdirectories from the root directory, as is organized
in the ZIP provided.

TODO: took out most of the superflous code from canbind_preprocessing_manager, which this was based on
      Runs fast, but probably could still take out more and be further optimized. 
"""


def y_gen(root_dir, debug=False):
    global COL_NAMES_CATEGORICAL
    global COL_NAMES_NA
    global FILENAMES
    global NUM_DATA_FILES
    global NUM_DATA_ROWS
    global NUM_DATA_COLUMNS

    # Setup directories to match STAR*D processing
    out_dir = os.path.join(root_dir, "processed_data")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir_final_xy_data_matrices = os.path.join(out_dir, "final_xy_data_matrices")
    if not os.path.exists(out_dir_final_xy_data_matrices):
        os.mkdir(out_dir_final_xy_data_matrices)


    col_names_categorical = {}
    col_names_na = {}

    filenames = []

    num_data_files = 0
    num_data_rows = 0
    num_data_columns = 0

    merged_df = pd.DataFrame([])

    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(subdir, filename)

            if filename in YGEN_SCALE_FILENAMES:
                filenames.append(filename)
                num_data_files += 1
                
                # Track counts and column names for sanity check
                with open(file_path, 'rt') as csvfile:
                    col_names = []
                    csv_reader = csv.reader(csvfile)
                    for i, row in enumerate(csv_reader):
                        num_data_rows += 1

                        # Store the column names
                        if i == 0:
                            col_names = row
                            num_data_columns += len(row)
                            for field in row:
                                field = field.upper()

                                # Collect names of columns that will be extended with extra columns based on event value
                                collect_columns_to_extend(field)
                        else:
                            # Determine all columns with categorical values
                            for j, field_value in enumerate(row):
                                col_name = col_names[j].upper()
                                if field_value == "":
                                    continue
                                elif is_number(field_value):
                                    continue
                                elif field_value == "NA":
                                    col_names_na[col_name] = True
                                else:
                                    col_names_categorical[col_name] = True

                csvfile.close()

                df = pd.read_csv(file_path)

                # Convert all column names to upper case to standardize names
                df.rename(columns=lambda x: x.upper(), inplace=True)

                # Append the CSV dataframe
                merged_df = merged_df.append(df, sort=False)

    # Sort the rows by the patient identifier
    merged_df = merged_df.sort_values(by=[COL_NAME_PATIENT_ID])

    # Back up full merged file for debugging purposes
    if debug: merged_df.to_csv(out_dir + "/merged-data.unprocessed_ygen.csv")

    #### FILTER ROWS AND COLUMNS ####

    # Filter out rows that are controls
    if COL_NAME_GROUP in merged_df:
        merged_df = merged_df.loc[~merged_df.GROUP.str.lower().isin(GROUP_WHITELIST)]
    
    # Filter out rows that were recorded beyond Week 8
    if COL_NAME_EVENTNAME in merged_df:
        merged_df = merged_df.loc[merged_df.EVENTNAME.str.lower().isin(YGEN_EVENTNAME_WHITELIST)]
        

    #### CREATE NEW COLUMNS AND MERGE ROWS ####

    # Handle column extension based on EVENTNAME or VISITSTATUS
    merged_df, extension_blacklist = extend_columns_eventbased(merged_df)

    # Collapse/merge patient rows
    merged_df = aggregate_rows(merged_df)

    # First replace empty strings with np.nan, as pandas knows to ignore creating one-hot columns for np.nan
    # This step is necessary for one-hot encoding and for replacing nan values with a median
    merged_df = merged_df.replace({"": np.nan})

    # Finalize the blacklist, then do a final drop of columns (original ones before one-hot and blacklist columns)
    blacklist_ygen = COL_NAMES_BLACKLIST_QIDS
    blacklist_ygen.extend(COL_NAMES_BLACKLIST_COMMON)
    blacklist_ygen.extend(extension_blacklist)
    merged_df.drop(blacklist_ygen, axis=1, inplace=True)

    # Create y target, eliminate invalid subjects in both X and y (those who don't make it to week 8), convert responder/nonresponder string to binary
    merged_df = get_valid_subjects(merged_df)
    merged_df = merged_df.drop(["RESPOND_WK8"], axis=1)
    merged_df = replace_target_col_values(merged_df, [TARGET_MAP])
    merged_df = merged_df.sort_values(by=[COL_NAME_PATIENT_ID])
    merged_df = merged_df.reset_index(drop=True)
    
    # Rename the column that will be used for the y value (target)
    merged_df = merged_df.rename({"QIDS_RESP_WK8_week 8": "QIDS_RESP_WK8"}, axis='columns', errors='raise')
    merged_df['QIDS_REM_WK8'] = np.nan
    
    # Back up proceesed file before ygeneration
    if debug: merged_df.to_csv(out_dir + "/merged-data_processed_ygen.csv")

    # Require that baseline QIDS-SR be above 5 to keep in line with Nie et al
    b_drop = len(merged_df)
    merged_df = merged_df[merged_df["QIDS_OVERL_SEVTY_baseline"] > 5]
    a_drop = len(merged_df)
    print(f'{a_drop - b_drop} subjects dropped due to having a baseline QIDS-SR 5 or less')

    # Generate targets, replace missing "QIDS_RESP_WK8" values by manually checking criteria
    for i, row in merged_df.iterrows():
        
        baseline_qids_sr = row['QIDS_OVERL_SEVTY_baseline']
        week4_qids_sr = row['QIDS_OVERL_SEVTY_week 4']
        week8_qids_sr = row['QIDS_OVERL_SEVTY_week 8']



        # Find LOCF, either week 8 or week 4
        if not(np.isnan(week8_qids_sr)):
            locf_qids_sr = week8_qids_sr
        elif not(np.isnan(week4_qids_sr)):
            locf_qids_sr = week4_qids_sr
        else:
            # If patient does not have a week 4 or 8 qids_sr, do not generate y, they will be dropped
            continue
        
        # Make qids-sr remission at 8 weeks from scratch
        if locf_qids_sr <= 5:
            merged_df.at[i, 'QIDS_REM_WK8'] = 1
        else:
            merged_df.at[i, 'QIDS_REM_WK8'] = 0   
            
        # Fill in any missing qids-sr response at 8 weeks
        if "QIDS_RESP_WK8" in row:
            if np.isnan(row["QIDS_RESP_WK8"]):
                if locf_qids_sr <= baseline_qids_sr*0.50:
                    merged_df.at[i, 'QIDS_RESP_WK8'] = 1
                else:
                    merged_df.at[i, 'QIDS_RESP_WK8'] = 0
            else:
                if locf_qids_sr <= baseline_qids_sr*0.50:
                    assert merged_df.at[i, 'QIDS_RESP_WK8'] == 1, "Found an error when manually checking QIDS_RESP_WK8"
                else:
                    assert merged_df.at[i, 'QIDS_RESP_WK8'] == 0, "Found an error when manually checking QIDS_RESP_WK8"

    y_wk8_resp = merged_df[['SUBJLABEL', 'QIDS_RESP_WK8']]
    y_wk8_resp.to_csv(out_dir_final_xy_data_matrices + "/y_wk8_resp_qids_sr_canbind.csv", index=False)
    
    y_wk8_rem = merged_df[['SUBJLABEL', 'QIDS_REM_WK8']]
    y_wk8_rem.to_csv(out_dir_final_xy_data_matrices + "/y_wk8_rem_qids_sr_canbind.csv", index=False)
    
    # Save the version containing NaN values just for debugging, not otherwise used
    if debug: merged_df.to_csv(out_dir + "/canbind-clean-aggregated-data.with-id.contains-blanks-ygen.csv")



def extend_columns_eventbased(orig_df):
    """
    Handles adding extra columns based on a condition the value of another column.

    :param orig_df: the original dataframe
    :return: a new, modified dataframe
    """
    global COL_NAMES_NEW_FROM_EXTENSION
    extension_blacklist = []
        
    # Create extra columns with name of event appended, initialized blank
    for scale_group in YGEN_COL_NAMES_TO_CONVERT:
        scale_name = scale_group[0]
        scale_events_whitelist = scale_group[1]
        col_names = scale_group[2]

        for col_name in col_names:
            for event in scale_events_whitelist:
                # Only add extra columns for entries with a non-empty and valid event
                if type(event) != type("") or is_number(event):
                    continue

                new_col_name = col_name + "_" + event

                # Set the value for the new column
                orig_df[new_col_name] = orig_df.apply(lambda row: get_event_based_value(row, event, col_name, scale_name), axis=1)

        extension_blacklist.extend(col_names)

    return orig_df, extension_blacklist


if __name__ == "__main__":
    if len(sys.argv) == 1:
        pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\canbind_data\\'
        y_gen(pathData, debug=False)
    elif len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        y_gen(sys.argv[1])
    else:
        print("Enter valid arguments\n"
               "\t path: the path to a real directory\n")
