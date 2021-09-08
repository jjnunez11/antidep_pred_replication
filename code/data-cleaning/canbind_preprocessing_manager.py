import os
import csv
import pandas as pd
import numpy as np
import re

from canbind_globals import COL_NAME_PATIENT_ID,COL_NAME_EVENTNAME, EVENTNAME_WHITELIST, ORIGINAL_SCALE_FILENAMES, COL_NAMES_WHITELIST_PSYHIS, COL_NAMES_BLACKLIST_DARS, COL_NAMES_BLACKLIST_SHAPS, COL_NAMES_BLACKLIST_PSYHIS
from canbind_globals import COL_NAME_GROUP, GROUP_WHITELIST, VALUE_REPLACEMENT_MAPS, QLESQ_COL_MAPPING, COL_NAMES_ONE_HOT_ENCODE, COL_NAMES_BLACKLIST_UNIQS, COLLISION_MANAGER
from canbind_utils import aggregate_rows, finalize_blacklist, one_hot_encode, merge_columns, add_columns_to_blacklist
from canbind_utils import is_number, replace_target_col_values_to_be_refactored, collect_columns_to_extend, extend_columns_eventbased
from utils import get_valid_subjects
""" 
Cleans and aggregates CAN-BIND data.

Example usages

    Basic:
        python canbind_preprocessing_manager.py /path/to/data/folders

    Verbose:
        python canbind_preprocessing_manager.py -v /path/to/data/folders

    Super verbose:
        python canbind_preprocessing_manager.py -v+ /path/to/data/folders


This will output CSV files containing the merged and clean data.

The method expects CSV files to be contained within their own subdirectories from the root directory, as is organized
in the ZIP provided.
"""
def aggregate_and_clean(root_dir, verbose=False, extra=False):
    global UNIQ_COLUMNS
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
    out_dir_clean_aggregated = os.path.join(out_dir, "clean_aggregated")
    if not os.path.exists(out_dir_clean_aggregated):
        os.mkdir(out_dir_clean_aggregated)
    
    uniq_columns = {}
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

            if filename in ORIGINAL_SCALE_FILENAMES:
                filenames.append(filename)
                num_data_files += 1
                
                # Convert to csv if data file is an xlsx
                root, ext = os.path.splitext(file_path)
                if (ext == '.xlsx'):
                    read_xlsx = pd.read_excel(file_path)
                    # IPAQ File uses "EVENTME" instead of "EVENTNAME", so replace
                    if "IPAQ" in filename:
                        read_xlsx = read_xlsx.rename({'EVENTME' : 'EVENTNAME'}, axis='columns', errors='raise')
                    file_path = root + '.csv'
                    read_xlsx.to_csv(file_path, index = None, header=True)
                elif (ext != '.csv'):
                    raise Exception("Provided a data file that is neither an xlsx or csv")
                
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
                                if field in uniq_columns:
                                    uniq_columns[field] += 1
                                else:
                                    uniq_columns[field] = 1

                                if field.startswith("DARS_"):
                                    COL_NAMES_BLACKLIST_DARS.append(field)
                                    continue
                                if field.startswith("SHAPS_"):
                                    COL_NAMES_BLACKLIST_SHAPS.append(field)
                                    continue
                                if field.startswith("PSYHIS_") and field not in COL_NAMES_WHITELIST_PSYHIS:
                                    COL_NAMES_BLACKLIST_PSYHIS.append(field)
                                    continue

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
    if verbose: merged_df.to_csv(out_dir + "/merged-data.unprocessed.csv")

    #### FILTER ROWS AND COLUMNS ####

    # Filter out rows that are controls
    if COL_NAME_GROUP in merged_df:
        merged_df = merged_df.loc[~merged_df.GROUP.str.lower().isin(GROUP_WHITELIST)]
    
    # Filter out rows that were recorded beyond Week 2
    if COL_NAME_EVENTNAME in merged_df:
        merged_df = merged_df.loc[merged_df.EVENTNAME.str.lower().isin(EVENTNAME_WHITELIST)]

    #### CREATE NEW COLUMNS AND MERGE ROWS ####

    # Handle column extension based on EVENTNAME or VISITSTATUS
    merged_df = extend_columns_eventbased(merged_df)

    # Collapse/merge patient rows
    merged_df = aggregate_rows(merged_df)

    # Handle replacing values in specific columns, see @VALUE_REPLACEMENT_MAPS
    merged_df = replace_target_col_values_to_be_refactored(merged_df, VALUE_REPLACEMENT_MAPS)

    # Merge QLESQ columns
    merged_df = merge_columns(merged_df, QLESQ_COL_MAPPING)

    # First replace empty strings with np.nan, as pandas knows to ignore creating one-hot columns for np.nan
    # This step is necessary for one-hot encoding and for replacing nan values with a median
    merged_df = merged_df.replace({"": np.nan})

    # One-hot encode specific columns, see @COL_NAMES_ONE_HOT_ENCODE
    merged_df = one_hot_encode(merged_df, COL_NAMES_ONE_HOT_ENCODE)

    # Finalize the blacklist, then do a final drop of columns (original ones before one-hot and blacklist columns)
    finalize_blacklist()

    # Recent addition, I believe due to pandas update, some of cols in blacklist will no longer be in df, so
    # instead just try to drop, and if not there, continue
    for col in COL_NAMES_BLACKLIST_UNIQS:
        try:
            merged_df.drop(col, axis=1, inplace=True)
        except KeyError:
            # print(f'col: {col} was not found!')
            continue
    
    # Eliminate invalid subjects in both X and y (those who don't make it to week 8)
    merged_df = get_valid_subjects(merged_df)
    merged_df = merged_df.drop(["RESPOND_WK8"], axis=1)
    # Convert responder/nonresponder string to binary
    ##merged_df = replace_target_col_values(merged_df, [TARGET_MAP])
    # Sort by pt ID
    merged_df = merged_df.sort_values(by=[COL_NAME_PATIENT_ID])
    merged_df = merged_df.reset_index(drop=True)
    
    # Fix a value in the data that was messed up in a recent version (pt had age of 56, switched to 14 recently, so switched back)
    if merged_df.at[68, 'AGE'] == 16:
        merged_df.at[68, 'AGE'] = 56
        print("Replaced misrecorded age")
        
    # Drop the columns that are used for y
    merged_df = merged_df.drop(["QIDS_RESP_WK8_week 2"], axis=1)
    # Replace "week 2" with "week2" in column names
    merged_df = merged_df.rename(columns=lambda x: re.sub('week 2','week2',x))
    # Write the data that has been cleaned and aggregated, contains blanks so needs imputation
    merged_df.to_csv(out_dir_clean_aggregated + "/canbind_clean_aggregated.csv", index=False)

    if verbose:
        UNIQ_COLUMNS = uniq_columns
        COL_NAMES_CATEGORICAL = col_names_categorical
        COL_NAMES_NA = col_names_na
        FILENAMES = filenames
        NUM_DATA_FILES = num_data_files
        NUM_DATA_ROWS = num_data_rows
        NUM_DATA_COLUMNS = num_data_columns
        print_info(merged_df, extra)

def print_info(merged_df, extra):
    print("\n____Data cleaning summary_____________________________________\n")
    print("Final dimension of the merged table:", merged_df.shape)
    print("Total data files merged:", NUM_DATA_FILES)
    if extra:
        for filename in FILENAMES:
            print("\t", filename)

    print("\nTotal data rows merged:", NUM_DATA_ROWS)
    print("Total data columns merged:", NUM_DATA_COLUMNS)

    repeats = 0
    print("\nColumns that appear more than once across files, which were merged:")
    for col_name, count in UNIQ_COLUMNS.items():
        if count > 1:
            print("\t", col_name, "-", count, "times")
            repeats += count

    if extra:
        print("\nPatient duplicate rows:")
        print(merged_df.groupby(['SUBJLABEL']).size().reset_index(name='Count'))

    print("\nThere are %d columns that have NA values" % len(COL_NAMES_NA))
    for col_name in COL_NAMES_NA:
        print("\t", col_name)

    print("\nThere are %d columns with categorical values:" % len(COL_NAMES_CATEGORICAL))
    for col_name in COL_NAMES_CATEGORICAL:
        if extra:
            print("\t", col_name)

    print("\nThere are %d columns with that had data collisions for a group of patient rows:" % len(COLLISION_MANAGER))
    for col_name in COLLISION_MANAGER:
        if extra:
            print("\t", col_name, COLLISION_MANAGER[col_name])




