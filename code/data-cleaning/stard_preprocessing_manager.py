import os
import sys
import pandas as pd
import numpy as np
from collections import namedtuple
import warnings
from stard_preprocessing_globals import ORIGINAL_SCALE_NAMES, BLACK_LIST_SCALES, SCALES, VALUE_CONVERSION_MAP, \
    VALUE_CONVERSION_MAP_IMPUTE, NEW_FEATURES, COL_NAME_SUBJECTKEY
from stard_preprocessing_globals import ROW_SELECTION_PREFIX, COLUMN_SELECTION_PREFIX, ONE_HOT_ENCODED_PREFIX, \
    VALUES_CONVERTED_PREFIX, AGGREGATED_ROWS_PREFIX, IMPUTED_PREFIX
from stard_preprocessing_globals import CSV_SUFFIX, DIR_PROCESSED_DATA, DIR_ROW_SELECTED, DIR_COLUMN_SELECTED, \
    DIR_ONE_HOT_ENCODED, DIR_VALUES_CONVERTED, DIR_AGGREGATED_ROWS, DIR_IMPUTED, DIR_Y_MATRIX, DIR_SUBJECT_SELECTED

""" 
This is the main code for our preprocess for the raw STAR*D data from the NIMH, producing the 
preprocessed STAR*D data, which can be used for ML or further processed into
the dataset overlapping with CAN-BIND, etc.

It is provided the holdout set label (holdout, non_holdout, all), as well as the subject ids corresponding, and generates 
all from there  

This will take in multiple text files (representing psychiatric scales) and output multiple CSV files, at least for each scale read in.
"""

LINE_BREAK = "*************************************************************"


def select_rows(input_dir_path, holdout_set_ids, holdout_label='all'):
    output_dir_root_path = os.path.join(input_dir_path, DIR_PROCESSED_DATA)
    output_dir_path = os.path.join(output_dir_root_path, holdout_label)
    output_row_selected_dir_path = output_dir_path + "/" + DIR_ROW_SELECTED + "/"

    print("\n--------------------------------1. ROW SELECTION-----------------------------------\n")
    if not os.path.exists(output_dir_root_path):
        os.mkdir(output_dir_root_path)

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_row_selected_dir_path):
            os.mkdir(output_row_selected_dir_path)

        scale_name = filename.split(".")[0]
        if scale_name not in ORIGINAL_SCALE_NAMES:
            continue
        if scale_name in BLACK_LIST_SCALES:
            continue

        curr_scale_path = input_dir_path + "/" + filename

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, sep='\t', skiprows=[1])
        scale_df = drop_empty_columns(scale_df)
        # Drop empty columns only when checking the entire set, or will end up with different columns in holdout_set etc
        scale_df = scale_df[scale_df[COL_NAME_SUBJECTKEY].isin(holdout_set_ids)]  # Only keep in data from this set

        print(LINE_BREAK)
        print("Handling scale = ", scale_name)

        selection_criteria = ORIGINAL_SCALE_NAMES[scale_name]

        if scale_name in ["ccv01"]:
            # print(f'{len(scale_df.columns)} columns THESE ARE THE COLUMNS IN CCV: {scale_df.columns}')
            if scale_df["week"].isnull().values.any():
                raise Exception("Numerical column should not contain any null values.")

            # Convert column to float type
            scale_df.loc[:, "week"] = scale_df["week"].astype("float")
            if scale_name == "ccv01":
                criteria_2_df = scale_df[(scale_df["level"] == "Level 1") & (2 <= scale_df["week"]) & (scale_df["week"] < 3)]

            output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w2"

            criteria_2_df = select_subject_rows(criteria_2_df, scale_name, selection_criteria)

            criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX, index=False)

        elif scale_name == "hcdm01":
            # Special case, hcdm01 is a modified version of dm01. To avoid changing all downstream naming, just re-assign.
            scale_name = "dm01"
            # So, also grab the correct row selection criteria since scale_name was changed.
            selection_criteria = ORIGINAL_SCALE_NAMES[scale_name]

            scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("float")
            criteria_1_df = scale_df[(scale_df["days_baseline"].notnull() & scale_df["days_baseline"] < 22)
                                     & (scale_df["resid"].notnull()
                                       | scale_df["marital"].notnull()
                                       | scale_df["student"].notnull()
                                       | scale_df["empl"].notnull()
                                       | scale_df["famim"].notnull())]
            criteria_2_df = scale_df[(scale_df["level"] == "Level 1")
                                     & (scale_df["inc_curr"].notnull()
                                       | scale_df["assist"].notnull()
                                       | scale_df["unempl"].notnull()
                                       | scale_df["otherinc"].notnull()
                                       | scale_df["totincom"].notnull())]

            output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "_enroll"
            output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w0"

            criteria_1_df = select_subject_rows(criteria_1_df, scale_name, selection_criteria)
            criteria_2_df = select_subject_rows(criteria_2_df, scale_name, selection_criteria)

            criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX, index=False)
            criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX, index=False)

        # Handles creating the preliminary file. See end of this function to see how qids was split up.
        elif scale_name == "qids01":
            # Starts with 84,932 rows of which 39,380 are null for column "week"
            print("Number of qids week column are null before replacing with week values matching key (subjectkey, "
                  "days_baseline, level): {}".format(sum(scale_df["week"].isnull()))) # 39380 are null
            Entry = namedtuple("Entry", ["subjectkey", "days_baseline", "level"])
            tracker = {}
            missed = {}

            # Add all non-blanks to dictionary
            week_nonnull_scale_df = scale_df[scale_df["week"].notnull()]
            for idx, row in week_nonnull_scale_df.iterrows():
                entry = Entry(row["subjectkey"], row["days_baseline"], row["level"])
                tracker[entry] = row["week"]

            # Replace all blanks for "week" with the value of a matching entry key
            week_null_scale_df = scale_df[scale_df["week"].isnull()]
            for idx, row in week_null_scale_df.iterrows():
                entry = Entry(row["subjectkey"], row["days_baseline"], row["level"])
                if entry in tracker:
                    scale_df.loc[row.name, "week"] = tracker[entry]
                else:
                    missed[entry] = row["week"]

            print("Number of qids week column are null after replacing with week values matching key (subjectkey, "
                  "days_baseline, level): {}".format(sum(scale_df["week"].isnull()))) # 13234 are null
            print("\nNumber of qids rows before eliminating rows empty for all of {} columns: {}"
                  .format(["vsoin", "vmnin", "vemin", "vhysm", "vmdsd"], scale_df.shape[0]))

            # Select rows where these following columns are not null
            scale_df = scale_df[(scale_df["vsoin"].notnull())
                                & (scale_df["vmnin"].notnull())
                                & (scale_df["vemin"].notnull())
                                & (scale_df["vhysm"].notnull())
                                & (scale_df["vmdsd"].notnull())]
            print("Number of qids rows after eliminating rows empty for all of {} columns: {}"
                  .format(["vsoin", "vmnin", "vemin", "vhysm", "vmdsd"], scale_df.shape[0]))

            # Select rows where qvtot is blank
            print("Number of qids week column are null: {}".format(sum(scale_df["week"].isnull())))

            scale_df = scale_df[(scale_df["qvtot"].isnull())]
            print("\nNumber of qids week column are null after selecting empty qvtot rows: {}".format(sum(scale_df["week"].isnull())))

            # Handle filling in week with 0's and 2's
            week_zero_cond = (scale_df["week"].isnull()) & (scale_df["days_baseline"] < 8) & (scale_df["level"] == "Level 1")
            week_two_cond = (scale_df["week"].isnull()) & (scale_df["days_baseline"] < 22) & (scale_df["level"] == "Level 1")

            scale_df.loc[week_zero_cond, "week"] = 0
            print("Number of qids week column are null after selecting conditions for filling with 0: {}".format(sum(scale_df["week"].isnull())))

            scale_df.loc[week_two_cond, "week"] = 2
            print("Number of qids week column are null after selecting conditions for filling with 2: {}".format(sum(scale_df["week"].isnull())))

            # Output
            output_file_name = "pre" + ROW_SELECTION_PREFIX + "pre" + scale_name
            scale_df.to_csv(output_row_selected_dir_path + output_file_name + CSV_SUFFIX, index=False)

        else:
            if scale_name in ["mhx01", "pdsq01", "phx01"]:
                scale_df = scale_df.drop_duplicates(subset='subjectkey', keep='first')
            elif scale_name in ["qlesq01", "sfhs01", "ucq01", "wpai01", "wsas01"]:
                scale_df = scale_df[scale_df["CallType"] == "Base"]
            elif scale_name == "crs01":
                scale_df = scale_df[scale_df["crcid"].notnull()]
            elif scale_name == "hrsd01":
                scale_df = scale_df[scale_df["level"] == "Enrollment"]
            elif scale_name == "idsc01":
                scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("int")
                scale_df = scale_df[(scale_df["level"] == "Level 1") & (scale_df["days_baseline"] < 22)]
            elif scale_name == "side_effects01":
                scale_df.loc[:, "week"] = scale_df["week"].astype("float")
                scale_df = scale_df[(scale_df["level"] == 1) & (scale_df["week"] < 3)]

            output_file_name = ROW_SELECTION_PREFIX + scale_name
            scale_df = select_subject_rows(scale_df, scale_name, selection_criteria)
            scale_df.to_csv(output_row_selected_dir_path + output_file_name + CSV_SUFFIX, index=False)

    # Handle preqids, after looping through the original scales
    preqids_file_path = output_row_selected_dir_path + "prers__preqids01.csv"
    if os.path.exists(preqids_file_path):
        scale_df = pd.read_csv(preqids_file_path)
        # scale_df = scale_df.drop(columns=["Unnamed: 0"])

        # Convert column to numeric type
        scale_df.loc[:, "week"] = scale_df["week"].astype("float")
        scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("int")

        # Split into 3 separate files
        criteria_1_df = scale_df[(scale_df["level"] == "Level 1")
                                 & (scale_df["week"] < 1)
                                 & (scale_df["version_form"] == "Clinician")]
        criteria_2_df = scale_df[(scale_df["level"] == "Level 1")
                                 & (scale_df["week"] < 1)
                                 & (scale_df["version_form"] == "Self Rating")]
        criteria_3_df_a = scale_df[(scale_df["level"] == "Level 1")
                                 & (2 <= scale_df["week"]) & (scale_df["week"] < 3)
                                 & (scale_df["version_form"] == "Clinician")]
        criteria_3_df_b = scale_df[(scale_df["level"] == "Level 2")
                                 & (7 < scale_df["days_baseline"]) & (scale_df["days_baseline"] < 22)
                                 & (scale_df["version_form"] == "Clinician")]
        criteria_3_df = pd.concat([criteria_3_df_a, criteria_3_df_b])
        criteria_4_df_a = scale_df[(scale_df["level"] == "Level 1")
                                 & (2 <= scale_df["week"]) & (scale_df["week"] < 3)
                                 & (scale_df["version_form"] == "Self Rating")]
        criteria_4_df_b = scale_df[(scale_df["level"] == "Level 2")
                                 & (7 <= scale_df["days_baseline"]) & (scale_df["days_baseline"] < 22)
                                 & (scale_df["version_form"] == "Self Rating")]
        criteria_4_df = pd.concat([criteria_4_df_a, criteria_4_df_b])

        scale_name = "qids01"
        selection_criteria = ORIGINAL_SCALE_NAMES[scale_name]
        criteria_1_df = select_subject_rows(criteria_1_df, scale_name, selection_criteria)
        criteria_2_df = select_subject_rows(criteria_2_df, scale_name, selection_criteria)
        criteria_3_df = select_subject_rows(criteria_3_df, scale_name, selection_criteria)
        criteria_4_df = select_subject_rows(criteria_4_df, scale_name, selection_criteria)

        output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "_w0c"
        output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w0sr"
        output_file_name_3 = ROW_SELECTION_PREFIX + scale_name + "_w2c"
        output_file_name_4 = ROW_SELECTION_PREFIX + scale_name + "_w2sr"

        criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX, index=False)
        criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX, index=False)
        criteria_3_df.to_csv(output_row_selected_dir_path + output_file_name_3 + CSV_SUFFIX, index=False)
        criteria_4_df.to_csv(output_row_selected_dir_path + output_file_name_4 + CSV_SUFFIX, index=False)


def select_subject_rows(scale_df, scale_name, selection_criteria, debug=False):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if selection_criteria == {}:
        return scale_df

    selector_col_name = selection_criteria["subjectkey_selector"]
    preference = selection_criteria["preference"]
    subject_group = scale_df.groupby(["subjectkey"])

    for subjectkey, subject_rows_df in subject_group:
        condition_val = np.nanmin(subject_rows_df[selector_col_name])
        if preference == "larger":
            condition_val = np.nanmax(subject_rows_df[selector_col_name])

        if debug:
            if np.isnan(condition_val):
                print("NaN:", subjectkey, scale_name)

        # There could be multiple matches, so select a single one based on the selection criteria configuration for a scale.
        matches = scale_df[(scale_df["subjectkey"] == subjectkey) & (scale_df[selector_col_name] == condition_val)]
        if scale_name == "dm01" and np.isnan(condition_val):
            # This is hard-coded, because it was added later and it's unknown whehter the desired effect for this scale
            # is shared with all the other scales. Would need to spend more time to evaluate this, and compare before/after
            # row selection cases for each scale. For now, keep it this way.
            # Reason: all values for the "enroll" selection criteria condition are blank.
            col_matches_condition_val = (scale_df[selector_col_name] == condition_val) | (np.isnan(scale_df[selector_col_name]))
            matches = scale_df[(scale_df["subjectkey"] == subjectkey) & (col_matches_condition_val)]

        if len(matches) > 0:
            if debug:
                print(scale_name, selector_col_name, subjectkey, condition_val, subject_rows_df[selector_col_name])
            scale_df = scale_df[(scale_df["subjectkey"] != subjectkey) | (scale_df.index == matches.index[0])]
        else:
            if debug:
                print(scale_name, selector_col_name, subjectkey, condition_val, subject_rows_df[selector_col_name])
            scale_df = scale_df[(scale_df["subjectkey"] != subjectkey)]

    return scale_df


"""
root_data_dir_path is the path to the root of the folder containing the original scales
"""


def select_columns(root_data_dir_path,  holdout_label='all'):
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    output_row_selected_dir_path = os.path.join(output_dir_path, DIR_ROW_SELECTED)
    output_column_selected_dir_path = os.path.join(output_dir_path, DIR_COLUMN_SELECTED)

    input_dir_path = output_row_selected_dir_path

    print("\n--------------------------------2. COLUMN SELECTION-----------------------------------\n")

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_column_selected_dir_path):
            os.mkdir(output_column_selected_dir_path)

        # This is the prefix for the row selected scales.
        if "rs" != filename.split("__")[0]:
            continue

        curr_scale_path = input_dir_path + "/" + filename

        scale_name = filename.split(".")[0].split("__")[-1]
        print(LINE_BREAK)
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path)

        # Drop empty columns
        # scale_df = drop_empty_columns(scale_df)

        whitelist = SCALES[scale_name]["whitelist"]

        # Add subject key so that you know which subject it is
        if scale_name != "qids01_w0c":
            whitelist.append("subjectkey")

        # Select columns in the whitelist
        scale_df = scale_df[whitelist]

        output_file_name = COLUMN_SELECTION_PREFIX + scale_name

        scale_df.to_csv(os.path.join(output_column_selected_dir_path, output_file_name + CSV_SUFFIX), index=False)

def one_hot_encode_scales(root_data_dir_path,  holdout_label='all'):
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    output_column_selected_dir_path = os.path.join(output_dir_path, DIR_COLUMN_SELECTED)
    output_one_hot_encoded_dir_path = os.path.join(output_dir_path, DIR_ONE_HOT_ENCODED)
    input_dir_path = output_column_selected_dir_path

    print("\n--------------------------------3. ONE HOT ENCODING-----------------------------------\n")

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_one_hot_encoded_dir_path):
            os.mkdir(output_one_hot_encoded_dir_path)

        # This is the prefix for the row, then column selected scales.
        if "rs__cs__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        cols_to_one_hot_encode = []
        if "one_hot_encode" in SCALES[scale_name]:
             cols_to_one_hot_encode = SCALES[scale_name]["one_hot_encode"]

        print(LINE_BREAK)
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file
        scale_df = pd.read_csv(input_dir_path + "/" + filename)

        if scale_name == "dm01_enroll":
            cols_to_convert = ['empl', 'volun', 'leave', 'publica', 'medicaid', 'privins']
            conversion_map = {15: np.nan, 9: np.nan, -7: np.nan}
            for col_name in cols_to_convert:
                scale_df[col_name] = scale_df[col_name].astype("object")
                scale_df[col_name] = scale_df[col_name].replace(to_replace=conversion_map)




        elif scale_name == "phx01":
            """
            Note: the raw files had these unique values per column (in order of the list below)
            [nan  0.  1.  2.]
            [nan  0.  2.  1.]
            [nan  0.  1.  2.]
            [nan  0.  2.  1.]
            [nan  0.  2.  1.]
            [0 1]
            
            After conversion, this is the result:
            [nan  1.  2.]
            [nan  2.  1.]
            [nan  1.  2.]
            [nan  2.  1.]
            [nan  2.  1.]
            [nan] <-- this is for bulimia. There were no actual values other than 0 or 1, so there is no one-hot encoding that will occur.
            We may need to manually create the columns for 2/5, 3, 4 and set them all to 0 (false).
            """
            cols_to_convert = ['alcoh', 'amphet', 'cannibis', 'opioid', 'ax_cocaine', 'bulimia']
            for col_name in cols_to_convert:
                if col_name == "bulimia":
                    conversion_map = {0: np.nan, 1: np.nan, 2: "2/5", 5: "2/5"}
                else:
                    conversion_map = {0: np.nan}
                scale_df[col_name] = scale_df[col_name].astype("object")
                scale_df[col_name] = scale_df[col_name].replace(to_replace=conversion_map)

            scale_df["bulimia||2/5"] = 0
            scale_df["bulimia||3"] = 0
            scale_df["bulimia||4"] = 0

        elif scale_name in ["ccv01_w2", "idsc01", "qids01_w0c"]:
            # No special conversion steps prior to one-hot encoding
            pass

        if cols_to_one_hot_encode is None or len(cols_to_one_hot_encode) == 0:
            scale_df = scale_df
        else:
            scale_df = one_hot_encode(scale_df, cols_to_one_hot_encode)
            scale_df = scale_df.drop(columns=cols_to_one_hot_encode)

        #When making a holdout set, some rare entires may not exist post one-hot, so check and add them if needed to
        # match the non-holdout set etc.
        if scale_name == "dm01_enroll":
            for rare_col in ['resid||7.0', 'resid||6.0']:
                if not (rare_col in scale_df):
                    scale_df[rare_col] = 0.0
            scale_df = scale_df.reindex(sorted(scale_df.columns), axis=1) # Sort to ensure all sets have same column order
        elif scale_name == "ccv01_w2":
            if not ("trtmt||3.0" in scale_df):
                scale_df["trtmt||3.0"] = 0.0
            scale_df = scale_df.reindex(sorted(scale_df.columns), axis=1) # Sort to ensure all sets have same column order


        output_file_name = ONE_HOT_ENCODED_PREFIX + scale_name
        scale_df.to_csv(os.path.join(output_one_hot_encoded_dir_path, output_file_name + CSV_SUFFIX) , index=False)


def convert_values(root_data_dir_path,  holdout_label):
    """
    Handles converting values for different variables per scale. This is similar to the imputation step, in that certain
    values are being derived, however the difference is that this step handles it solely on the scale-level. For (1) generic
    value conversion that is common across all scales, or (2) value conversion/imputation is dependent on values of features
    between different scales, then this will be handled in step #6, imputation. 
    """
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    output_one_hot_encoded_dir_path = os.path.join(output_dir_path, DIR_ONE_HOT_ENCODED)
    output_values_converted_dir_path = os.path.join(output_dir_path, DIR_VALUES_CONVERTED)
    input_dir_path = output_one_hot_encoded_dir_path

    print("\n--------------------------------4. VALUE CONVERSION-----------------------------------\n")

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_values_converted_dir_path):
            os.mkdir(output_values_converted_dir_path)

        if "rs__cs__ohe__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        print(LINE_BREAK)
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file
        scale_df = pd.read_csv(input_dir_path + "/" + filename)

        for col_name in scale_df.columns.values:
            for key, dict in VALUE_CONVERSION_MAP.items():
                if key == "minus":
                    continue
                elif col_name in dict["col_names"]:
                    scale_df[col_name] = scale_df[col_name].astype("object")
                    scale_df[col_name] = scale_df[col_name].replace(to_replace=dict["conversion_map"])
                elif key == "blank_to_zero":
                    if col_name in dict["col_names"]:
                        scale_df = handle_replace_if_row_null(scale_df, col_name)

        if scale_name == "sfhs01":
            config = VALUE_CONVERSION_MAP["minus"]
            for key, list_of_cols in config.items():
                print(key, list_of_cols)
                for col_name in list_of_cols:
                    conversion_map = {}
                    for k, value in scale_df[col_name].iteritems():
                        if value in conversion_map:
                            continue
                        elif key == 6 or key == 3:
                            conversion_map[value] = key - value
                        elif key == 1:
                            conversion_map[value] = value - 1
                    scale_df[col_name] = scale_df[col_name].replace(to_replace=conversion_map)

        if scale_name == "dm01_enroll":
            scale_df["resm"] = scale_df["resy"] * 12 + scale_df["resm"]
            scale_df = scale_df.drop(columns=["resy"])
            
        if scale_name == "phx01":
            scale_df["episode_date"] = abs(scale_df["episode_date"]) 

        output_file_name = VALUES_CONVERTED_PREFIX + scale_name
        scale_df.to_csv(os.path.join(output_values_converted_dir_path, output_file_name + CSV_SUFFIX), index=False)

def handle_replace_if_row_null(df, col_name):
    """
    Handles blank to zero conversion for rows which are null for a given scale. 
    """
    for i, row in df.iterrows():
        # If all column values are empty for this row, then leave it all null
        if sum(row.isnull()) == len(row):
            continue
        # But if there are non-empty values, then convert the col_name value to 0
        else:
            df.at[i, col_name] = 0
    return df


def aggregate_rows(root_data_dir_path,  holdout_label='all'):
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    output_aggregated_rows_dir_path = os.path.join(output_dir_path, DIR_AGGREGATED_ROWS)
    output_values_converted_dir_path = os.path.join(output_dir_path, DIR_VALUES_CONVERTED)
    input_dir_path = output_values_converted_dir_path

    print("\n--------------------------------5. ROW AGGREGATION-----------------------------------\n")

    main_keys = ['subjectkey', 'gender||F', 'gender||M', 'interview_age']
    aggregated_df = pd.DataFrame()

    for i, filename in enumerate(os.listdir(input_dir_path)):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_aggregated_rows_dir_path):
            os.mkdir(output_aggregated_rows_dir_path)

        if "rs__cs__ohe__vc__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        print(LINE_BREAK)
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file
        scale_df = pd.read_csv(input_dir_path + "/" + filename)

        # Append scale name and version to the column name
        cols = {}
        for col_name in scale_df.columns.values:
            if col_name in main_keys:
                continue
            else:
                cols[col_name] = scale_name + "__" + str(col_name)
        scale_df = scale_df.rename(columns = cols)

        if i == 0:
            aggregated_df = scale_df
        else:
            aggregated_df["subjectkey"] = aggregated_df["subjectkey"].astype(object)
            scale_df["subjectkey"] = scale_df["subjectkey"].astype(object)

            # The left df has to be the one with more rows, as joining the two will ensure all subjects are grabbed. With an outer join (as opposed to left join) 
            # this is likely not relevant anymore but I'm leaving this as I don't have time to check this. It will not cause any issues anyway. 
            if aggregated_df.shape[0] >= scale_df.shape[0]:
                left = aggregated_df
                right = scale_df
            else:
                left = scale_df
                right = aggregated_df

            aggregated_df = left.merge(right, on="subjectkey", how="outer")

    output_file_name = AGGREGATED_ROWS_PREFIX + "stard_data_matrix"
    aggregated_df = aggregated_df.reindex(columns=(main_keys + list([a for a in aggregated_df.columns if a not in main_keys])))
    aggregated_df.to_csv(os.path.join(output_aggregated_rows_dir_path, output_file_name + CSV_SUFFIX), index=False)


def impute(root_data_dir_path, holdout_set_ids, holdout_label='all'):
    warnings.filterwarnings("ignore", category=FutureWarning)
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    output_aggregated_rows_dir_path = os.path.join(output_dir_path, DIR_AGGREGATED_ROWS)
    output_imputed_dir_path = os.path.join(output_dir_path, DIR_IMPUTED)
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
        agg_df = pd.read_csv(os.path.join(input_dir_path, filename))

        # Handle replace with mode or median
        agg_df = replace_with_median(agg_df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_median"]["col_names"]))
        agg_df = replace_with_mode(agg_df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_mode"]["col_names"]))

        # Handle direct value conversions (NaN to a specific number)
        blank_to_zero_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_zero"]
        blank_to_one_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_one"]
        blank_to_twenty_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_twenty"]
        agg_df = replace(agg_df, list(blank_to_zero_config["col_names"]), blank_to_zero_config["conversion_map"])
        agg_df = replace(agg_df, list(blank_to_one_config["col_names"]), blank_to_one_config["conversion_map"])
        agg_df = replace(agg_df, list(blank_to_twenty_config["col_names"]), blank_to_twenty_config["conversion_map"])

        crs01_df = pd.read_csv(root_data_dir_path + "/crs01.txt", sep="\t", skiprows=[1]) #TODO: REPLACED
        crs01_df = crs01_df[crs01_df[COL_NAME_SUBJECTKEY].isin(holdout_set_ids)]  # Only keep in data from this set

        crs01_df.loc[:, "interview_age"] = crs01_df["interview_age"].astype("float")

        for new_feature in NEW_FEATURES:
            agg_df[new_feature] = np.nan

        # Handle imputation based on cross-column conditions
        for i, row in agg_df.iterrows():
            if ('gender||F' in row and 'gender||M' in row) and (np.isnan(row['gender||F']) or np.isnan(row['gender||M'])):
                    # If one of the dummy variables for gender are empty, then grab it from the scale crs01. 
                    gender_series = crs01_df.loc[crs01_df['subjectkey'] == row['subjectkey']]['gender']
                    if len(gender_series) == 0:
                        # Ultimately want to remove subjects that have age or gender missing, likely.
                        print("This subject [" + row['subjectkey'] + "] does not have a value stored in 'crs01' for gender. Keep this blank for now.")
                    else:
                        gender = gender_series.iloc[0]
                        if gender == "M":
                            agg_df.at[i, 'gender||F'] = 0
                            agg_df.at[i, 'gender||M'] = 1
                        elif gender == "F" or np.isnan(gender):
                            agg_df.at[i, 'gender||F'] = 1
                            agg_df.at[i, 'gender||M'] =  0
            if 'interview_age' in row and np.isnan(row['interview_age']):
                age_series = crs01_df.loc[crs01_df['subjectkey'] == row['subjectkey']]['interview_age']
                if len(age_series) == 0:
                    # Ultimately want to remove subjects that have age or gender missing, likely.
                    print("This subject [" + row['subjectkey'] + "] does not have a value stored in 'crs01' for interview_age. Keep this blank for now.")
                else:
                    age = age_series.iloc[0]
                    if np.isnan(age):
                        print("Age is null", row["subjectkey"])
                        agg_df.at[i, 'interview_age'] =  agg_df['interview_age'].median()
                    else:
                        agg_df.at[i, 'interview_age'] =  age
            if 'ucq01__ucq010' in row:
                val = row['ucq01__ucq010']
                #if row['ucq01__ucq010'] == 0:
                #    val = 0
                if np.isnan(row['ucq01__ucq020']):
                    agg_df.at[i, 'ucq01__ucq020'] = val
                if np.isnan(row['ucq01__ucq030']):
                    agg_df.at[i, 'ucq01__ucq030'] =  val
            if 'wpai01__wpai01' in row:
                if row['wpai01__wpai01'] == 1:
                    if np.isnan(row['dm01_w0__inc_curr']):
                        agg_df.at[i, 'dm01_w0__inc_curr'] =  1
                    if np.isnan(row['dm01_w0__mempl']):
                        agg_df.at[i, 'dm01_w0__mempl'] = 2000
                    if np.isnan(row['dm01_enroll__empl||1.0']):
                        agg_df.at[i, 'dm01_enroll__empl||1.0'] =  0
                    if np.isnan(row['dm01_enroll__empl||3.0']):
                        agg_df.at[i, 'dm01_enroll__empl||3.0'] = 1
                    if np.isnan(row['dm01_enroll__privins||0.0']):
                        agg_df.at[i, 'dm01_enroll__privins||0.0'] = 0
                    if np.isnan(row['dm01_enroll__privins||1.0']):
                        agg_df.at[i, 'dm01_enroll__privins||1.0'] =  1

                elif row['wpai01__wpai01'] == 0 or np.isnan(row['wpai01__wpai01']):
                    if np.isnan(row['dm01_w0__inc_curr']):
                        agg_df.at[i, 'dm01_w0__inc_curr'] = 0
                    if np.isnan(row['dm01_w0__mempl']):
                        agg_df.at[i, 'dm01_w0__mempl'] =  0
                    if np.isnan(row['dm01_enroll__empl||1.0']):
                        agg_df.at[i, 'dm01_enroll__empl||1.0'] = 1
                    if np.isnan(row['dm01_enroll__privins||1.0']):
                        agg_df.at[i, 'dm01_enroll__privins||1.0'] = 0
                    if np.isnan(row['dm01_enroll__empl||3.0']):
                        agg_df.at[i, 'dm01_enroll__empl||3.0'] = 0
                    if np.isnan(row['dm01_enroll__privins||0.0']):
                        agg_df.at[i, 'dm01_enroll__privins||0.0'] = 1    

                else:
                    # Above two scenarios should handle all, print message if wpai is something else
                    print("Unxpected wpai01 value during imputation: " + str(row['wpai01__wpai01']))
                    
                    ##agg_df.set_value(i, 'dm01_enroll__empl||3.0', 0)
                    ##agg_df.set_value(i, 'dm01_enroll__privins||0.0', 1)
                    ##agg_df.set_value(i, 'dm01_enroll__privins||1.0', 0)
            if 'wsas01__totwsas' in row and np.isnan(row['wsas01__totwsas']):
                col_names = ['wsas01__wsas01','wsas01__wsas02', 'wsas01__wsas03', 'wsas01__wsas04', 'wsas01__wsas05']
                agg_df.at[i, 'wsas01__totwsas'] = np.sum(row.reindex(col_names))
            if 'hrsd01__hdtot_r' in row and np.isnan(row['hrsd01__hdtot_r']):
                col_names = ['hrsd01__hsoin',
                             'hrsd01__hmnin',
                             'hrsd01__hemin',
                             'hrsd01__hmdsd',
                             'hrsd01__hpanx'
                             'hrsd01__hinsg',
                             'hrsd01__happt',
                             'hrsd01__hwl',
                             'hrsd01__hsanx',
                             'hrsd01__hhypc',
                             'hrsd01__hvwsf',
                             'hrsd01__hsuic',
                             'hrsd01__hintr',
                             'hrsd01__hengy',
                             'hrsd01__hslow',
                             'hrsd01__hagit',
                             'hrsd01__hsex']

                agg_df.at[i, 'hrsd01__hdtot_r'] = np.sum(row.reindex(col_names))
            if 'qids01_w0sr__qstot' in row:
                value = np.nanmax(list(row[["qids01_w0sr__vsoin", "qids01_w0sr__vmnin", "qids01_w0sr__vemin", "qids01_w0sr__vhysm"]])) \
                + np.nanmax(list(row[["qids01_w0sr__vapdc", "qids01_w0sr__vapin", "qids01_w0sr__vwtdc", "qids01_w0sr__vwtin"]])) \
                + np.nanmax(list(row[["qids01_w0sr__vslow", "qids01_w0sr__vagit"]])) \
                + np.sum(row[["qids01_w0sr__vmdsd", "qids01_w0sr__vengy", "qids01_w0sr__vintr", "qids01_w0sr__vsuic", "qids01_w0sr__vvwsf", "qids01_w0sr__vcntr"]])
                agg_df.at[i, 'qids01_w0sr__qstot'] =  value
            if 'qids01_w2sr__qstot' in row:
                value = np.nanmax(list(row[["qids01_w2sr__vsoin", "qids01_w2sr__vmnin", "qids01_w2sr__vemin", "qids01_w2sr__vhysm"]])) \
                + np.nanmax(list(row[["qids01_w2sr__vapdc", "qids01_w2sr__vapin", "qids01_w2sr__vwtdc", "qids01_w2sr__vwtin"]])) \
                + np.nanmax(list(row[["qids01_w2sr__vslow", "qids01_w2sr__vagit"]])) \
                + np.sum(row[["qids01_w2sr__vmdsd", "qids01_w2sr__vengy", "qids01_w2sr__vintr", "qids01_w2sr__vsuic", "qids01_w2sr__vvwsf", "qids01_w2sr__vcntr"]])
                agg_df.at[i, 'qids01_w2sr__qstot'] =  value
            if 'qids01_w0c__qstot' in row:
                value = np.nanmax(list(row[["qids01_w0c__vsoin", "qids01_w0c__vmnin", "qids01_w0c__vemin", "qids01_w0c__vhysm"]])) \
                + np.nanmax(list(row[["qids01_w0c__vapdc", "qids01_w0c__vapin", "qids01_w0c__vwtdc", "qids01_w0c__vwtin"]])) \
                + np.nanmax(list(row[["qids01_w0c__vslow", "qids01_w0c__vagit"]])) \
                + np.sum(row[["qids01_w0c__vmdsd", "qids01_w0c__vengy", "qids01_w0c__vintr", "qids01_w0c__vsuic", "qids01_w0c__vvwsf", "qids01_w0c__vcntr"]])
                agg_df.at[i, 'qids01_w0c__qstot'] =  value
            if 'qids01_w2c__qstot' in row:
                value = np.nanmax(list(row[["qids01_w2c__vsoin", "qids01_w2c__vmnin", "qids01_w2c__vemin", "qids01_w2c__vhysm"]])) \
                + np.nanmax(list(row[["qids01_w2c__vapdc", "qids01_w2c__vapin", "qids01_w2c__vwtdc", "qids01_w2c__vwtin"]])) \
                + np.nanmax(list(row[["qids01_w2c__vslow", "qids01_w2c__vagit"]])) \
                + np.sum(row[["qids01_w2c__vmdsd", "qids01_w2c__vengy", "qids01_w2c__vintr", "qids01_w2c__vsuic", "qids01_w2c__vvwsf", "qids01_w2c__vcntr"]])
                agg_df.at[i, 'qids01_w2c__qstot'] =  value

            
        # Re-iterate through agg_df for imputation of the new features, so that "row" contains updated values, to fix imput_qidscpccg bug
        for i, row in agg_df.iterrows():
            agg_df = add_new_imputed_features(agg_df, row, i)

        # Drop columns
        agg_df = agg_df.drop(columns=['wsas01__wsastot'])

        # Loop through each row, and replace specific columns by the combination of other columns.
        final_data_matrix = agg_df

    output_file_name = IMPUTED_PREFIX + "stard_data_matrix"
    output_path = os.path.join(output_imputed_dir_path, output_file_name + CSV_SUFFIX)
    final_data_matrix.to_csv(output_path, index=False)
    print("File has been written to:", output_path)

def add_new_imputed_features(df, row, i):
    imput_anyanxiety = ['phx01__psd', 'phx01__pd_ag', 'phx01__pd_noag', 'phx01__specphob', 'phx01__soc_phob', 'phx01__gad_phx']
    val = 1 if sum(row[imput_anyanxiety] == 1) > 0 else 0
    df.at[i, 'imput_anyanxiety'] = val

    imput_bech = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hpanx', 'hrsd01__heng']
    df.at[i, 'imput_bech'] = np.sum(row.reindex(imput_bech))

    imput_maier = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hpanx', 'hrsd01__heng', 'hrsd01__hagit']
    df.at[i, 'imput_maier'] = np.sum(row.reindex(imput_maier))

    imput_santen = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hpanx', 'hrsd01__heng', 'hrsd01__hsuic']
    df.at[i, 'imput_santen'] = np.sum(row.reindex(imput_santen))

    imput_gibbons = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hpanx', 'hrsd01__heng', 'hrsd01__hsuic', 'hrsd01__hagit', 'hrsd01__hsanx', 'hrsd01__hsex']
    df.at[i, 'imput_gibbons'] = np.sum(row.reindex(imput_gibbons))

    imput_hamd7 = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hpanx', 'hrsd01__hsanx', 'hrsd01__ hengy', 'hrsd01__hsuicide']
    df.at[i, 'imput_hamd7'] = np.sum(row.reindex(imput_hamd7))

    imput_hamdret = ['hrsd01__hmdsd', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hsex']
    df.at[i, 'imput_hamdret'] = np.sum(row.reindex(imput_hamdret))

    imput_hamdanx = ['hrsd01__hpanx', 'hrsd01__hsanx', 'hrsd01__happt', 'hrsd01__hengy', 'hrsd01__hhypc']
    df.at[i, 'imput_hamdanx'] = np.sum(row.reindex(imput_hamdanx))

    imput_hamdsle = ['hrsd01__hsoin', 'hrsd01__hmnin', 'hrsd01__hemin']
    df.at[i, 'imput_hamdsle'] = np.sum(row.reindex(imput_hamdsle))

    imput_idsc5w0 = ['qids01_w0c__vmdsd', 'qids01_w0c__vintr', 'qids01_w0c__vengy', 'qids01_w0c__vvwsf', 'qids01_w0c__vslow']
    val_imput_idsc5w0 = np.sum(row.reindex(imput_idsc5w0))
    df.at[i, 'imput_idsc5w0'] = val_imput_idsc5w0

    imput_idsc5w2 = ['qids01_w2c__vmdsd', 'qids01_w2c__vintr', 'qids01_w2c__vengy', 'qids01_w2c__vvwsf', 'qids01_w2c__vslow']
    val_imput_idsc5w2 = np.sum(row.reindex(imput_idsc5w2))
    df.at[i, 'imput_idsc5w2'] = val_imput_idsc5w2

    val = round((val_imput_idsc5w2 - val_imput_idsc5w0) / val_imput_idsc5w0 if val_imput_idsc5w0 else 0, 3)
    df.at[i, 'imput_idsc5pccg'] = val

    val = round((row['qids01_w2c__qstot'] - row['qids01_w0c__qstot']) / row['qids01_w0c__qstot'] if row['qids01_w0c__qstot'] else 0, 3)
    df.at[i, 'imput_qidscpccg'] = val

    return df

def replace_with_median(df, col_names):
    if set(col_names).issubset(df.columns):
        df[col_names] = df[col_names].apply(lambda col: col.fillna(col.median()), axis=0)
        print("Imputed blanks with median")
    else:
        raise Exception("Column names are not subset: {}".format(set(col_names).difference(df.columns)))
    return df

def replace_with_mode(df, col_names):
    if set(col_names).issubset(df.columns):
        # df[col_names] = df[col_names].apply(lambda col: col.fillna(float(col.mode())), axis=0)
        for col in col_names:
            # As mode may (infrequently lol) result in more than one answer, randomly pick 1
            mode = df[col].mode()
            one_mode = float(mode.sample(n=1, random_state=1))
            df[col] = df[col].fillna(one_mode)

            #try:

            #except:
                #print(f'{col} is {df[col].dtype} and mode is {df[col].mode()}')
                #print(f'here is the bad col, with shape {df[col].shape}')

                #print(df[col])
                #raise Exception
            #print(f'{col} imputed!')
        #print("Imputed blanks with mode")
    else:
        raise Exception("Column names are not subset: {}".format(set(col_names).difference(df.columns)))
    return df

def replace(df, col_names, conversion_map):
    if set(col_names).issubset(df.columns):
        df[col_names] = df[col_names].replace(to_replace=conversion_map)
        print("Replaced", conversion_map)
    else:
        raise Exception("Column names are not subset: {}".format(set(col_names).difference(df.columns)))
    return df

def one_hot_encode(df, columns):
    # Convert categorical variables to indicator variables via one-hot encoding
    for col_name in columns:
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name, prefix_sep="||")], axis=1)
    return df

def drop_empty_columns(df):
    return df.dropna(axis="columns", how="all")  # Drop columns that are all empty


def generate_y(root_data_dir_path, holdout_set_ids, holdout_label='all'):
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    output_y_dir_path = os.path.join(output_dir_path, DIR_Y_MATRIX)

    print("\n--------------------------------7. Y MATRIX GENERATION-----------------------------------\n")
    y_wk8_rem_qids_c = pd.DataFrame()
    y_wk8_rem_qids_sr = pd.DataFrame()

    for filename in os.listdir(root_data_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_y_dir_path):
            os.mkdir(output_y_dir_path)

        scale_name = filename.split(".")[0]
        if scale_name not in ['ccv01', 'qids01']:
            continue

        curr_scale_path = root_data_dir_path + "/" + filename

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, sep='\t', skiprows=[1])
        scale_df = scale_df[scale_df[COL_NAME_SUBJECTKEY].isin(holdout_set_ids)]  # Only keep in data from this set


        print(LINE_BREAK)
        print("Handling scale = ", scale_name)

        # Reversed 0 and 1 from previous; 1 is now TRD, 0 is non-TRD, as we're predicting TRD. 
        if scale_name == "qids01":
            
            over21_df = scale_df.loc[scale_df['days_baseline'] > 21] # New, use this to check subjects remained 4 weeks.

            for vers in ['c', 'sr']:
                # Temporary dataframes to store TRD, remission, or response,
                # and then assign to either Clinician or Self-Report
                y_wk8_rem_qids01 = pd.DataFrame()
                y_wk8_resp_qids01 = pd.DataFrame()
                y_nolvl1drop_trdrem_qids01 = pd.DataFrame()
                
                if vers == 'c': 
                    version_form = 'Clinician'
                elif vers == 'sr': 
                    version_form = 'Self Rating'
                else:
                    Exception()
                
                i = 0
                for subject_id, group in scale_df.groupby(['subjectkey']):
                    if subject_id in over21_df['subjectkey'].values: # Only generate y if this subject stayed in
                        # study for 4 weeks

                        # Generate TRD status. Based closely on a snippet provided by Qingqin Li. It does seem to vary
                        # slight from the Nie et al paper.

                        # Make subset of entries for this subject in level 1 and in 2 or 2.1
                        subset_lvl1 = group[(group['version_form'] == version_form) & (group['level'] == "Level 1")]
                        subset_lvl2 = group[(group['version_form'] == version_form) & ((group['level'] == "Level 2") | (group['level'] == "Level 2.1"))]

                        # Drop NaN entries
                        subset_lvl1 = subset_lvl1[subset_lvl1['qstot'].notna()]
                        subset_lvl2 = subset_lvl2[subset_lvl2['qstot'].notna()]

                        if subset_lvl1.shape[0] == 0 and subset_lvl2.shape[0] == 0:
                            # Skip this subject if no entries in either
                            continue

                        # Check if remitted in Level 1
                        remission_lvl1 = 'MISSING'
                        if subset_lvl1.shape[0] != 0:
                            sorted_lvl1 = subset_lvl1.sort_values(by=['days_baseline'], ascending=False)
                            baseline_lvl1 = sorted_lvl1.iloc[-1]['qstot']
                            locf_lvl1 = sorted_lvl1.iloc[0]['qstot']
                            locf_lvl1_day = sorted_lvl1.iloc[0]['days_baseline']
                            # As week data is often missing, use greater than days_baseline 21 as a stand in for week 4

                            if baseline_lvl1 < 6:
                                ValueError('woh there was a baseline value less than 5!')

                            if locf_lvl1 <= 5 and locf_lvl1_day > 21 and baseline_lvl1 > 5:
                                remission_lvl1 = 'YES'
                            elif locf_lvl1 >= 6 and locf_lvl1_day > 21 and baseline_lvl1 > 5:
                                remission_lvl1 = 'NO'

                        # Level 2 Remission
                        remission_lvl2 = 'MISSING'
                        if subset_lvl2.shape[0] != 0:
                            sorted_lvl2 = subset_lvl2.sort_values(by=['days_baseline'], ascending=False)
                            baseline_lvl2 = sorted_lvl2.iloc[-1]['qstot']
                            locf_lvl2 = sorted_lvl2.iloc[0]['qstot']
                            baseline_lvl2_start_day = sorted_lvl2.iloc[-1]['days_baseline']
                            locf_lvl2_end_day = sorted_lvl2.iloc[0]['days_baseline']
                            lvl2_days_in = locf_lvl2_end_day - baseline_lvl2_start_day  # As week data is often
                            # missing, use difference in days_baseline instead, as over 21 days corresponds to 4 or
                            # more weeks according to the week data we do have

                            if baseline_lvl2 < 6:
                                ValueError('woh there was a baseline value less than 5! in Level 2!')

                            if locf_lvl2 <= 5 and baseline_lvl2 > 5:
                                remission_lvl2 = 'YES'
                            elif locf_lvl2 >= 6 and lvl2_days_in > 21 and baseline_lvl2 > 5:
                                remission_lvl2 = 'NO'

                        # Write out based on remission or reaching Level 2. Use same logic as provided by Nie et al
                        # code snippet, which could be simplified.
                        if remission_lvl1 == 'YES':
                            y_nolvl1drop_trdrem_qids01.loc[i, "subjectkey"] = subject_id
                            y_nolvl1drop_trdrem_qids01.loc[i, "target"] = 0
                        elif remission_lvl1 == 'NO' and remission_lvl2 == 'NO':
                            y_nolvl1drop_trdrem_qids01.loc[i, "subjectkey"] = subject_id
                            y_nolvl1drop_trdrem_qids01.loc[i, "target"] = 1
                        elif remission_lvl1 == 'NO' and remission_lvl2 == 'YES':
                            y_nolvl1drop_trdrem_qids01.loc[i, "subjectkey"] = subject_id
                            y_nolvl1drop_trdrem_qids01.loc[i, "target"] = 0
                        elif remission_lvl1 == 'MISSING' and remission_lvl2 == 'YES':
                            y_nolvl1drop_trdrem_qids01.loc[i, "subjectkey"] = subject_id
                            y_nolvl1drop_trdrem_qids01.loc[i, "target"] = 0

                        i += 1
    
                # Create Week 9 QIDS Response and Remission Targets, for use with CAN-BIND overlapping targets etc
                # Referred to as Week 8 as it is used to compare/ext validate  with CAN-BIND week 8
                i = 0
                for subject_id, group in scale_df.groupby(['subjectkey']):
                    if subject_id in over21_df['subjectkey'].values:  # Only generate y if this subject stayed in
                        # study for 4 weeks

                        # Grab the relevant entries between week 0 and week 8 in the study
                        subset_wk8 = group[(group['version_form'] == version_form) & (group['days_baseline'] <= 77)]
                        # Drop blank/NaN entries
                        subset_wk8 = subset_wk8[subset_wk8['qstot'].notna()]

                        # Skip subject (will be dropped) if does not have any relevant entries
                        if subset_wk8.shape[0] == 0:
                            continue

                        # If does, proceed to check for response or remission
                        if subset_wk8.shape[0] != 0:
                            sorted_wk8 = subset_wk8.sort_values(by=['days_baseline'], ascending=False)
                            baseline_wk8 = sorted_wk8.iloc[-1]['qstot']
                            locf_wk8 = sorted_wk8.iloc[0]['qstot']
                            locf_day = sorted_wk8.iloc[0]['days_baseline']

                            if baseline_wk8 < 6:
                                ValueError('woh there was a baseline value less than 5!')

                            # Check for remission, must be >21 days (wk4) to count either way,
                            # and then must achieve LOCF of 5 or less
                            if locf_wk8 <= 5 and locf_day > 21 and baseline_wk8 > 5:
                                y_wk8_rem_qids01.loc[i, "subjectkey"] = subject_id
                                y_wk8_rem_qids01.loc[i, "target"] = 1
                            elif locf_wk8 > 5 and locf_day > 21 and baseline_wk8 > 5:
                                y_wk8_rem_qids01.loc[i, "subjectkey"] = subject_id
                                y_wk8_rem_qids01.loc[i, "target"] = 0

                            # Check for response, must be <21 days (wk4) to count either way,
                            # and then must achieve LOCF of 50% or less of baseline
                            if locf_wk8 <= 0.5*baseline_wk8 and locf_day > 21 and baseline_wk8 > 5:
                                y_wk8_resp_qids01.loc[i, "subjectkey"] = subject_id
                                y_wk8_resp_qids01.loc[i, "target"] = 1
                            elif locf_wk8 > 0.5*baseline_wk8 and locf_day > 21 and baseline_wk8 > 5:
                                y_wk8_resp_qids01.loc[i, "subjectkey"] = subject_id
                                y_wk8_resp_qids01.loc[i, "target"] = 0

                    i += 1

                # Copy temporary targets to the correct QIDS version, self-rated or clinician rated
                if vers == 'c':
                    y_nolvl1drop_trdrem_qids01_c = y_nolvl1drop_trdrem_qids01
                    y_wk8_resp_qids_c = y_wk8_resp_qids01
                    y_wk8_rem_qids_c = y_wk8_rem_qids01
                elif vers == 'sr': 
                    y_nolvl1drop_trdrem_qids01_sr = y_nolvl1drop_trdrem_qids01
                    y_wk8_resp_qids_sr = y_wk8_resp_qids01
                    y_wk8_rem_qids_sr = y_wk8_rem_qids01
                else:
                    raise ValueError("Version must be c, clinician, or sr, self-rated")
            
    y_nolvl1drop_trdrem_qids01_c.to_csv(os.path.join(output_y_dir_path, "y_nolvl1drop_trdrem_qids01_c" + CSV_SUFFIX), index=False)
    y_nolvl1drop_trdrem_qids01_sr.to_csv(os.path.join(output_y_dir_path, "y_nolvl1drop_trdrem_qids01_sr" + CSV_SUFFIX), index=False)

    y_wk8_resp_qids_c.to_csv(os.path.join(output_y_dir_path, "y_wk8_resp_qids_c" + CSV_SUFFIX), index=False)
    y_wk8_resp_qids_sr.to_csv(os.path.join(output_y_dir_path, "y_wk8_resp_qids_sr" + CSV_SUFFIX), index=False)
    
    y_wk8_rem_qids_c.to_csv(os.path.join(output_y_dir_path, "y_wk8_rem_qids_c" + CSV_SUFFIX), index=False)
    y_wk8_rem_qids_sr.to_csv(os.path.join(output_y_dir_path, "y_wk8_rem_qids_sr" + CSV_SUFFIX), index=False)

    print("Y output files have  been written to:", output_y_dir_path)


def select_subjects(root_data_dir_path,  holdout_label='all'):
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    input_imputed_dir_path = os.path.join(output_dir_path, DIR_IMPUTED)
    input_y_generation_dir_path = os.path.join(output_dir_path, DIR_Y_MATRIX)
    input_row_selected_dir_path = os.path.join(output_dir_path, DIR_ROW_SELECTED)
    output_subject_selected_path = os.path.join(output_dir_path, DIR_SUBJECT_SELECTED)

    output_imputed_dir_path = os.path.join(output_dir_path, DIR_IMPUTED)
    print("\n--------------------------------8. SUBJECT SELECTION-----------------------------------\n")

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    if not os.path.exists(output_subject_selected_path):
        os.mkdir(output_subject_selected_path)

    orig_data_matrix = pd.read_csv(os.path.join(input_imputed_dir_path, "rs__cs__ohe__vc__ag__im__stard_data_matrix.csv"))

    # New final X matrices
    X_nolvl1drop_qids_c = orig_data_matrix
    X_nolvl1drop_qids_sr = orig_data_matrix
    X_tillwk4_qids_c = orig_data_matrix
    X_tillwk4_qids_sr = orig_data_matrix
    
    # Select subjects from imputed (aggregated) data based on the y matrices

    # Handle the TRD stuff
    
    y_nolvl1drop_trdrem_qids01_c = pd.read_csv(os.path.join(input_y_generation_dir_path, "y_nolvl1drop_trdrem_qids01_c" + CSV_SUFFIX))
    y_nolvl1drop_trdrem_qids01_sr = pd.read_csv(os.path.join(input_y_generation_dir_path, "y_nolvl1drop_trdrem_qids01_sr" + CSV_SUFFIX))

    X_nolvl1drop_qids_c__final = handle_subject_selection_conditions(input_row_selected_dir_path, X_nolvl1drop_qids_c, y_nolvl1drop_trdrem_qids01_c, 'c', pd.read_csv)
    X_nolvl1drop_qids_sr__final = handle_subject_selection_conditions(input_row_selected_dir_path, X_nolvl1drop_qids_sr, y_nolvl1drop_trdrem_qids01_sr, 'sr', pd.read_csv)

    # Subset the y matrices so that it matches the X matrices
    y_nolvl1drop_trdrem_qids01_c__final = y_nolvl1drop_trdrem_qids01_c[y_nolvl1drop_trdrem_qids01_c.subjectkey.isin(X_nolvl1drop_qids_c__final.subjectkey)]
    y_nolvl1drop_trdrem_qids01_sr__final = y_nolvl1drop_trdrem_qids01_sr[y_nolvl1drop_trdrem_qids01_sr.subjectkey.isin(X_nolvl1drop_qids_sr__final.subjectkey)]

    # Handle the week8 response stuff
    y_wk8_resp_qids_c = pd.read_csv(os.path.join(input_y_generation_dir_path, "y_wk8_resp_qids_c" + CSV_SUFFIX))
    y_wk8_resp_qids_sr = pd.read_csv(os.path.join(input_y_generation_dir_path, "y_wk8_resp_qids_sr" + CSV_SUFFIX))

    # Handle the week8 remission stuff
    y_wk8_rem_qids_c = pd.read_csv(os.path.join(input_y_generation_dir_path, "y_wk8_rem_qids_c" + CSV_SUFFIX))
    y_wk8_rem_qids_sr = pd.read_csv(os.path.join(input_y_generation_dir_path, "y_wk8_rem_qids_sr" + CSV_SUFFIX))

    # Handle the X matrices of subjects who stayed until week 4
    # These use one of the y's, but it doesn't matter as only uses for subject selection which is same between those y matrices
    X_tillwk4_qids_c__final = handle_subject_selection_conditions(input_row_selected_dir_path, X_tillwk4_qids_c, y_wk8_rem_qids_c, 'c', pd.read_csv)
    X_tillwk4_qids_sr__final = handle_subject_selection_conditions(input_row_selected_dir_path, X_tillwk4_qids_sr, y_wk8_rem_qids_sr, 'sr', pd.read_csv)

    # Subset the y matrices so that they matches the X matrices
    y_wk8_resp_qids_c__final = y_wk8_resp_qids_c[y_wk8_resp_qids_c.subjectkey.isin(X_tillwk4_qids_c__final.subjectkey)]
    y_wk8_resp_qids_sr__final = y_wk8_resp_qids_sr[y_wk8_resp_qids_sr.subjectkey.isin(X_tillwk4_qids_sr__final.subjectkey)]

    y_wk8_rem_qids_c__final = y_wk8_rem_qids_c[y_wk8_rem_qids_c.subjectkey.isin(X_tillwk4_qids_c__final.subjectkey)]
    y_wk8_rem_qids_sr__final = y_wk8_rem_qids_sr[y_wk8_rem_qids_sr.subjectkey.isin(X_tillwk4_qids_sr__final.subjectkey)]

    # Sort both X and y matrices by 'subject' to make sure they match; y should already be sorted by this
    X_nolvl1drop_qids_c__final = X_nolvl1drop_qids_c__final.sort_values(by=['subjectkey'])
    X_nolvl1drop_qids_sr__final = X_nolvl1drop_qids_sr__final.sort_values(by=['subjectkey'])
    X_tillwk4_qids_c__final = X_tillwk4_qids_c__final.sort_values(by=['subjectkey'])
    X_tillwk4_qids_sr__final = X_tillwk4_qids_sr__final.sort_values(by=['subjectkey'])
    
    y_nolvl1drop_trdrem_qids01_c__final = y_nolvl1drop_trdrem_qids01_c__final.sort_values(by=['subjectkey'])
    y_nolvl1drop_trdrem_qids01_sr__final = y_nolvl1drop_trdrem_qids01_sr__final.sort_values(by=['subjectkey'])
    y_wk8_resp_qids_c__final = y_wk8_resp_qids_c__final.sort_values(by=['subjectkey'])
    y_wk8_resp_qids_sr__final = y_wk8_resp_qids_sr__final.sort_values(by=['subjectkey'])
    y_wk8_rem_qids_c__final = y_wk8_rem_qids_c__final.sort_values(by=['subjectkey'])
    y_wk8_rem_qids_sr__final = y_wk8_rem_qids_sr__final.sort_values(by=['subjectkey'])

    # Make two new y matrices to evaluate performance if a different inclusion is used
    y_wk8_resp_qids_sr_nolvl1drop = y_wk8_resp_qids_sr__final[y_wk8_resp_qids_sr__final.subjectkey.isin(X_nolvl1drop_qids_sr__final.subjectkey)]
    y_wk8_resp_qids_c_nolvl1drop = y_wk8_resp_qids_c__final[y_wk8_resp_qids_c__final.subjectkey.isin(X_nolvl1drop_qids_c__final.subjectkey)]

    # Output X matrices to CSV
    X_nolvl1drop_qids_c__final.to_csv(os.path.join(output_subject_selected_path, "X_nolvl1drop_qids_c_" + holdout_label + CSV_SUFFIX), index=False)
    X_nolvl1drop_qids_sr__final.to_csv(os.path.join(output_subject_selected_path, "X_nolvl1drop_qids_sr_" + holdout_label + CSV_SUFFIX), index=False)
    X_tillwk4_qids_c__final.to_csv(os.path.join(output_subject_selected_path, "X_tillwk4_qids_c_" + holdout_label + CSV_SUFFIX), index=False)
    X_tillwk4_qids_sr__final.to_csv(os.path.join(output_subject_selected_path, "X_tillwk4_qids_sr_" + holdout_label + CSV_SUFFIX), index=False)

    # Output y matrices to CSV
    y_nolvl1drop_trdrem_qids01_c__final.to_csv(os.path.join(output_subject_selected_path, "y_nolvl1drop_trdrem_qids_c_" + holdout_label + CSV_SUFFIX), index=False)
    y_nolvl1drop_trdrem_qids01_sr__final.to_csv(os.path.join(output_subject_selected_path, "y_nolvl1drop_trdrem_qids_sr_" + holdout_label + CSV_SUFFIX), index=False)
    y_wk8_resp_qids_c__final.to_csv(os.path.join(output_subject_selected_path, "y_wk8_resp_qids_c_" + holdout_label + CSV_SUFFIX), index=False)
    y_wk8_resp_qids_sr__final.to_csv(os.path.join(output_subject_selected_path, "y_wk8_resp_qids_sr_" + holdout_label + CSV_SUFFIX), index=False)
    y_wk8_rem_qids_c__final.to_csv(os.path.join(output_subject_selected_path, "y_wk8_rem_qids_c_" + holdout_label + CSV_SUFFIX), index=False)
    y_wk8_rem_qids_sr__final.to_csv(os.path.join(output_subject_selected_path, "y_wk8_rem_qids_sr_" + holdout_label + CSV_SUFFIX), index=False)

    y_wk8_resp_qids_sr_nolvl1drop.to_csv(os.path.join(output_subject_selected_path, "y_wk8_resp_qids_sr_nolvl1drop_" + holdout_label + CSV_SUFFIX), index=False)
    y_wk8_resp_qids_c_nolvl1drop.to_csv(os.path.join(output_subject_selected_path, "y_wk8_resp_qids_c_nolvl1drop_" + holdout_label + CSV_SUFFIX), index=False)
    
    print("Files written to: ", output_subject_selected_path)

def handle_subject_selection_conditions(input_row_selected_dir_path, X, y_df, qids_version, read_csv_filter):
    # New subject selection handling function, will select based on qids version ('sr' or 'c')
    
    # Select subjects with corresponding y values
    y = y_df.dropna(axis='rows') # Drop subjects lacking a y value
    X = X[X["subjectkey"].isin(y["subjectkey"])]
    
    # Select subjects that have ucq entries, aka eliminate subjects that don't have ucq entries, as a proxy for the small amount of subjects missing most patients. 
    file_ucq = pd.read_csv(input_row_selected_dir_path + "/rs__ucq01" + CSV_SUFFIX)
    X = X[X["subjectkey"].isin(file_ucq["subjectkey"])]
    
    # Eliminate subjects that don't have week0 QIDS entries from either QIDS-C or QIDS-SR
    file_qids01_w0c = pd.read_csv(input_row_selected_dir_path + "/rs__qids01_w0" + qids_version + CSV_SUFFIX)
    X = X[X["subjectkey"].isin(file_qids01_w0c["subjectkey"])]

    return X


def get_row(scale_df, subjectkey):
    return scale_df[(scale_df["level"] == "Level 1") & (scale_df["subjectkey"] == subjectkey)
                    & (scale_df["inc_curr"].notnull()
                    | scale_df["assist"].notnull()
                    | scale_df["unempl"].notnull()
                    | scale_df["otherinc"].notnull()
                    | scale_df["totincom"].notnull())]




