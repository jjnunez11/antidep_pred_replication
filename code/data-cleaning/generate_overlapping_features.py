import os
import sys
import pandas as pd
import numpy as np

from overlapping_globals import HEADER_CONVERSION_DICT, CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP, NEW_FEATURES_CANBIND, \
    QIDS_STARD_TO_CANBIND_DICT as Q_DICT_C
from overlapping_globals import STARD_OVERLAPPING_VALUE_CONVERSION_MAP, STARD_TO_DROP

"""
Generates the overlapping datasets from the processed datasets, for canbind data, thats the product of canbind_imputer

Takes in the processed dataset, and generates the overlapping datasets
Can generate an overlapping dataset one at a time if needed, or 
by default will do both

See main() for description of arguments

"""
from stard_preprocessing_globals import DIR_PROCESSED_DATA, DIR_SUBJECT_SELECTED, CSV_SUFFIX

Q_DICT_S = dict(map(reversed, Q_DICT_C.items()))


def convert_stard_to_overlapping(root_data_dir_path, holdout_label):
    output_dir_path = os.path.join(root_data_dir_path, DIR_PROCESSED_DATA, holdout_label)
    # Write to same dir as subject selection, as that's where the other final files are
    subject_selected_path = os.path.join(output_dir_path, DIR_SUBJECT_SELECTED)

    df = pd.read_csv(os.path.join(subject_selected_path, "X_tillwk4_qids_sr_" + holdout_label + CSV_SUFFIX))

    # Take whitelist columns first
    df = df[STARD_OVERLAPPING_VALUE_CONVERSION_MAP["whitelist"]]  ## + ["days_baseline"]]

    # Warn if any null found, as they should not be!
    check_missing_values(df)

    # Then process them
    for case, config in STARD_OVERLAPPING_VALUE_CONVERSION_MAP.items():
        if case == "keep":
            # Nothing to do, already grabbed from whitelist
            continue
        elif case == "multiply":
            for col_name, multiple in config["col_names"].items():
                df[col_name] = df[col_name].apply(lambda x: x * multiple)
        elif case == "other":
            for i, row in df.iterrows():
                # phx01__alcoh set 1 if either 1
                if row["phx01__alcoh||2.0"] == 1:
                    df.at[i, "phx01__alcoh||1.0"] = 1
                # phx01__bulimia||2/5: set 1 if any of bulimia one-hots
                bullemias = ['phx01__bulimia||3', 'phx01__bulimia||4']
                set_if_found_in_others(i, row, 'phx01__bulimia||2/5', bullemias, 1, df)
                # phx01__amphet||1.0: set 1 if any of the non alchol substance uses
                non_alch_sub = ['phx01__amphet||2.0', 'phx01__cannibis||1.0', 'phx01__cannibis||2.0',
                                'phx01__opioid||1.0', 'phx01__opioid||2.0', 'phx01__ax_cocaine||1.0',
                                'phx01__ax_cocaine||2.0']
                set_if_found_in_others(i, row, 'phx01__amphet||1.0', non_alch_sub, 1, df)
                # phx01__dep: 1 if any family history conditions are 1 
                family_hxs = ['phx01__deppar', 'phx01__depsib', 'phx01__depchld', 'phx01__bip', 'phx01__bippar',
                              'phx01__bipsib', 'phx01__bipchld', 'phx01__alcohol', 'phx01__alcpar', 'phx01__alcsib',
                              'phx01__alcchld', 'phx01__drug_phx', 'phx01__drgpar', 'phx01__drgsib', 'phx01__drgchld',
                              'phx01__suic_phx', 'phx01__suicpar', 'phx01__suicsib', 'phx01__suicchld']
                set_if_found_in_others(i, row, 'phx01__dep', family_hxs, 1, df)
                # dm01_enroll__empl||3.0 to 1 if any full time employment options
                full_time_statuses = ["dm01_enroll__empl||4.0", "dm01_enroll__empl||5.0"]
                set_if_found_in_others(i, row, 'dm01_enroll__empl||3.0', full_time_statuses, 1, df)

                # dm01_w0__totincom: Converts monthly usd old income to current CAD, and then categorizes per canbind
                totincom_convert = row[
                                       "dm01_w0__totincom"] * 12 * 1.25 * 1.19  # Converts to annual, then inflation, then USDtoCAD
                if totincom_convert < 10000:
                    df.at[i, "dm01_w0__totincom"] = 1
                elif totincom_convert < 25000:
                    df.at[i, "dm01_w0__totincom"] = 2
                elif totincom_convert < 50000:
                    df.at[i, "dm01_w0__totincom"] = 3
                elif totincom_convert < 75000:
                    df.at[i, "dm01_w0__totincom"] = 4
                elif totincom_convert < 100000:
                    df.at[i, "dm01_w0__totincom"] = 5
                elif totincom_convert < 150000:
                    df.at[i, "dm01_w0__totincom"] = 6
                elif totincom_convert < 200000:
                    df.at[i, "dm01_w0__totincom"] = 7
                elif totincom_convert >= 200000:
                    df.at[i, "dm01_w0__totincom"] = 8

                # Add in new features
                add_new_imputed_features_stard(df, row, i)  # fill in new features

    # Eliminate subjects that don't have any records > 21 This section removed, should be done already beforehand in generation of X_stard
    ##df = eliminate_early_leavers(df)
    ##df = df.drop(["days_baseline"], axis=1)

    # Drop columns that were used for calcs above
    for to_drop in STARD_TO_DROP:
        df = df.drop([to_drop], axis=1)

    # Rename Column Headers according to dict
    df = df.rename(HEADER_CONVERSION_DICT, axis=1)

    # Check that all column headers have ::: to ensure they have correspondance in CAN-BIND
    for header in list(df.columns.values):
        if not (':::' in header):
            print('Warning! Likely unwanted column in output: ' + header)

    # Sort and output
    df = df.reset_index(drop=True)
    df = df.sort_index(axis=1)  # Newly added, sorts columns alphabetically so same for both matrices
    # df = df.sort_values(by=['SUBJLABEL:::subjectkey'])
    df = df.set_index(['SUBJLABEL:::subjectkey'])
    df.to_csv(os.path.join(subject_selected_path, "X_tillwk4_overlap_qids_sr_" + holdout_label + CSV_SUFFIX),
              index=True)


def convert_canbind_to_overlapping(root_dir):
    data_dir = os.path.join(root_dir, "processed_data")
    out_dir_final_xy_data_matrices = os.path.join(data_dir, "final_xy_data_matrices\\")

    input_filename = os.path.join(out_dir_final_xy_data_matrices, "X_tillwk4_canbind.csv")
    try:
        orig_df = pd.read_csv(input_filename)
    except:
        raise Exception(f'Could not read X_tillwk4_canbind.csv in {out_dir_final_xy_data_matrices}')

    # Drop index column in present
    if "Unnamed: 0" in orig_df:
        df = orig_df.drop(["Unnamed: 0"], axis=1)
    else:
        df = orig_df

    # Take whitelist columns first
    df = df[CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP["whitelist"]]

    # Warn if any NaN found, as they should not be!
    check_missing_values(df)

    # Add new features as blank
    for new_feature in NEW_FEATURES_CANBIND:
        df[new_feature] = np.nan

    # Then process them
    for case, config in CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP.items():
        if case == "keep":
            # Nothing to do, already grabbed from whitelist
            continue
        elif case == "multiply":
            for col_name, multiple in config["col_names"].items():
                df[col_name] = df[col_name].apply(lambda x: x * multiple)
        elif case == "other":
            for i, row in df.iterrows():
                if row["MINI_SBSTNC_DPNDC_NONALCHL_TIME"] == 1:
                    df.at[i, "MINI_SBSTNC_ABUSE_NONALCHL_TIME"] = 1
                if row["MINI_ALCHL_DPNDC_TIME"] == 1:
                    df.at[i, "MINI_ALCHL_ABUSE_TIME"] = 1
                if row["MINI_AN_TIME"] == 1:
                    df.at[i, "MINI_AN_BINGE_TIME"] = 1
                if (row['EMPLOY_STATUS_6.0'] == 1) or (row['EMPLOY_STATUS_3.0'] == 1):
                    df.at[i, "EMPLOY_STATUS_1.0"] = 1
                if row['EMPLOY_STATUS_4.0'] == 1:
                    df.at[i, "EMPLOY_STATUS_2.0"] = 1
                add_new_imputed_features_canbind(df, row, i)  # fill in new features

    # Drop columns that were used for calcs above
    for todrop in ["MINI_SBSTNC_DPNDC_NONALCHL_TIME", "MINI_ALCHL_DPNDC_TIME", "MINI_AN_TIME", "EMPLOY_STATUS_6.0",
                   "EMPLOY_STATUS_3.0", "EMPLOY_STATUS_4.0"]:
        df = df.drop([todrop], axis=1)

    # Filter out those without valid response/nonresponse values
    ## Already filtered so ignore
    ##df = get_valid_subjects(df)
    ##df = df.drop(["RESPOND_WK8"], axis=1)

    # Rename Column Headers according to dict
    df = df.rename(HEADER_CONVERSION_DICT, axis=1)

    # Check that all column headers have ::: to ensure they have correspondance in STAR*D
    for header in list(df.columns.values):
        if not (':::' in header):
            print('Warning! Likely unwanted column in output: ' + header)

    # Sort and output
    df = df.reset_index(drop=True)
    df = df.sort_index(axis=1)  # Newly added, sorts columns alphabetically so same for both matrices
    df = df.set_index(['SUBJLABEL:::subjectkey'])
    df.to_csv(out_dir_final_xy_data_matrices + "X_tillwk4_overlap_canbind.csv", index=True)


def add_new_imputed_features_canbind(df, row, i):
    # imput_anyanxiety
    imput_anyanxiety = ['MINI_PTSD_TIME', 'MINI_PD_DX', 'MINI_AGRPHOBIA_TIME', 'MINI_SOCL_PHOBIA_DX', 'MINI_GAD_TIME']
    val = 1 if sum(row[imput_anyanxiety] == 1) > 0 else 0
    df.at[i, ':::imput_anyanxiety'] = val

    # imput_QIDS_SR_perc_change
    val = round((row[Q_DICT_C['qids01_w2sr__qstot']] - row[Q_DICT_C['qids01_w0sr__qstot']]) / row[
        Q_DICT_C['qids01_w0sr__qstot']] if row[Q_DICT_C['qids01_w0sr__qstot']] else 0, 3)
    df.at[i, 'imput_QIDS_SR_perc_change:::'] = val

    # Imputed new QIDS features
    for time in ['week0', 'week2']:
        time2 = 'baseline' if time == 'week0' else 'week2'  # week0 is sometimes called _baseline

        # imput_QIDS_SR_sleep_domain
        val = round(np.nanmax(
            list(row[['QIDS_SR_1_' + time2, 'QIDS_SR_2_' + time2, 'QIDS_SR_3_' + time2, 'QIDS_SR_4_' + time2]])))
        df.at[i, 'imput_QIDS_SR_sleep_domain_' + time + ':::'] = val

        # imput_QIDS_SR_appetite_domain
        val = round(np.nanmax(
            list(row[['QIDS_SR_6_' + time2, 'QIDS_SR_7_' + time2, 'QIDS_SR_8_' + time2, 'QIDS_SR_9_' + time2]])))
        df.at[i, 'imput_QIDS_SR_appetite_domain_' + time + ':::'] = val

        # imput_QIDS_SR_psychomot_domain
        val = round(np.nanmax(list(row[['QIDS_SR_15_' + time2, 'QIDS_SR_16_' + time2]])))
        df.at[i, 'imput_QIDS_SR_psychomot_domain_' + time + ':::'] = val

        # imput_QIDS_SR_overeating
        val = round(np.nanmax(list(row[['QIDS_SR_7_' + time2, 'QIDS_SR_9_' + time2]])))
        df.at[i, 'imput_QIDS_SR_overeating_' + time + ':::'] = val

        # imput_QIDS_SR_insomnia
        val = round(np.nanmax(list(row[['QIDS_SR_1_' + time2, 'QIDS_SR_2_' + time2, 'QIDS_SR_3_' + time2]])))
        df.at[i, 'imput_QIDS_SR_insomnia_' + time + ':::'] = val


def add_new_imputed_features_stard(df, row, i):
    # imput_anyanxiety
    imput_anyanxiety = ['phx01__psd', 'phx01__pd_noag', 'phx01__pd_ag', 'phx01__soc_phob', 'phx01__gad_phx',
                        'phx01__specphob']
    val = 1 if sum(row[imput_anyanxiety] == 1) > 0 else 0
    df.at[i, ':::imput_anyanxiety'] = val

    # imput_QIDS_SR_perc_change
    val = round((row['qids01_w2sr__qstot'] - row['qids01_w0sr__qstot']) / row['qids01_w0sr__qstot'] if row[
        'qids01_w0sr__qstot'] else 0, 3)
    df.at[i, 'imput_QIDS_SR_perc_change:::'] = val

    # PSYHIS_MDD_PREV:::
    val = 1 if row['phx01__epino'] >= 2 else 0
    df.at[i, 'PSYHIS_MDD_PREV:::'] = val

    # 'QLESQA_TOT_QLESQB_TOT_merged:::'
    val = 0
    for j in range(1, 15):  # Sum items 1-14
        str_j = str(j)
        str_j = "0" + str_j if j < 10 else str_j
        val = val + row['qlesq01__qlesq' + str_j]
    df.at[i, 'QLESQA_TOT_QLESQB_TOT_merged:::'] = val

    # Imputed new QIDS features
    for time in ['week0', 'week2']:
        time2 = 'baseline' if time == 'week0' else 'week2'  # week0 is sometimes called _baseline
        time3 = 'w0' if time == 'week0' else 'w2'  # week0 is sometimes called w0, and week2 w2

        # imput_QIDS_SR_sleep_domain
        val = round(np.nanmax(list(row[[Q_DICT_S['QIDS_SR_1_' + time2], Q_DICT_S['QIDS_SR_2_' + time2],
                                        Q_DICT_S['QIDS_SR_3_' + time2], Q_DICT_S['QIDS_SR_4_' + time2]]])))
        df.at[i, 'imput_QIDS_SR_sleep_domain_' + time + ':::'] = val

        # imput_QIDS_SR_appetite_domain
        val = round(np.nanmax(list(row[[Q_DICT_S['QIDS_SR_6_' + time2], Q_DICT_S['QIDS_SR_7_' + time2],
                                        Q_DICT_S['QIDS_SR_8_' + time2], Q_DICT_S['QIDS_SR_9_' + time2]]])))
        df.at[i, 'imput_QIDS_SR_appetite_domain_' + time + ':::'] = val

        # imput_QIDS_SR_psychomot_domain
        val = round(np.nanmax(list(row[[Q_DICT_S['QIDS_SR_15_' + time2], Q_DICT_S['QIDS_SR_16_' + time2]]])))
        df.at[i, 'imput_QIDS_SR_psychomot_domain_' + time + ':::'] = val

        # imput_QIDS_SR_overeating
        val = round(np.nanmax(list(row[[Q_DICT_S['QIDS_SR_7_' + time2], Q_DICT_S['QIDS_SR_9_' + time2]]])))
        df.at[i, 'imput_QIDS_SR_overeating_' + time + ':::'] = val

        # imput_QIDS_SR_insomnia
        val = round(np.nanmax(list(
            row[[Q_DICT_S['QIDS_SR_1_' + time2], Q_DICT_S['QIDS_SR_2_' + time2], Q_DICT_S['QIDS_SR_3_' + time2]]])))
        df.at[i, 'imput_QIDS_SR_insomnia_' + time + ':::'] = val

        # imput_QIDS_SR_ATYPICAL
        val = round(np.sum(list(row[['qids01_' + time3 + 'sr__vhysm', 'qids01_' + time3 + 'sr__vapin',
                                     'qids01_' + time3 + 'sr__vwtin', 'qids01_' + time3 + 'sr__vengy']])))
        df.at[i, 'QIDS_ATYPICAL_' + time2 + ':::'] = val


def set_if_found_in_others(i, row, to_set, others, val, df):
    """
    Quick helper function. Sets row[to_set][i] to value if 
    any of the values in others ==1.
    Useful whem collapsing a one-hot set of columns to 1 if 
    any of the columns are 1. 
    """
    for c in others:
        other_val = int(row[c])
        if other_val == 1:
            df.at[i, to_set] = val
        elif other_val == 0:
            continue
        else:
            raise Exception('set_if_found should only be used for columns that are 1 or 0, given: ' + str(other_val))


def check_missing_values(df):
    nulls = df.isnull().sum().sum()
    if nulls != 0:
        print("WARNING! A total of: " + str(nulls) + " missing values found, this should likely be 0!")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-bothdefault":
        convert_stard_to_overlapping()
        convert_canbind_to_overlapping()

    elif len(sys.argv) == 4 and sys.argv[1] == "-both" and os.path.isdir(sys.argv[2]):
        convert_stard_to_overlapping(sys.argv[2])
        convert_canbind_to_overlapping(sys.argv[3])

    elif len(sys.argv) == 3 and sys.argv[1] == "-sd" and os.path.isdir(sys.argv[2]):
        convert_stard_to_overlapping(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "-cb" and os.path.isdir(sys.argv[2]):
        convert_canbind_to_overlapping(sys.argv[2])

    else:
        print("Enter valid arguments\n"
              "\t options: -both to generate overlapping for both, -sd for STAR*D, -cb for CAN-BIND\n"
              "\t path: the path to a real directory with the preprocessed data\n"
              "\t path: if generating the overlapping for both, this will be the CAN-BIND data, and the 2nd argument STAR*D")
