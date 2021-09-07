import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from stard_preprocessing_globals import COL_NAME_SUBJECTKEY
from stard_preprocessing_manager import select_rows, select_columns, one_hot_encode_scales, convert_values, \
    aggregate_rows, impute, generate_y, select_subjects
from generate_overlapping_features import convert_stard_to_overlapping

"""
This is our preprocess for the raw STAR*D data from the NIMH, producing the 
preprocessed STAR*D data, which can be used for ML or further processed into
the dataset overlapping with CAN-BIND, etc.

It is provided the holdout set label (holdout, non_holdout, all), as well as the subject ids corresponding, 
and generates all from there 

Takes 3 Arguments on command-line:
    Directory containing all the raw STAR*D data from the NDA
    
    Holdout set selection (all, entire, ho, non_ho) to choose which sets are being made. Entire is all data, while
    holdout and non-holdout are those. 
    
    Run-option. See main for complete list, allows only one part of the 
    preprocessing to be ran at a time. Use "--run-all" or "-a" to 
    do the entire preprocessing. 

Example terminal command:
"""
# python stard_preprocessor.py C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data
# \stard_data -a -all


def separate_holdout_ids(data_dir):
    qids_df = pd.read_csv(os.path.join(data_dir, 'qids01.txt'), sep='\t', skiprows=[1])
    qids_subj_ids = qids_df[COL_NAME_SUBJECTKEY].unique().tolist()  # Get set of all subject ids

    split_non_holdout_ids, split_holdout_ids = train_test_split(qids_subj_ids, test_size=0.20, random_state=5)
    print(
        f'Number of subjects in each set {len(qids_subj_ids)}, {len(split_holdout_ids)}, {len(split_non_holdout_ids)}')
    return qids_subj_ids, split_non_holdout_ids, split_holdout_ids


def process_set(set_label, set_ids, processing_option):
    if processing_option in ["--row-select", "-rs"]:
        select_rows(data_dir_path, set_ids, set_label)

    elif processing_option in ["--column-select", "-cs"]:
        select_columns(data_dir_path, set_label)

    elif processing_option in ["--one-hot-encode", "-ohe"]:
        one_hot_encode_scales(data_dir_path, set_label)

    elif processing_option in ["--value-convert", "-vc"]:
        convert_values(data_dir_path, set_label)

    elif processing_option in ["--aggregate-rows", "-ag"]:
        aggregate_rows(data_dir_path, set_label)

    elif processing_option in ["--impute", "-im"]:
        impute(data_dir_path, set_ids, set_label)

    elif processing_option in ["--y-generation", "-y"]:
        generate_y(data_dir_path, set_ids, set_label)

    elif processing_option in ["--subject-select", "-ss"]:
        select_subjects(data_dir_path, set_label)

    elif processing_option in ["--overlapping", "-ov"]:
        convert_stard_to_overlapping(data_dir_path, set_label)
        print("\t Overlapping datasets with CAN-BIND also generated\n")

    elif processing_option in ["--run-all", "-a"]:
        select_rows(data_dir_path, set_ids, set_label)
        select_columns(data_dir_path, set_label)
        one_hot_encode_scales(data_dir_path, set_label)
        convert_values(data_dir_path, set_label)
        aggregate_rows(data_dir_path, set_label)
        impute(data_dir_path, set_ids, set_label)
        generate_y(data_dir_path, set_ids, set_label)
        select_subjects(data_dir_path, set_label)
        # Will have the creation of overlapping data commented out for now, as needs the CAN-BIND data
        # convert_stard_to_overlapping(data_dir_path, set_label

        print("\nSteps complete:\n" +
              "\t Row selection\n" +
              "\t Column selection\n" +
              "\t One-hot encoding\n" +
              "\t Value conversion\n" +
              "\t Row aggregation (generate a single matrix)\n" +
              "\t Imputation of missing values\n" +
              "\t Generation of y matrices\n" +
              "\t Subject selection\n" +
              "\t Did not create overlapping datasets, use -ov to do so\n")

    elif processing_option in ["--run-all-and-ov", '-aov']:
        process_set(set_label, set_ids, '-a')
        process_set(set_label, set_ids, '-ov')
    else:
        raise Exception(f"Enter valid argument for holdout option, {processing_option} is not valid\n"
                        "\t path: the path to a real directory\n"
                        "\t option: preprocessing option, e.g. -a\n"
                        "\t ho_option: holdout set option e.g. all\n"
                        "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3 -a "
                        "-all")


if __name__ == "__main__":
    data_dir_path = sys.argv[1]
    option = sys.argv[2]
    ho_option = sys.argv[3]
    is_valid = len(sys.argv) == 4 and os.path.isdir(data_dir_path)

    entire_ids, non_holdout_ids, holdout_ids, = separate_holdout_ids(data_dir_path)

    assert os.path.isdir(data_dir_path), f"{data_dir_path} is not a valid path"
    assert len(sys.argv) == 4, f'Expected 3 command-line arguments: ' \
                               f'"\t path: the path to a real directory\n"' \
                               f'"\t option: preprocessing option, e.g. -a\n"' \
                               f'"\t ho_option: holdout set option e.g. all\n"' \
                               f'"\t e.g. python stard_preprocessing_manager.py ' \
                               f'/Users/teyden/Downloads/stardmarch19v3 -a -all'

    if ho_option == '-all':
        process_set('entire', entire_ids, option)
        process_set('holdout', holdout_ids, option)
        process_set('non_holdout', non_holdout_ids, option)
    elif ho_option == '-ho':
        process_set('holdout', holdout_ids, option)
    elif ho_option == '-non_ho':
        process_set('non_holdout', non_holdout_ids, option)
    elif ho_option == '-entire':
        process_set('entire', entire_ids, option)
    else:
        raise Exception("Enter valid argument for holdout option\n"
                        "\t must be one of all, entire, ho, non_ho")
