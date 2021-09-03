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

It is provided the holdout set label (holdout, non_holdout, all), as well as the subject ids corresponding, and generates 
all from there  

Takes 2 Arguments on command-line:
    Directory containing all the raw STAR*D data from the NDA
    
    Run-option. See main for complete list, allows only one part of the 
    preprocessing to be ran at a time. Use "--run-all" or "-a" to 
    do the entire preprocessing. 

Example terminal command:
"""
# python stard_preprocessor.py C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\stard_data -a


def separate_holdout_ids(data_dir):
    qids_df = pd.read_csv(os.path.join(data_dir, 'qids01.txt'), sep='\t', skiprows=[1])
    qids_subj_ids = qids_df[COL_NAME_SUBJECTKEY].unique().tolist()  # Get set of all subject ids

    split_non_holdout_ids, split_holdout_ids = train_test_split(qids_subj_ids, test_size=0.20, random_state=5)
    print(f'Number of subjects in each set {len(qids_subj_ids)}, {len(split_holdout_ids)}, {len(split_non_holdout_ids)}')
    return qids_subj_ids, split_non_holdout_ids, split_holdout_ids

if __name__ == "__main__":
    data_dir_path = sys.argv[1]
    option = sys.argv[2]
    ho_option = sys.argv[3]
    is_valid = len(sys.argv) == 4 and os.path.isdir(data_dir_path)

    entire_ids, non_holdout_ids, holdout_ids, = separate_holdout_ids(data_dir_path)

    if ho_option == '-all':
        ho_labels = ['entire', 'holdout', 'non_holdout']
        ho_filters = [entire_ids, holdout_ids, non_holdout_ids]
    elif ho_option == '-ho':
        ho_labels = ['holdout']
        ho_filters = [holdout_ids]
    elif ho_option == '-non_ho':
        ho_labels = ['non_holdout']
        ho_filters = [non_holdout_ids]
    elif ho_option == '-entire':
        ho_labels = ['entire']
        ho_filters = [entire_ids]
    else:
        raise Exception("Enter valid argument for holdout option\n"
                        "\t path: the path to a real directory\n"
                        "\t option: preprocessing option, e.g. -a\n"
                        "\t ho_option: holdout set option e.g. all\n"
                        "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3 -a -all")

    for holdout_label, filtered_ids in zip(ho_labels, ho_filters):
        #read_csv_filter = create_read_csv_filter(filtered_ids)

        if is_valid and option in ["--row-select", "-rs"]:
            select_rows(data_dir_path, filtered_ids, holdout_label)

        elif is_valid and option in ["--column-select", "-cs"]:
            select_columns(data_dir_path,  holdout_label)

        elif is_valid and option in ["--one-hot-encode", "-ohe"]:
            one_hot_encode_scales(data_dir_path,  holdout_label)

        elif is_valid and option in ["--value-convert", "-vc"]:
            convert_values(data_dir_path,  holdout_label)

        elif is_valid and option in ["--aggregate-rows", "-ag"]:
            aggregate_rows(data_dir_path,  holdout_label)

        elif is_valid and option in ["--impute", "-im"]:
            impute(data_dir_path, filtered_ids, holdout_label)

        elif is_valid and option in ["--y-generation", "-y"]:
            generate_y(data_dir_path, filtered_ids, holdout_label)

        elif is_valid and option in ["--subject-select", "-ss"]:
            select_subjects(data_dir_path,  holdout_label)

        elif is_valid and option in ["--overlapping", "-ov"]:
            convert_stard_to_overlapping(data_dir_path, holdout_label)

        elif is_valid and option in ["--run-all", "-a"]:
            select_rows(data_dir_path, filtered_ids,  holdout_label)
            select_columns(data_dir_path,  holdout_label)
            one_hot_encode_scales(data_dir_path,  holdout_label)
            convert_values(data_dir_path,  holdout_label)
            aggregate_rows(data_dir_path,  holdout_label)
            impute(data_dir_path, filtered_ids, holdout_label)
            generate_y(data_dir_path, filtered_ids, holdout_label)
            select_subjects(data_dir_path,  holdout_label)
            # Will have the creation of overlapping data commented out for now, as needs the CAN-BIND data
            # convert_stard_to_overlapping(data_dir_path, holdout_label

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

        else:
            raise Exception("Enter valid argument for holdout option\n"
                            "\t path: the path to a real directory\n"
                            "\t option: preprocessing option, e.g. -a\n"
                            "\t ho_option: holdout set option e.g. all\n"
                            "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3 -a "
                            "-all")
