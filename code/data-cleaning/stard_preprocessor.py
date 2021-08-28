import sys
import os
import pandas as pd
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

Example Run configuration:
runfile('C:/Users/jjnun/Documents/Sync/Research/1_CANBIND_Replication/teyden-git/code/data-cleaning/stard_preprocessing_manager.py', args='C:/Users/jjnun/Documents/Sync/Research/1_CANBIND_Replication/teyden-git/data/stard_data -a', wdir='C:/Users/jjnun/Documents/Sync/Research/1_CANBIND_Replication/teyden-git/code/data-cleaning')

"""
from stard_preprocessing_manager import select_rows, select_columns, one_hot_encode_scales, convert_values, \
    aggregate_rows, impute, generate_y, select_subjects


def create_read_csv_filter(holdout_label, data_dir_path):
    """
    Creates a function which calls read_csv, and then filters the ensuing subject rows to match the holdout filter.
    :param data_dir_path: path where the raw STAR*D data is located
    :param holdout_label: string, either holdout, non_holdout, or all
    :return: a function as above
    """

    def read_csv_filter(f, *args, **kwargs):
        df = pd.read_csv(f, *args, **kwargs)
        # TODO: create actual filter lol.
        filtered_df = df
        return filtered_df

    return read_csv_filter


if __name__ == "__main__":
    data_dir_path = sys.argv[1]
    option = sys.argv[2]
    is_valid = len(sys.argv) == 3 and os.path.isdir(data_dir_path)

    for holdout_label in ['holdout', 'non_holdout', 'all']:
        read_csv_filter = create_read_csv_filter(holdout_label, data_dir_path)

        if is_valid and option in ["--row-select", "-rs"]:
            select_rows(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--column-select", "-cs"]:
            select_columns(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--one-hot-encode", "-ohe"]:
            one_hot_encode_scales(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--value-convert", "-vc"]:
            convert_values(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--aggregate-rows", "-ag"]:
            aggregate_rows(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--impute", "-im"]:
            impute(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--y-generation", "-y"]:
            generate_y(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--subject-select", "-ss"]:
            select_subjects(data_dir_path, read_csv_filter, holdout_label)

        elif is_valid and option in ["--run-all", "-a"]:
            select_rows(data_dir_path, read_csv_filter, holdout_label)
            select_columns(data_dir_path, read_csv_filter, holdout_label)
            one_hot_encode_scales(data_dir_path, read_csv_filter, holdout_label)
            convert_values(data_dir_path, read_csv_filter, holdout_label)
            aggregate_rows(data_dir_path, read_csv_filter, holdout_label)
            impute(data_dir_path, read_csv_filter, holdout_label)
            generate_y(data_dir_path, read_csv_filter, holdout_label)
            select_subjects(data_dir_path, read_csv_filter, holdout_label)

            print("\nSteps complete:\n" +
                  "\t Row selection\n" +
                  "\t Column selection\n" +
                  "\t One-hot encoding\n" +
                  "\t Value conversion\n" +
                  "\t Row aggregation (generate a single matrix)\n" +
                  "\t Imputation of missing values\n" +
                  "\t Generation of y matrices\n" +
                  "\t Subject selection\n")

    else:
        raise Exception("Enter valid arguments\n"
                        "\t path: the path to a real directory\n"
                        "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3 -a")
