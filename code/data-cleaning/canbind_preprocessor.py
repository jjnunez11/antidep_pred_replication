import sys
import os
from canbind_preprocessing_manager import aggregate_and_clean
from canbind_ygen import y_gen
from canbind_imputer import impute
from generate_overlapping_features import convert_canbind_to_overlapping


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        pathData = sys.argv[1]
        aggregate_and_clean(pathData, verbose=False, extra=False)
        y_gen(pathData)
        impute(pathData)
        convert_canbind_to_overlapping(pathData)

    elif len(sys.argv) == 3 and sys.argv[1] == "-v" and os.path.isdir(sys.argv[2]):
        pathData = sys.argv[2]
        aggregate_and_clean(pathData, verbose=True, extra=False)
        y_gen(pathData)
        impute(pathData)

    elif len(sys.argv) == 3 and sys.argv[1] == "-v+" and os.path.isdir(sys.argv[2]):
        pathData = sys.argv[2]
        aggregate_and_clean(pathData, verbose=True, extra=True)
        y_gen(pathData)
        impute(pathData)

    elif len(sys.argv) == 1:
        pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\canbind_data\\'
        aggregate_and_clean(pathData, verbose=False)
        y_gen(pathData)
        impute(pathData)
        convert_canbind_to_overlapping(pathData)
    else:
        print("Enter valid arguments\n"
              "\t options: -v for verbose, -v+ for super verbose\n"
              "\t path: the path to a real directory\n")