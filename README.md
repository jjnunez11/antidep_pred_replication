# Replication of Machine Learning Methods to Predict Treatment Outcome with Antidepressant Medications in Patients with Major Depressive Disorder from STAR*D and CAN-BIND-1
## Accepted for publication in PLOS One May 27, 2021

README by John-Jose Nunez, jjnunez11@gmail.com

Thank you for your interest in our project! Our aim is to make it as reproducible as possible, please do not hesitate to reach out. 

## Data Availability
The raw clinical data from STAR*D is found [in our NIMH NDA project collection](http://dx.doi.org/10.15154/1503299)

You can download the data freshly from NDA via the measures tab, or can download a zip of the exact data we used going to the "Data Analysis" tab
and downloading the file [Version Of Raw Data Used with One Modification] https://nda.nih.gov/study.html?tab=result&id=640. 

The raw clinical data from CAN-BIND-1 can obtained from [Brian-CODE](https://braininstitute.ca/research-data-sharing/brain-code)

The processed STAR*D datasets used directly for machine learning are obtainable through our NIMH NDA project, in the data analysis tab,
and is named [Processed STARD Datasets used for ML](https://nda.nih.gov/study.html?tab=result&id=640)

The processed CAN-BIND datasets used directly for machine learning are obtainable through [Brian-CODE](https://braininstitute.ca/research-data-sharing/brain-code).
We may update the NDA collection with them when possible. 

The results used in our paper are located within zips for each table in the [results folder](./results)

The exact models objects used to obtain our results, compressed via Python's pickle protocol, are obtain in this [OneDrive folder](https://onedrive.live.com/embed?cid=3270DE108C079AD9&resid=3270DE108C079AD9%2113381&authkey=AGH3p3NQb5bCa9w) due to space constraints of Github. 

## Data Processing

Please find our data cleaning and processing code in [data-cleaning](./code/data-cleaning/)

The raw STAR*D and CAN-BIND data should be placed in separate folders within [data](./data/)

### Run order
1. Generate STAR*D dataset with [stard_preprocessor.py](./code/data-cleaning/stard_preprocessor.py)
2. Generate initial CAN-BIND dataset with  [canbind_preprocessor.py](./code/data-cleaning/canbind_preprocessor.py)
3. Generate overlapping datasets with [generate_overlapping_features](./code/data-cleaning/generate_overlapping_features.py)

## Running Machine Learning Analysis
1. Update the path of your data and result directories in [run_globals](./code/run_globals.py)
2. If using the preprocessing scripts above, the final data matrices will be saved within ./processed_data/final_xy_data_matrices/
3. Place your .csv files with the X and y matrices you want to run (or ours as downloaded) into the data directory above
4. Run the machine learning with [run_results](./code/run_results.py) script. This script calls [run_result](./code/run_results.py), assuming the filename of the 
dataset csv's will be passed along, minus the '.csv' ending. 

### Code Versions
The project's code was last completely ran in October, 2020. The following versions were used:
Python 3.6.10
scikit-learn 0.23.1
pandas 1.0.5
numpy 1.17.0
scipy 1.5.0
