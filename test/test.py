import sys
sys.path.append("..")

from src import FlairDataset
path = "path_to_dir"
FlairDataset.csv_classification(data_folder=path,filename='data',column_mapping=['text','label'])