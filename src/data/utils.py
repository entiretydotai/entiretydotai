import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List

# logger = logging.getLogger("entiretydotai")

def sel_column_label(path: Path, col_to_tag: List = ['text','label']):
    """[summary]
    
    Args:
        path (Path): [description]
        col_to_tag (List, optional): [description]. Defaults to ['text','label'].
    
    Returns:
        [type]: [description]
    """    
    logging.debug(f'Loading file of size: {path.stat().st_size} bytes')
    df = pd.read_csv(path)
    columns = dict(keys = list(df.columns))
    for col in col_to_tag:
        columns[col] = True
    return df, dict(filter(lambda elem: elem[1] == True,columns.items()))
    #print(dict((k, columns[k]) for k in columns if columns[k] == True))

def train_val_test_split(df,val_test_size: List = [0.1,0.1]):
    """[summary]
    
    Args:
        df ([type]): [description]
        val_test_size (List, optional): [description]. Defaults to [0.1,0.1].
    
    Returns:
        [type]: [description]
    """    
    names = ["valid","test"]
    size  = dict()
    for i, frac in enumerate(val_test_size):
        size[names[i]] = frac
    np.random.seed(123)
    ## Use decorator for sel_column_label
    train, test = train_test_split(df, stratify = df.target, test_size = size['test'] )
    train,valid = train_test_split(train, stratify = train.target, test_size = size['valid'] )
    logging.debug(f'NUmber of rows in the dataset: {df.shape[0]}')
    logging.debug(f'NUmber of rows in train dataset: {train.shape[0]}')
    logging.debug(f'NUmber of rows in valid dataset: {valid.shape[0]}')
    logging.debug(f'NUmber of rows in test dataset: {test.shape[0]}')
    return train , valid, test


if __name__== "__main__":
    sel_column_label("path_to_csv_file")