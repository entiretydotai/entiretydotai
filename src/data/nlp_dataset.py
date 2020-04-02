import logging
import pathlib
import pandas as pd
import coloredlogs
from pathlib import Path
from typing import Union,Dict,List
from .utils import sel_column_label, train_val_test_split, save_csv
from flair.datasets import CSVClassificationCorpus

# logger = logging.getLogger("nlp_dataset")
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)
# coloredlogs.install(fmt='%(asctime)s %(name)s %(levelname)s %(message)s',level='DEBUG',logger = logger)

logger = logging.getLogger("entiretydotai")

class FlairDataset():
    """[summary]
    
    Raises:
        FileNotFoundError: [description]
    
    Returns:
        [type]: [description]
    """    
    def __init__(self,
        data_folder: Union[str, Path],
            column_name_map: Dict[int, str],
            train_file=None,
            test_file=None,
            dev_file=None,
            file_format = None,
            delimiter = None,
            encoding: str = "utf-8",
            train_data: pd.DataFrame = None,
            val_data: pd.DataFrame = None,
            test_data : pd.DataFrame = None):
        super().__init__()

        self.data_folder = data_folder
        self.column_name_map = column_name_map
        self.train_file = train_file
        self.test_file = test_file
        self.dev_file = dev_file
        self.file_format = file_format
        self.delimiter = delimiter
    
        if self.file_format == '.csv':
            logger.debug(f'Loading data in Flair CSV ClassificationCorpus from path :{self.data_folder}')
            self.corpus = CSVClassificationCorpus(
                data_folder=self.data_folder,
                train_file=self.train_file,
                dev_file=self.dev_file,
                test_file=self.test_file,
                column_name_map=self.column_name_map,
                delimiter = self.delimiter)
            logger.debug(f'Number of Sentences loaded[Train]:{self.corpus.train.total_sentence_count}')
            logger.debug(f'Type of tokenizer:{self.corpus.train.tokenizer.__name__}')
            logger.debug(f'Sample sentence and Label from [Train]:{self.corpus.train.__getitem__(1)}\n')
            logger.debug(f'Number of Sentences loaded[Valid]:{self.corpus.dev.total_sentence_count}')
            logger.debug(f'Type of tokenizer:{self.corpus.dev.tokenizer.__name__}')
            logger.debug(f'Sample sentence and Label from [Train]:{self.corpus.dev.__getitem__(1)}\n')
            logger.debug(f'Number of Sentences loaded[Test]:{self.corpus.test.total_sentence_count}')
            logger.debug(f'Type of tokenizer:{self.corpus.test.tokenizer.__name__}')
            logger.debug(f'Sample sentence and Label from [Train]:{self.corpus.test.__getitem__(1)}\n')
            self.train_data = train_data
            self.valid_data = val_data
            self.test_data = test_data

    @classmethod
    def csv_classification(cls,
        data_folder = Union[str,Path],
        file_format: str = 'csv',
        filename: str = 'data',
        train_val_test_split_flag: str = True ,
        column_mapping: List = None, 
        val_split_size: List = [0.1,0.1]):
        
            p = Path(data_folder).resolve()
            if p.is_dir():
                logger.debug(f'Found directory : {p}')
                files  = list(p.rglob('*.'+file_format))
                logger.debug(f'Number of files found {len(files)}')
                if len(files) < 2:
                    logger.debug(f'Found 1 file : {files[0].name}')
                    train_val_test_split_flag = True
                    logger.debug("Setting train_val_test_split_flag to True")
                    
                if train_val_test_split_flag:
                    if files[0].stem.lower() == filename:
                        train_file = files[0].name
                        flair_mapping = ['text','label']
                        df, column_name_map = sel_column_label(files[0],column_mapping,flair_mapping)
                        logger.debug(f'[column_name_map] {column_name_map}')
                        train, valid, test = train_val_test_split(df, val_split_size)
                        path_to_save = Path(p.parent.parent/'interim')
                        save_csv(train, path_to_save, 'train')
                        save_csv(valid, path_to_save, 'valid')
                        save_csv(test, path_to_save, 'test')
                        return FlairDataset(data_folder = path_to_save,
                                            column_name_map = column_name_map ,
                                            train_file= 'train.csv',
                                            test_file = 'test.csv',
                                            dev_file = 'valid.csv',
                                            file_format = '.csv',
                                            delimiter = ",",
                                            train_data = train,
                                            val_data = valid,
                                            test_data = test)                     
                    else:
                        raise FileNotFoundError

                else:
                    raise NotImplementedError




            else:
                pass



if __name__== "__main__":
        
    FlairDataset.csv_classification(
        data_folder='path_to_dir',column_mapping=['text','label'])

