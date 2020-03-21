import logging
import pathlib
from pathlib import Path
from typing import Union,Dict,List
from .utils import sel_column_label, train_val_test_split

logging.basicConfig( level=logging.DEBUG,
     format='%(asctime)s:%(levelname)s:%(message)s')
#logger = logging.getLogger("entiretydotai")
#logging.getLogger("entiretydotai")


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
            encoding: str = "utf-8"):
        super().__init__()

        self.data_folder = data_folder
        self.column_name_map = column_name_map
        self.train_file = train_file
        self.test_file = test_file
        self.dev_file = dev_file
    
    
    @classmethod
    def load_for_classification(cls,
        data_folder = Union[str,Path],
        file_format: str = 'csv',
        filename: str = 'data',
        train_val_test_split_flag: str = True ,
        val_split_size: List = [0.1,0.1]):
        
            p = Path(data_folder).resolve()
            if p.is_dir():
                logging.debug(f'Found directory : {p.absolute()}')
                files  = list(p.rglob('*.'+file_format))
                logging.debug(f'Number of files found {len(files)}')
                if len(files) < 2:
                    logging.debug(f'Found 1 file : {files[0].name}')
                    train_val_test_split_flag = True
                    logging.debug("Setting train_val_test_split_flag to True")
                    
                if train_val_test_split_flag:
                    if files[0].stem.lower() == filename:
                        train_file = files[0].name
                        df, column_name_map = sel_column_label(files[0])
                        
                        ## Save original file(df) in  data/external folder
                        
                        train, valid, test = train_val_test_split(df, val_split_size)
                        
                        ## Save into data/interim folder
                        #path_to_save = p.cwd() / 'datasets'
                        #path_to_save.mkdir(parents=True, exist_ok=True)
                        #filepath = path_to_save /'valid.csv'
                        #valid.to_csv(filepath,index=False)
                        return FlairDataset(data_folder = p, column_name_map = column_name_map ,
                                                        train_file=train_file,
                                                        test_file=None,
                                                        dev_file=None )
                    else:
                        raise FileNotFoundError

                else:
                    pass




            else:
                pass



if __name__== "__main__":
        
    FlairDataset.load_for_classification(
        data_folder='path_to_dir')

