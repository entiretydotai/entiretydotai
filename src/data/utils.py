import numpy as np
import pandas as pd
import logging
import coloredlogs
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Union
from functools import wraps
from flair.data import Sentence
from flair.models import SequenceTagger
from .gen_utils import clean_flair_tags

# logger = logging.getLogger("utils")
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)
# coloredlogs.install(fmt='%(asctime)s %(name)s %(levelname)s %(message)s',level='DEBUG',logger = logger)
logger = logging.getLogger("entiretydotai")

def my_logger(orig_func):
    """[Used to log the passed arguments to a function ]
    
    Args:
        orig_func ([type]): [description]
    
    Returns:
        [type]: [description]
    """    

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logger.debug(
            f'[{orig_func.__name__}] : Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    """[Used to log time taken in running a process]
    
    Args:
        orig_func ([type]): [description]
    
    Returns:
        [type]: [description]
    """    
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        tik = time.time()
        result = orig_func(*args, **kwargs)
        tok = time.time() - tik
        logger.debug(f'[{orig_func.__name__}] ran in: {tok} sec')
        return result

    return wrapper


# def pandascsv_logger(func):
#     """[USed to log pandas dataframe statistics]
    
#     Args:
#         func ([type]): [description]
    
#     Returns:
#         [type]: [description]
#     """    
#     import pandas as pd
    
#     @wraps(orig_func)
#     def wrapper(*args, **kwargs):
#         logging.debug(
#             'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
#         return orig_func(*args, **kwargs)

#     return wrapper


def read_csv(path: Union[str, Path]):
    """[Reads a csv file into a pandas Dataframe]
    
    Args:
        path (Union[str, Path]): [description]
    
    Returns:
        [DataFrame]: [pandas Dataframe]
    """    
    logging.debug(f'Loading file of size: {path.stat().st_size} bytes')
    return pd.read_csv(path,sep=",")


@my_logger
@my_timer
def sel_column_label(path: Path, col_to_tag: List = ['text','label'], flair_mapping : List = ['text','label']):
    """[summary]
    
    Args:
        path (Path): [description]
        col_to_tag (List, optional): [description]. Defaults to ['text','label'].
    
    Returns:
        [type]: [description]
    """    
    df = read_csv(path)
    columns = list(df.columns)
    index = []
    for col in col_to_tag:
        index.extend([columns.index(col)])
    column_name_map = dict(zip(index,col_to_tag))
    return df , column_name_map
    # return df, dict(filter(lambda elem: elem[1] == True,columns.items()))
    # print(dict((k, columns[k]) for k in columns if columns[k] == True))
    #print(dict((k, columns[k]) for k in columns if columns[k] == True))

@my_timer
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
    train, test = train_test_split(df, stratify = df['label'], test_size = size['test'] )
    train,valid = train_test_split(train, stratify = train['label'], test_size = size['valid'] )
    logger.debug(f'NUmber of rows in the dataset: {df.shape[0]}')
    logger.debug(train.columns)
    logger.debug(f'NUmber of rows in train dataset: {train.shape[0]}')
    logger.debug(f'NUmber of rows in valid dataset: {valid.shape[0]}')
    logger.debug(f'NUmber of rows in test dataset: {test.shape[0]}')
    # train = drop_cols(train,['id'],1)
    # valid = drop_cols(valid,['id'],1)
    # test = drop_cols(test,['id'],1)
    return train , valid, test

@my_timer
def save_csv(df: pd.DataFrame, path: Path, filename = 'noname'):
    df.to_csv(str(path/filename)+".csv",sep=",",index=False)
    logger.debug(f'Successfully saved {filename} at : {path}')

@my_timer
def drop_cols(df: pd.DataFrame, to_drop_cols : List, axis : int = 1):
    df = df.drop(to_drop_cols,axis=1)
    logger.debug(f'Successfully dropped {to_drop_cols} from dataframe ')
    return df


def extract_tag(rows : List =  None, model: Union[str, Path] = 'ner-fast'):
    sentence = Sentence(rows)
    #logger.debug(f'Processing sentence: {rows}')
    #print("Processsing sentence",rows)
    model.predict(sentence)
    return sentence


@my_timer
def flair_tags(rows : List, model_name: Union[str, Path] = 'ner-fast',
            tag_type: str = 'ner',extract_tags : bool = False,
            return_score: float = False):
    
    model = SequenceTagger.load(model_name)
    logger.debug(f'MODEL: [{model_name}] is loaded ...')
    sentences = [extract_tag(i,model) for i in rows]
    if extract_tags:
        corpus_text = []
        corpus_cleaned_tag = []
        corpus_score = []
        for sentence in sentences:
            #text = []
            cleaned_tag = []
            score = []
            text = sentence.to_tokenized_string().split(" ")
            entity_tagged = sentence.get_spans(tag_type)
            tagged_text = [ent.text for ent in entity_tagged]
            tagged_label = [ent.tag for ent in entity_tagged]
            if return_score:
                tagged_score = [ent.score for ent in entity_tagged]
            for i in text:
                if i in tagged_text:
                    #corpus.append(i)
                    index = tagged_text.index(i)
                    cleaned_tag.append(tagged_label[index])
                    if return_score:
                        score.append(round(tagged_score[index],2))
                
                else:
                    #corpus.append(i)
                    cleaned_tag.append("NA")
                    if return_score:
                        score.append(np.NaN)

            corpus_text.append(text)
            corpus_cleaned_tag.append(cleaned_tag)
            corpus_score.append(score)
        if return_score:
            return [corpus_text,corpus_cleaned_tag,corpus_score]
        else:
            return [corpus_text,corpus_cleaned_tag]

    else:
        corpus_tags = [i.to_tagged_string() for i in sentences]
        return [corpus_tags]

@my_timer
def flair_tags_as_string(rows : List, model_name: Union[str, Path] = 'ner-fast',
            tag_type: str = 'ner',clean_tags: bool = True):
    
    model = SequenceTagger.load(model_name)
    logger.debug(f'MODEL: [{model_name}] is loaded ...')
    sentences = [extract_tag(i,model) for i in rows]
    tagged_sentences = [i.to_tagged_string() for i in sentences]
    corpus = []
    corpus_tags = []
    for sent in tagged_sentences:
        corpus.append([i for i in sent.split(" ") if not i.strip().startswith("<")])
        if clean_tags:
            corpus_tags.append([clean_flair_tags(i) for i in sent.split(" ") if i.strip().startswith("<")])
        else:
            corpus_tags.append([i for i in sent.split(" ") if  i.strip().startswith("<")])

    return [tagged_sentences,corpus,corpus_tags]


if __name__== "__main__":
    #sel_column_label("path_to_csv_file")
    from pathlib import Path
    df = sel_column_label(Path("path_to_file"),
    col_to_tag=['text','label'],flair_mapping= None)