import numpy as np
from glob import glob

import tensorflow as tf

import logging

_logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str ) -> pd.DataFrame:
    
    _data = glob('../datasets/subject*')
    np.random.shuffle(files)
    
    return _data

def prepare_data(df, train_data):



    return X,y

if __name__ == '__main__':
    
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    d = data.copy()
    
    X, y = prepare_data(data,True)
    XX,yy = prepare_data(d,False)
    
    print( sum(X.index() != XX.index()) )
    
