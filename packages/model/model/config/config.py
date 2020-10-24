import pathlib
import os

import pandas as pd
import model

PACKAGE_ROOT = pathlib.Path(model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'data'
DATA_FILES = 'subject*'


# Data Parameters
H = 120 # full size is 360
W = 160 # full size is 480

# Data Loader Parameters
BATCH_SIZE = 1

# model
MODEL_NAME = 'mymodel3.pt'
PIPELINE_NAME = 'pipe'
ENCODER_NAME = 'encoder'

# CNN Parameters
FEATURE_DIM = 64
EPOCHS = 20

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()
    
MODEL_FILE_NAME = f'{MODEL_NAME}_{_version}.pt'
#MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME)

PIPELINE_FILE_NAME = f'{PIPELINE_NAME}_{_version}.pkl'
#PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_FILE_NAME)

ENCODER_FILE_NAME = f'{ENCODER_NAME}_{_version}.pkl'
ENCODER_PATH = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME)