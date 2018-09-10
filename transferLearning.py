%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from setting import *

# Parameters will probably want to do some hyperparameter optimization later
BASE_MODEL = 'VGG16'
IMG_SIZE = (384,384)
BATCH_SIZE = 24
DENSE_COUND = 128
DROPOUT = 0.25
LEARN_RATE = 1e-4
TRAIN_SAMPLES = 8000
TEST_SAMPLES = 800
USE_ATTN = False

image_bbox_df = pd.read_csv(os.path.join(DS_dir, ''))