%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from setting import *
from sklearn.preprocessing import LabelEncoder, OneHotEncode

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

# image_bbox_df = {
# patientId
# x
# y
# width
# height
# Target
# class
# boxes
# path
# PatientAge
# BodyPartExamined
# ViewPosition
# PatientSex
# }
image_bbox_df = pd.read_csv(os.path.join(DS_dir, ''))
image_bbox_df['path'] = image_bbox_df['path'].map(lambda x: x.replace('input', ''))
print(image_bbox_df.shape[0], 'images')
image_bbox_df.sample(3)

class_enc = LabelEncoder()
image_bbox_df['class_idx'] = class_enc.fit_transform(image_bbox_df['class'])
oh_enc = OneHotEncode(sparse=False)
image_bbox_df['class_vec'] = oh_enc.fit_transform(
    image_bbox_df['class_idx'].values.reshape(-1,1)).tolist()
image_bbox_df.sample(3)