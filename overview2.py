# %matplotlib inline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os
from matplotlib.patches import Rectangle
from util import *
from setting import *

det_class_path = os.path.join(DS_dir, 'stage_1_detailed_class_info.csv')
bbox_path = os.path.join(DS_dir, 'stage_1_train_labels.csv')
dicom_dir = os.path.join(DS_dir, 'stage_1_train_images/')

# detailed_class_info:{
#   index
#   patientId
#   class(labels)
# }

print('\n--Overview of det_class_info.csv--\n')
det_class_df = pd.read_csv(det_class_path)
print(det_class_df.shape[0], 'class infos loaded')
print(det_class_df['patientId'].value_counts().shape[0], 'patientId cases')
det_class_df.groupby('class').size().plot.bar()
print(det_class_df.sample(3))

# train_labels:{
#   index
#   patientId
#   x
#   y
#   width
#   height
#   Target
# }

print('\n--Overview of train_labels.csv--\n')
bbox_df = pd.read_csv(bbox_path)
print(bbox_df.shape[0],'boxex loaded')
print(bbox_df['patientId'].value_counts().shape[0], 'patient cases')
print(bbox_df.sample(3))

comb_bbox_df = pd.merge(bbox_df, det_class_df, how='inner', on='patientId')
print('\n', comb_bbox_df.shape[0], ' combined cases')
print(comb_bbox_df.columns)

comb_bbox_df = pd.concat([bbox_df, det_class_df.drop('patientId', 1)], 1)
print('\n', comb_bbox_df.shape[0], 'combined cases')
print(comb_bbox_df.sample(3))

box_df = comb_bbox_df.groupby('patientId').size().reset_index(name='boxes')
comb_bbox_df = pd.merge(comb_bbox_df, box_df, on='patientId')

box_df.groupby('boxes').size().reset_index(name='patients')
print(box_df.sample(3))



