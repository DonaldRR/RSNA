import glob, pylab, pandas as pd
import pydicom, numpy as np
from os.path import join
from util import *
from setting import *

df = pd.read_csv(join(DS_dir,'stage_1_train_labels.csv'))
# print('\nShow training label: Does not have pneumonia')
# print(df.iloc[0])
# print('\nShow training label: Have pneumonia')
# print(df.iloc[4])

patientId = df['patientId'][0]
dcm_file = join(DS_dir ,'stage_1_train_images/%s.dcm' % patientId)
dcm_data = pydicom.read_file(dcm_file)
# print(dcm_data)

im = dcm_data.pixel_array
# print(type(im))
# print(im.dtype)
# print(im.shape)

pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')

parsed = parse_data(df)

print(parsed['00436515-870c-4b36-a041-de91049b9ab4'])