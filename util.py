import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob, pylab, pandas as pd
import pydicom, numpy as np
from os.path import join
from setting import *
from os.path import join

def parse_data(df):

    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():

        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': join(DS_dir, 'stage_1_train_images/%s.dcm' % pid),
                'label': row['Target'],
                'boxes': []
            }

        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


def draw(data):
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array
    im = np.stack([im]*3, axis=2)

    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')
    pylab.show()

def overlay_box(im, box, rgb, stroke=1):

    box = [int(b) for b in box]

    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

