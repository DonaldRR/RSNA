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