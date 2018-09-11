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
print('\n##Sample dcm data')
print(dcm_data)

im = dcm_data.pixel_array
print('\n## Type of image')
print(type(im))
print('\n## Data type of image')
print(im.dtype)
print('\n## Shape of image')
print(im.shape)

pylab.figure()
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.show()
pylab.axis('off')

parsed = parse_data(df)
print('\n## Sample parsed item')
print(parsed['00436515-870c-4b36-a041-de91049b9ab4'])

draw(parsed['00436515-870c-4b36-a041-de91049b9ab4'])
df_detailed = pd.read_csv(join(DS_dir, 'stage_1_detailed_class_info.csv'))

patientId = df_detailed['patientId'][0]
draw(parsed[patientId])

summary = {}
for n, row in df_detailed.iterrows():
    if row['class'] not in summary:
        summary[row['class']] = 0
    summary[row['class']] += 1
print('\n##Summary')
print(summary)