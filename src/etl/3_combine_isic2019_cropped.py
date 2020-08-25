import pandas as pd
import os.path as osp


x = pd.read_csv('../../data/train_with_stratified_splits.csv')
y = pd.read_csv('../../data/isic2019/train_with_splits.csv')
y['image'] = [osp.join('data/isic2019/ISIC_2019_Training_Input/', _) for _ in y['image']]
x['image'] = [osp.join('data/cropped/train/', _) for _ in x['image']]
y['label'] = y['MEL']
y = y[['image', 'label']]
y['isic'] = 2019
x['isic'] = 2020

cols = [c for c in x.columns if 'outer' in c or 'inner' in c]
for c in cols: y[c] = 888

# Combine everything
df = pd.concat([x,y])
df.to_csv('../../data/complete_cropped_combined_train_with_splits.csv', index=False)

# Multiply melanoma from ISIC20 by 8
df = pd.concat([x,y]+[x[x['target'] == 1]]*7)
df.to_csv('../../data/upsampled_cropped_combined_train_with_splits.csv', index=False)

# Only add melanomas from ISIC2019
df = pd.concat([x,y[y['label'] == 1]]+[x[x['target'] == 1]]*7)
df.to_csv('../../data/upsampled_cropped_combined_train_with_splits.csv', index=False)
