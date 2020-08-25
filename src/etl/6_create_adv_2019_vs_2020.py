import pandas as pd
import os.path as osp

from sklearn.model_selection import GroupKFold


x = pd.read_csv('../../data/test.csv')
y = pd.read_csv('../../data/isic2019/ISIC_2019_Training_GroundTruth.csv')


x['image'] = [osp.join('data/jpeg/test/', _) for _ in x['image_name']]
y['image'] = [osp.join('data/isic2019/ISIC_2019_Training_Input/', _) for _ in y['image']]

x['label'] = 1
y['label'] = 0

y['patient_id'] = y['image']

df = pd.concat([x,y]).reset_index(drop=True)
group_kfold = GroupKFold(n_splits=5)
df['outer'] = -1

fold = 0
for train, test in group_kfold.split(df['image'], df['label'], df['patient_id']):
    df.loc[test, 'outer'] = fold
    fold += 1


for fold in range(5):
    train = df[df['outer'] == fold]
    test  = df[df['outer'] != fold]
    overlap = list(set(train['patient_id']) & set(test['patient_id']))
    assert len(overlap) == 0


df.to_csv('../../data/train_adv_2019_2020.csv', index=False)