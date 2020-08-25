import pandas as pd
import numpy as np
import os.path as osp


def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)


BORDERS = [30, 35, 40, 45, 50, 55, 60, 65, 70]

x = pd.read_csv('../../data/train_cdeotte.csv')
x = x[x['tfrecord'] != -1]
x['outer'] = x['tfrecord'] // 3
x['image'] = x['image_name']
x['age_approx'] = x['age_approx'].fillna(np.median(x['age_approx']))
x['sex'] = x['sex'].fillna('male')
x['anatom_site_general_challenge'] = x['anatom_site_general_challenge'].fillna('torso')
x['label'] = x['target']
x['age_cat'] = [to_bins(_, borders=BORDERS) for _ in x['age_approx']]

y = pd.read_csv('../../data/isic2019/ISIC_2019_Training_GroundTruth.csv')
ymeta = pd.read_csv('../../data/isic2019/ISIC_2019_Training_Metadata.csv')
y = y.merge(ymeta, on='image')
y.loc[y['anatom_site_general'].isin(['anterior torso', 'posterior torso', 'lateral torso']), 'anatom_site_general'] = 'torso'
y['anatom_site_general_challenge'] = y['anatom_site_general']
y['age_approx'] = y['age_approx'].fillna(np.median(y['age_approx']))
y['sex'] = y['sex'].fillna('male')
y['anatom_site_general_challenge'] = y['anatom_site_general_challenge'].fillna('torso')

y['age_cat'] = [to_bins(_, borders=BORDERS) for _ in y['age_approx']]
y['outer'] = 888

y['label'] = y['MEL']

y['isic'] = 2019
x['isic'] = 2020


# Combine everything
# Only add melanomas from ISIC2019
upsample = int(np.ceil(y['label'].sum()/x['label'].sum()))-1
df = pd.concat([x,y]+[x[x['label'] == 1]]*upsample)
df['sex'] = pd.Categorical(df['sex']).codes
df['anatom_site_general_challenge'] = pd.Categorical(df['anatom_site_general_challenge']).codes

df.to_csv('../../data/combined_train_cdeotte_meta.csv', index=False)
