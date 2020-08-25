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
x['sex'] = pd.Categorical(x['sex']).codes
x['anatom_site_general_challenge'] = pd.Categorical(x['anatom_site_general_challenge']).codes

x.to_csv('../../data/train_cdeotte_meta.csv', index=False)
