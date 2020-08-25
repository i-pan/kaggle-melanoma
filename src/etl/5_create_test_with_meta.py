import pandas as pd
import numpy as np


def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)


BORDERS = [30, 35, 40, 45, 50, 55, 60, 65, 70]

df = pd.read_csv('../../data/test.csv')
df['age_approx'] = df['age_approx'].fillna(np.median(df['age_approx']))
df['sex'] = pd.Categorical(df['sex'].fillna('male')).codes
df['anatom_site_general_challenge'] = pd.Categorical(df['anatom_site_general_challenge'].fillna('torso')).codes

df['age_cat'] = [to_bins(_, BORDERS) for _ in df['age_approx']]
df.to_csv('../../data/test_with_meta.csv', index=False)