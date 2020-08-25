import pickle, pandas as pd
import numpy as np
import glob
import os.path as osp
import re

from scipy.stats import rankdata as rd


def identity(x): return x

def load_pickle(fp):
    with open(fp, 'rb') as f: return pickle.load(f)

def get_df(fp):
    pp = rd if RANKDATA else identity
    x = load_pickle(fp)
    if np.asarray(x['label']).ndim == 4:
        if np.asarray(x['label']).shape[-1] == 2:
            x['label'] = np.asarray(x['label'])[:,:,0,1]
        elif np.asarray(x['label']).shape[-1] == 3:
            x['label'] = np.asarray(x['label'])[:,:,0,2]
    else:
        x['label'] = np.asarray(x['label'])[:,:,0]
    x['label'] = np.asarray([pp(x['label'][:,i]) for i in range(x['label'].shape[1])])
    x['label'] = np.mean(x['label'], axis=0)
    df = pd.DataFrame({
        'image_name': [_.split('.')[0] for _ in x['image_id']],
        'target': x['label']
    })
    return df

RANKDATA = False

files = [
    '../../lb-predictions/bee508_5fold.pkl',
    '../../lb-predictions/bee517_5fold.pkl',
    '../../lb-predictions/bee608_5fold.pkl'
]

df = pd.concat([get_df(_).target for _ in files], axis=1)
df.columns = [_.split('/')[-1].replace('_5fold.pkl', '') for _ in df_list]
df.to_csv('final_submission.csv', index=False)


