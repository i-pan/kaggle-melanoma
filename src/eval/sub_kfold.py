import pickle, pandas as pd
import numpy as np
import glob
import os.path as osp
import re

from scipy.stats import rankdata as rd


def identity(x): return x

def load_pickle(fp):
    with open(fp, 'rb') as f: return pickle.load(f)

def get_df(fp, savedir='../submissions/'):
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
    df.to_csv(osp.join(savedir, osp.basename(fp).replace('pkl','csv')), index=False)
    return df

RANKDATA = False

df_list = glob.glob('../lb-predictions/bee*pkl')
df_list = [_ for _ in df_list if re.search(r'08|17', _)]

df = pd.concat([get_df(_).target for _ in df_list], axis=1)
df.columns = [_.split('/')[-1].replace('_5fold.pkl', '') for _ in df_list]
df.corr()

df.to_csv('../submissions/bee015_5fold.csv', index=False)


###

x = load_pickle('../lb-predictions/skp024-5fold.pkl')
x['label'] = np.mean(np.asarray([rd(_[:,1]) for _ in x['label']]), axis=0)
df = pd.DataFrame({
    'image_name': [_.split('.')[0] for _ in x['image_id']],
    'target': x['label']
})

df.to_csv('../submissions/skp024-5fold.csv', index=False)