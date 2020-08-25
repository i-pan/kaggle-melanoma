import pickle, pandas as pd
import numpy as np

from scipy.stats import rankdata as rd


def load_pickle(fp):
    with open(fp, 'rb') as f: return pickle.load(f)



x = load_pickle('../lb-predictions/skp017.pkl')
x['label'] = np.mean(np.asarray([rd(_[:,1]) for _ in x['label']]), axis=0)
df = pd.DataFrame({
    'image_name': [_.split('.')[0] for _ in x['image_id']],
    'target': x['label']
})

df.to_csv('../submissions/skp017.csv', index=False)

###

x = load_pickle('../lb-predictions/skp024-5fold.pkl')
x['label'] = np.mean(np.asarray([rd(_[:,1]) for _ in x['label']]), axis=0)
df = pd.DataFrame({
    'image_name': [_.split('.')[0] for _ in x['image_id']],
    'target': x['label']
})

df.to_csv('../submissions/skp024-5fold.csv', index=False)