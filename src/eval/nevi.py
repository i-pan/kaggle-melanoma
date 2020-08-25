import pickle
import pandas as pd
import numpy as np
import torch


def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


preds = load_pickle('../../train-predictions/isic2019-mk001.pkl')
p = preds['label']
p = torch.softmax(torch.from_numpy(preds['label'][0]), dim=1).numpy()

mel_prob = p[:,0]
nev_prob = p[:,1]

df = pd.DataFrame({
        'image_id': preds['image_id'],
        'mel_prob': mel_prob,
        'nev_prob': nev_prob
    })
df['image_name'] = [_.replace('.jpg','') for _ in df.image_id]

train_df = pd.read_csv('../../data/train.csv')
train_df = train_df.merge(df, on='image_name')

known_nevi = train_df[train_df['diagnosis'] == 'nevus']
np.percentile(known_nevi.nev_prob, [0,2.5,5,10,25,50,75,90,95,100])
unknown = train_df[train_df['diagnosis'] == 'unknown']
np.percentile(unknown.nev_prob, [0,2.5,5,10,25,50,75,90,95,100])

cv_df = pd.read_csv('../../data/train_cdeotte.csv')
cv_df = cv_df.merge(df, on='image_name')
cv_df.loc[(cv_df.diagnosis == 'unknown') & (cv_df.nev_prob >= np.percentile(known_nevi.nev_prob, 5)),'diagnosis'] = 'nevus'
label_map = {'nevus': 1, 'melanoma': 2}
cv_df['label'] = [label_map[_] if _ in [*label_map] else 0 for _ in cv_df.diagnosis]
cv_df.to_csv('../../data/train_cdeotte_nevi.csv', index=False)