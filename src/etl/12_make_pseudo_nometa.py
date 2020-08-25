import pandas as pd
import numpy as np
import pickle

def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

df = pd.read_csv('../../data/test_with_meta.csv')

x = load_pickle('../../lb-predictions/bee508_5fold.pkl')
for i in dfs:
    assert np.mean(i.image_name.apply(lambda s: s+'.jpg').values == np.asarray(x['image_id'])) == 1

x = np.asarray(x['label'])
x = np.mean(x[:,:,0], axis=1)

df['label'] = [f'{i},{j},{k}' for i,j,k in x]
df['label_mel'] = x[:,-1]
df['image'] = df.image_name
upsampled = pd.concat([df.sort_values('label_mel', ascending=False).head(n=275)]*7)
df = pd.concat([upsampled, df[~df.image.isin(upsampled.image)]])
df['isic'] = 2021
df['outer'] = 888

train = pd.read_csv('../../data/combined_train_cdeotte_nevi_meta.csv')
pseudo = pd.concat([df, train])
pseudo.to_csv('../../data/combined_pseudolabel_3class.csv', index=False)