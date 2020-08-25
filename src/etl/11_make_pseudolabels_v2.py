import pandas as pd

df = pd.read_csv('../../data/test_with_meta.csv')

x = pd.read_csv('../../submissions/bee108_5fold.csv')
y = pd.read_csv('../../submissions/bee117_5fold.csv')
z = pd.read_csv('../../submissions/bee208_5fold.csv')
x['target'] = (x['target'] + y['target'] + z['target']) / 3.
x.columns = ['image_name', 'label']
df = df.merge(x, on='image_name')
colnames = list(df.columns)
colnames[0] = 'image'
df.columns = colnames

df['isic'] = 2021
df['outer'] = 888

train = pd.read_csv('../../data/train_cdeotte_meta.csv')
train['isic'] = 2020
pseudo = pd.concat([df, train])
pseudo.to_csv('../../data/combined_pseudolabel_v2.csv', index=False)