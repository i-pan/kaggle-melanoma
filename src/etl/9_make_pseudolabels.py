import pandas as pd

df = pd.read_csv('../../data/test_with_meta.csv')

# x = pd.read_csv('../../submissions/bee008_5fold.csv')
# y = pd.read_csv('../../submissions/bee017_5fold.csv')
# x['target'] = (x['target'] + y['target']) / 2.

x = pd.read_csv('../../submissions/bee108_5fold.csv')
y = pd.read_csv('../../submissions/bee117_5fold.csv')
z = pd.read_csv('../../submissions/bee208_5fold.csv')
x['target'] = (x['target'] + y['target'] + z['target']) / 3.
x.columns = ['image_name', 'label']
df = df.merge(x, on='image_name')
colnames = list(df.columns)
colnames[0] = 'image'
df.columns = colnames

upsampled = pd.concat([df.sort_values('label', ascending=False).head(n=275)]*3)
df = pd.concat([upsampled, df[~df.image.isin(upsampled.image)]])
df['isic'] = 2021
df['outer'] = 888

train = pd.read_csv('../../data/combined_train_cdeotte_meta_no2019.csv')
pseudo = pd.concat([df, train])
pseudo.to_csv('../../data/combined_pseudolabel_no2019.csv', index=False)