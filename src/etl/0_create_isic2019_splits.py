import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold


def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else StratifiedKFold
    outer_spl = splitter(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_counter = 0
    for outer_train, outer_test in outer_spl.split(df) if stratified is None else outer_spl.split(df, df[stratified]):
        df.loc[outer_test, 'outer'] = outer_counter
        inner_spl = splitter(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_counter = 0
        df['inner{}'.format(outer_counter)] = 888
        inner_df = df[df['outer'] != outer_counter].reset_index(drop=True)
        # Determine which IDs should be assigned to inner train
        for inner_train, inner_valid in inner_spl.split(inner_df) if stratified is None else inner_spl.split(inner_df, inner_df[stratified]):
            inner_train_ids = inner_df.loc[inner_valid, id_col]
            df.loc[df[id_col].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
            inner_counter += 1
        outer_counter += 1
    return df


df = pd.read_csv('../../data/isic2019/ISIC_2019_Training_GroundTruth.csv')

dxs = list(df.columns)
dxs.remove('image') ; dxs.remove('UNK')
dxs = df[dxs].values

df['label'] = dxs.argmax(axis=1)
# 0- Melanoma
# 1- Nevi
# 2- Basal Cell Carcinoma
# 3- Actinic Keratosis 
# 4- Benign Keratosis
# 5- Dermatofibroma
# 6- Vascular
# 7- Squamous Cell Carcinoma

cv_df = df[['image']].drop_duplicates().reset_index(drop=True)
cv_df = create_double_cv(cv_df, 'image', 10, 10)
df = df.merge(cv_df, on='image')
df.to_csv('../../data/isic2019/train_with_splits.csv', index=False)


