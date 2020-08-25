import pandas as pd
import glob
import re

from PIL import Image
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else MultilabelStratifiedKFold
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


# Get images
imgfiles = glob.glob('../../data/jpeg/train/*.jpg')
shapes = [Image.open(i).size for i in imgfiles]
w, h = [s[0] for s in shapes], [s[1] for s in shapes]
shape_df = pd.DataFrame({'h': h, 'w': w, 'image_name': [i.split('/')[-1].split('.')[0] for i in imgfiles]})

df = pd.read_csv('../../data/train.csv') 
df = df.merge(shape_df, on='image_name')
df['h_group'] = pd.qcut(df['h'], q=4, labels=[0,1,2,3], duplicates='drop')
df['w_group'] = pd.qcut(df['w'], q=2, labels=[0,1], duplicates='drop')

# Sort each pt DF by malignant so pt goes on top
pt_dfs = [_[1] for _ in list(df.groupby('patient_id'))]
pt_dfs = [_.sort_values(['benign_malignant', 'diagnosis'], ascending=[False, True]) for _ in pt_dfs]
# Take the first row of each patient DF for CV splitting
cv_df = pd.concat([_.iloc[:1] for _ in pt_dfs])

cv_df = cv_df.drop_duplicates().reset_index(drop=True)
stratified_cols = ['benign_malignant', 'diagnosis', 'h_group', 'w_group']
cv_df = create_double_cv(cv_df, 'patient_id', 5, 5, stratified=stratified_cols)
cv_df = cv_df[['patient_id'] + [col for col in cv_df.columns if re.search(r'outer|inner', col)]]

df = df.merge(cv_df, on='patient_id')
df['label'] = df['benign_malignant']
df['label'] = [1 if _ == 'malignant' else 0 for _ in df['label']]
df['image'] = df['image_name']
df.to_csv('../../data/train_with_stratified_splits.csv', index=False)


