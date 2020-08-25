import pandas as pd
import numpy as np
import pickle
import glob

from scipy.stats import rankdata as rd
from sklearn.metrics import roc_auc_score


def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def load_and_merge(predictions):
    if not isinstance(predictions, list): predictions = list(predictions)
    preds = []
    for i,p in enumerate(predictions):
        p = load_pickle(p)
        vals = p['y_pred']
        if vals.ndim > 1:
            if vals.shape[-1] == 2:
                vals = vals[:,1]
            else:
                vals = vals[:,0]
        pdf = pd.DataFrame({'image_name': p['image_id'], f'y_pred{i}': vals})
        preds += [pdf]
    maindf = preds[0]
    for i in range(1, len(preds)):
        maindf = maindf.merge(preds[i], on='image_name')
    train = pd.read_csv('../data/train.csv') 
    maindf = maindf.merge(train, on='image_name')
    return maindf


df = load_and_merge([f'../cv-predictions/fold0/skp00{i}.pkl' for i in [0,3,7,8,9]])

for c in df.columns:
    if 'y_pred' in c:
        print(f'AUC={roc_auc_score(df["target"], df[c]):.4f}')


roc_auc_score(df['target'], rd(df['y_pred0'])+rd(df['y_pred1'])+1.5*rd(df['y_pred2'])+0.5*rd(df['y_pred3'])+rd(df['y_pred4']))


###


def make_df(pickled):
    df = pd.DataFrame({'image': pickled['image_id'], 'y_pred': pickled['y_pred'][:,1], 'y_true': pickled['y_true']})
    #df['y_pred'] = rd(df['y_pred'])
    return df

preds = pd.concat([make_df(load_pickle(_)) for _ in glob.glob('../cv-predictions/fold*/skp007.pkl')])

malign = preds[preds['y_true'] == 1]
benign = preds[preds['y_true'] == 0]

benign = benign.sort_values('y_pred', ascending=False).iloc[:4500]
new_preds = pd.concat([benign, malign])
roc_auc_score(new_preds['y_true'], new_preds['y_pred'])

train = pd.read_csv('../data/train_with_stratified_splits.csv')
new_preds = new_preds[['image']].merge(train, on='image')
print(new_preds.shape)

new_preds.to_csv('../data/hard_examples_with_stratified_splits.csv')






