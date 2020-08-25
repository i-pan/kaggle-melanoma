import argparse
import numpy as np
import glob
import os, os.path as osp

from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    return parser.parse_args()


def get_metric_from_checkpoint(fp):
    return float(fp.split('/')[-1].split('-')[-1].replace('.PTH', ''))


def main():
    args = parse_args()
    folds = np.sort(glob.glob(osp.join(args.dir, '*')))
    folds = list(folds)

    folds_dict = defaultdict(list)
    for fo in folds:
        checkpoints = glob.glob(osp.join(fo, '*.PTH'))
        if len(checkpoints) == 0:
            continue

        for ckpt in checkpoints:
            value = get_metric_from_checkpoint(ckpt)
            folds_dict[fo].append(value)

    for fo in np.sort([*folds_dict]):
        print(f'{fo.split("/")[-1].upper()} : {np.max(folds_dict[fo]):.4f}')

    print('=====')
    print(f'CVAVG : {np.mean([np.max(v) for v in folds_dict.values()]):.4f}')


if __name__ == '__main__':
    main()
