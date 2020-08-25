import argparse
import cv2
import glob
import os, os.path as osp

from tqdm import tqdm
from scipy.ndimage.interpolation import zoom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../../data/jpeg/train/')
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--size', type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    images = glob.glob(osp.join(args.data_dir, '*'))
    if not osp.exists(args.save_dir): os.makedirs(args.save_dir)

    for i in tqdm(images, total=len(images)):
        img = cv2.imread(i)
        # Resize shortest side to size
        s = min(img.shape[:2])
        factor = args.size/s
        img = zoom(img, [factor, factor, 1], order=1, prefilter=False)
        filename = osp.basename(i)
        filename = osp.join(args.save_dir, filename.replace('dcm','png'))
        cv2.imwrite(filename, img)


if __name__ == '__main__':
    main()

