import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
import numpy as np
import cv2
import re 
import glob
import os, os.path as osp

from PIL import Image
from torch.utils.data import Dataset, Sampler
from .utils import _isnone
from .crop_tta import crop_tta, resize_for_crop

import numpy as np


def square_crop(img, random=True):
    h, w = img.shape[:2]
    if h == w: return img
    s = min(h, w)
    short_side = 0 if h<w else 1
    xc, yc = h//2, w//2
    if random: 
        offset = np.abs(h-w)
        offset = np.random.randint(-offset, offset)
        if short_side:
            xc += offset//2
        else:
            yc += offset//2
    x1, y1 = xc-s//2, yc-s//2
    x1, y1 = max(0,x1), max(0,y1)
    img_crop = img[x1:x1+s, y1:y1+s]
    if img_crop.shape[0] != img_crop.shape[1]:
        print(f'Shape is {img_crop.shape}')
        return img
    return img_crop


def generate_crops(img, num_crops=10):
    h, w = img.shape[:2]
    if h == w: return [img]
    s = min(h, w)
    short_side = 0 if h<w else 1
    xc, yc = h//2, w//2
    offset = np.abs(h-w)
    offsets = np.unique(np.linspace(-offset+1, offset-1, num_crops).astype('int'))
    crops = []
    for off in offsets:
        if short_side:
            new_xc = xc-off//2
            x1, y1 = new_xc-s//2, yc-s//2
        else:
            new_yc = yc-off//2
            x1, y1 = xc-s//2, new_yc-s//2
        x1, y1 = max(0,x1), max(0,y1)
        crops += [img[x1:x1+s, y1:y1+s]]
        if crops[-1].shape[0] != crops[-1].shape[1]:
            print(f'Shape is {crops[-1].shape}')
            print(img.shape)
            print(x1, y1, s)
            crops[-1] = img
    return crops


class SkinDataset(Dataset):


    def __init__(self,
                 imgfiles,
                 labels,
                 meta=None,
                 square=False,
                 square_tta=None,
                 crop_tta=None,
                 pad=None,
                 resize=None,
                 transform=None,
                 crop=None,
                 preprocessor=None,
                 flip=False,
                 verbose=True,
                 test_mode=False,
                 jsd=False,
                 onehot=False):
        self.imgfiles = imgfiles
        self.labels = labels
        self.meta = meta
        self.square = square
        self.square_tta = square_tta
        self.crop_tta = crop_tta
        self.pad = pad 
        self.resize = resize
        self.transform = transform
        self.crop = crop 
        if self.crop: 
            self.crop_size = (self.crop.transforms[0].height, self.crop.transforms[0].width)
        self.preprocessor = preprocessor
        self.flip = flip
        self.verbose = verbose
        self.test_mode = test_mode
        self.jsd = jsd
        self.onehot = onehot


    def process_image(self, X, jsd=False):
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        if self.transform and not jsd: X = self.transform(image=X)['image']
        if self.crop and not self.test_mode: 
            X = resize_for_crop(X, crop_size=self.crop_size)
            X = self.crop(image=X)['image']
        if self.preprocessor: X = self.preprocessor.preprocess(X)
        return X.transpose(2, 0, 1)


    def get(self, i):
        try:
            X = cv2.imread(self.imgfiles[i])
            if _isnone(X):
                X = cv2.imread(self.imgfiles[i].replace('jpg','png'))

            if _isnone(X):
                return None 
                
            if not _isnone(self.meta):
                X = {'img': X}
                X.update(self.meta[i])

            return X
        except Exception as e:
            if self.verbose: print(e)
            return None


    @staticmethod
    def flip_array(X, mode):
        if mode == 0:
            X = X[:,::-1]
        elif mode == 1:
            X = X[:,:,::-1]
        elif mode == 2:
            X = X[:,::-1,::-1]
        elif mode == 3 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0,2,1)
        X = np.ascontiguousarray(X) 
        return X


    def __len__(self):
        return len(self.imgfiles)


    def __getitem__(self, i):
        X = self.get(i)
        while _isnone(X):
            if self.verbose: print('Failed to read {} !'.format(self.imgfiles[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        if self.test_mode and self.square_tta:
            if isinstance(X, dict):
                X['img'] = generate_crops(X['img'], num_crops=self.square_tta)
                X['img'] = np.asarray([self.process_image(_) for _ in X['img']])                
                for k,v in X.items():
                    if k == 'img': continue
                    X[k] = np.repeat(np.expand_dims(v, axis=0), X['img'].shape[0], axis=0)
            else:
                X = generate_crops(X, num_crops=self.square_tta)
                X = np.asarray([self.process_image(_) for _ in X])
        elif self.test_mode and self.crop_tta:
            if isinstance(X, dict):
                X['img'] = crop_tta(X['img'], crop_size=self.crop_size, num_crops=self.crop_tta)
                X['img'] = np.asarray([self.process_image(_) for _ in X['img']])                
                for k,v in X.items():
                    if k == 'img': continue
                    X[k] = np.repeat(np.expand_dims(v, axis=0), X['img'].shape[0], axis=0)
            else:
                X = crop_tta(X, crop_size=self.crop_size, num_crops=self.crop_tta)
                X = np.asarray([self.process_image(_) for _ in X])
        else:
            if isinstance(X, dict):
                if self.square: X['img'] = square_crop(X['img'], random=not self.test_mode)
                if self.jsd: raise Exception('JSD not supported when using metadata')
                X['img'] = self.process_image(X['img'])
            else:
                if self.square: X = square_crop(X, random=not self.test_mode)
                if self.jsd and not self.test_mode: X_orig = X.copy()
                X = self.process_image(X)
                if self.jsd and not self.test_mode:
                    # Additional aug
                    X_aug  = self.process_image(X_orig)
                    X_orig = self.process_image(X_orig, jsd=True)
        
        if self.onehot and not self.test_mode:
            onehot_y = {
                0: [1.,0.,0.],
                1: [0.,1.,0.],
                2: [0.,0.,1.]
            }
            y = self.labels[i]
            if isinstance(y, str):
                y = y.split(',')
                y = [float(_) for _ in y]
            else:
                y = onehot_y[int(y)]
            if len(y) == 1:
                y = onehot_y[int(y[0])]
        else:
            y = self.labels[i]

        if isinstance(y, str):
            y = float(y)

        if self.flip and not self.test_mode:
            # X.shape = (C, H, W)
            mode = np.random.randint(5)
            if isinstance(X, dict):
                X['img'] = self.flip_array(X['img'], mode)
            else:
                X = self.flip_array(X, mode)
            if self.jsd and not self.test_mode:
                X_aug  = self.flip_array(X_aug, mode)
                X_orig = self.flip_array(X_orig, mode)

        if isinstance(X, dict):
            X = {k: torch.tensor(v) for k,v in X.items()}
        else:
            X = torch.tensor(X)
        if self.jsd and not self.test_mode: 
            X = (torch.tensor(X_orig), torch.tensor(X), torch.tensor(X_aug))

        y = torch.tensor(y)
        return X, y


class SiameseDataset(Dataset):


    def __init__(self,
                 imgfiles,
                 labels,
                 pad=None,
                 resize=None,
                 transform=None,
                 crop=None,
                 preprocessor=None,
                 flip=False,
                 verbose=True,
                 test_mode=False):
        self.imgfiles = imgfiles
        self.labels = labels
        self.pad = pad 
        self.resize = resize
        self.transform = transform
        self.crop = crop 
        self.preprocessor = preprocessor
        self.flip = flip
        self.verbose = verbose
        self.test_mode = test_mode

        self.posfiles = [self.imgfiles[i] for i in range(len(self.imgfiles)) if self.labels[i] == 1]        
        self.negfiles = [self.imgfiles[i] for i in range(len(self.imgfiles)) if self.labels[i] == 0]

        self.get = self.get_test if self.test_mode else self.get_train


    def process_image(self, X):
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        if self.transform: X = self.transform(image=X)['image']
        if self.crop: X = self.crop(image=X)['image']
        if self.preprocessor: X = self.preprocessor.preprocess(X)
        return X.transpose(2, 0, 1)


    def _read_image(self, fp):
        X = cv2.imread(fp)
        if _isnone(X):
            X = cv2.imread(fp.replace('jpg','png'))
        return X


    def get_test(self, i):
        try:
            return self._read_image(self.imgfiles[i])
        except Exception as e:
            if self.verbose: print(e)
            return None


    def get_train(self, i):
        try:
            pair_type = np.random.randint(4)
            if pair_type <= 1:
                X1 = self._read_image(np.random.choice(self.posfiles))
                X2 = self._read_image(np.random.choice(self.negfiles))
            elif pair_type == 2:
                X1 = self._read_image(np.random.choice(self.posfiles))
                X2 = self._read_image(np.random.choice(self.posfiles))
            elif pair_type == 3:
                X1 = self._read_image(np.random.choice(self.negfiles))
                X2 = self._read_image(np.random.choice(self.negfiles))
            return [X1, X2], pair_type
        except Exception as e:
            if self.verbose: print(e)
            return None


    def __len__(self):
        return len(self.imgfiles)


    def __getitem__(self, i):
        X = self.get(i)
        while _isnone(X):
            if self.verbose: print('Failed to read {} !'.format(self.imgfiles[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        if self.test_mode:
            X = self.process_image(X)
            y = self.labels[i]
        else:
            X, pair_type = X
            X = np.asarray([self.process_image(_) for _ in X])
            if pair_type <= 1:
                y = 0 # different
            else:
                y = 1 # same
        

        if self.flip and not self.test_mode:
            # X.shape = (2, C, H, W)
            mode = np.random.randint(5)
            if mode == 0:
                X = X[...,::-1]
            elif mode == 1:
                X = X[...,::-1,:]
            elif mode == 2:
                X = X[...,::-1,::-1]
            elif mode == 3 and X.shape[-1] == X.shape[-2]:
                X = X.swapaxes(-1, -2)
            X = np.ascontiguousarray(X)

        X = torch.tensor(X)
        y = torch.tensor(y)

        return X, y


class Upsampler(Sampler):


    def __init__(self, dataset, upsample_factor=25):
        super().__init__(data_source=dataset)
        self.labels = np.asarray(dataset.labels)
        self.num_pos = np.sum(self.labels >= 0.5)
        self.num_neg = np.sum(self.labels <  0.5)
        self.upsample_factor = upsample_factor
        self.length = self.num_neg + upsample_factor * self.num_pos


    def __len__(self):
        return self.length 


    def __iter__(self):
        indices = []
        indices += list(np.where(self.labels < 0.5)[0])
        indices += list(np.random.choice(np.where(self.labels >= 0.5)[0], self.upsample_factor * self.num_pos, replace=True))
        indices = np.random.permutation(indices)
        return iter(indices.tolist())


class BalancedSampler(Sampler):


    def __init__(self, dataset, weights=[2,1], pos_label=1):
        super().__init__(data_source=dataset)
        self.labels = np.asarray(dataset.labels)
        self.pos_label = pos_label
        self.num_pos = np.sum(self.labels == pos_label)
        self.num_neg = np.sum(self.labels != pos_label)
        # weights ordered as [neg, pos]
        self.weights = np.asarray(weights)
        self.weights = self.weights / np.sum(self.weights)
        self.length  = len(dataset.imgfiles)


    def __len__(self):
        return self.length 


    def __iter__(self):
        indices = []
        sample_neg = int(self.length * self.weights[0])
        sample_pos = int(self.length * self.weights[1])
        indices += list(np.random.choice(np.where(self.labels == self.pos_label)[0], sample_pos, replace=self.num_pos<sample_pos))
        indices += list(np.random.choice(np.where(self.labels != self.pos_label)[0], sample_neg, replace=self.num_neg<sample_neg))
        indices = np.random.permutation(indices)
        return iter(indices.tolist())



class BenignSampler(Sampler):


    def __init__(self, dataset, probas=None, weights=[2,1], pos_label=1):
        super().__init__(data_source=dataset)
        self.dataset = dataset # store
        self.labels = np.asarray(dataset.labels)
        self.imgfiles = np.asarray(dataset.imgfiles)
        self.pos_label = pos_label
        # Need to map image file to indices
        self.img2index = {i : index for index, i in enumerate(self.imgfiles)}
        self.negfiles = [im for i, im in enumerate(self.imgfiles) if self.labels[i] != pos_label]
        self.num_pos = np.sum(self.labels == pos_label)
        self.num_neg = np.sum(self.labels != pos_label)
        # weights ordered as [neg, pos]
        self.weights = np.asarray(weights)
        self.weights = self.weights / np.sum(self.weights)
        self.length  = self.num_pos * 4
        if type(probas) == type(None):
            # Assign equal weight to all benigns
            p = 1.0 / len(self.negfiles)
            probas = {i : p for i in self.negfiles}
        self.probas = probas


    def __len__(self):
        return self.length 


    def __iter__(self):
        indices = []
        probas = {self.img2index[k] : v for k,v in self.probas.items()}
        sample_neg = int(self.length * self.weights[0])
        sample_pos = int(self.length * self.weights[1])
        indices += list(np.random.choice(np.where(self.labels == self.pos_label)[0], sample_pos, replace=self.num_pos<sample_pos))
        # For negatives, sample based on weight
        keys = [*probas]
        values = np.asarray([probas[k] for k in keys])
        indices += list(np.random.choice(keys, sample_neg, replace=sample_neg>len(keys), p=values))
        indices = np.random.permutation(indices)
        return iter(indices.tolist())














