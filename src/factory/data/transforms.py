import albumentations as albu
import numpy as np
import cv2

from .gridmask import GridMask
from .augmix import AugMix
from .rand_augment import RandAugment


def pad_to_ratio(array, ratio):
    # Default is ratio=1 aka pad to create square image
    ratio = float(ratio)
    # Given ratio, what should the height be given the width? 
    h, w = array.shape[:2]
    desired_h = int(w * ratio)
    # If the height should be greater than it is, then pad height
    if desired_h > h: 
        hdiff = int(desired_h - h) ; hdiff = int(hdiff / 2)
        pad_list = [(hdiff, desired_h-h-hdiff), (0,0), (0,0)]
    # If height should be smaller than it is, then pad width
    elif desired_h < h: 
        desired_w = int(h / ratio)
        wdiff = int(desired_w - w) ; wdiff = int(wdiff / 2)
        pad_list = [(0,0), (wdiff, desired_w-w-wdiff), (0,0)]
    elif desired_h == h: 
        return array 
    return np.pad(array, pad_list, 'constant', constant_values=np.min(array))


def resize(x, y=None):
    if y is None: y = x
    return albu.Compose([
        albu.Resize(x, y, always_apply=True, interpolation=cv2.INTER_CUBIC, p=1)
        ], p=1)


def crop(x, y=None, test_mode=False):
    if y is None: y = x
    if test_mode:
        return albu.Compose([
            albu.CenterCrop(x, y, always_apply=True, p=1)
            ], p=1, additional_targets={'image{}'.format(_) : 'image' for _ in range(1, 101)})
    else:
        return albu.Compose([
            albu.RandomCrop(x, y, always_apply=True, p=1)
            ], p=1, additional_targets={'image{}'.format(_) : 'image' for _ in range(1, 101)})


def vanilla_transform(p):
    return albu.Compose([
        # Flips
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # Shift, scale, rotate
        albu.ShiftScaleRotate(rotate_limit=15, 
                              scale_limit=(-0.15, 0.1),  
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=[255, 255, 255],
                              p=0.5),
        # Noise
        albu.IAAAdditiveGaussianNoise(p=0.2),
        # Contrast, brightness
        albu.OneOf([
                albu.RandomGamma(p=1),
                albu.RandomContrast(p=1),
                albu.RandomBrightness(p=1)
            ], p=0.5),
        # Blur
        albu.GaussianBlur(p=0.2),
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1, 101)})


def minty_transform(p):
    return albu.Compose([
        # Shift, scale, rotate
        albu.ShiftScaleRotate(rotate_limit=15, 
                              scale_limit=(-0.15, 0.1),  
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=[255, 255, 255],
                              p=0.25),
        # Noise
        albu.IAAAdditiveGaussianNoise(p=0.25),
        # Contrast, brightness
        albu.OneOf([
                albu.RandomGamma(p=1),
                albu.RandomContrast(p=1),
                albu.RandomBrightness(p=1)
            ], p=0.55),
        # Blur
        albu.GaussianBlur(p=0.25),
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1, 101)})


def choco_transform(p):
    return albu.Compose([
        # Flips
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # Shift, scale, rotate
        albu.ShiftScaleRotate(rotate_limit=0, 
                         scale_limit=0.15,  
                         border_mode=cv2.BORDER_CONSTANT, 
                         value=[255, 255, 255],
                         p=0.5),
        # Noise
        albu.IAAAdditiveGaussianNoise(p=0.2),
        # Color
        albu.Solarize(p=0.2),
        albu.ToGray(p=0.2),
        # Contrast, brightness
        albu.OneOf([
                albu.RandomGamma(p=1),
                albu.RandomContrast(p=1),
                albu.RandomBrightness(p=1)
            ], p=0.5)
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1, 101)})


def spatial_and_noise(p):
    return albu.Compose([
        albu.ShiftScaleRotate(rotate_limit=30,
                              scale_limit=15,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=-1024,
                              p=0.5),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.2),
        albu.IAAPiecewiseAffine(p=0.2)
    ], p=p, additional_targets={'image{}'.format(_) : 'image' for _ in range(1, 101)})


def grid_mask(**kwargs):
    return GridMask(**kwargs)

class Preprocessor(object):
    """
    Object to deal with preprocessing.
    Easier than defining a function.
    """
    def __init__(self, image_range, input_range, mean, sdev):
        self.image_range = image_range
        self.input_range = input_range
        self.mean = mean 
        self.sdev = sdev


    def preprocess(self, img, mode='numpy'): 
        ''' 
        Preprocess an input image. 
        '''

        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])

        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])

        image_range = image_max - image_min
        model_range = model_max - model_min 

        img = (((img - image_min) * model_range) / image_range) + model_min 

        if mode == 'numpy': 

            if img.shape[-1] == 3: 
            # Assume image is RGB 
            # Unconvinced that RGB<>BGR matters for transfer learning ...
                img = img[..., ::-1].astype('float32')
                img[..., 0] -= self.mean[0] 
                img[..., 1] -= self.mean[1] 
                img[..., 2] -= self.mean[2] 
                img[..., 0] /= self.sdev[0] 
                img[..., 1] /= self.sdev[1] 
                img[..., 2] /= self.sdev[2] 

            else:
                avg_mean = np.mean(self.mean)
                avg_sdev = np.mean(self.sdev)

                img -= avg_mean
                img /= avg_sdev

        elif mode == 'torch':

            if img.size(1) == 3:
                img = img[:,[2,1,0]]
                img[:, 0] -= self.mean[0] 
                img[:, 1] -= self.mean[1] 
                img[:, 2] -= self.mean[2] 
                img[:, 0] /= self.sdev[0] 
                img[:, 1] /= self.sdev[1] 
                img[:, 2] /= self.sdev[2]

            else:
                avg_mean = np.mean(self.mean)
                avg_sdev = np.mean(self.sdev)

                img -= avg_mean
                img /= avg_sdev 

        return img