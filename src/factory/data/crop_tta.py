import numpy as np
import math

from scipy.ndimage.interpolation import zoom


def factint(n):
    pos_n = abs(n)
    max_candidate = int(math.sqrt(pos_n))
    for candidate in range(max_candidate, 0, -1):
        if pos_n % candidate == 0:
            break
    return candidate, n // candidate


def crop_tta(image, crop_size, num_crops=9):
    h, w = image.shape[:2]
    # Rotate image so that orientation matches crop orientation
    tall = crop_size[0] > crop_size[1]
    if tall:
        if w > h:
            image = image.swapaxes(0,1)
    else:
        if h > w:
            image = image.swapaxes(0,1)
    image = resize_for_crop(image, crop_size)
    h, w = image.shape[:2]
    if tuple(crop_size) == (h, w): return [image] 
    rows, cols = factint(num_crops)
    assert crop_size[0] <= h and crop_size[1] <= w
    # Get coordinates
    xc = np.linspace(crop_size[0]//2, h-crop_size[0]//2, rows)
    xc = np.unique(xc.astype('int'))
    yc = np.linspace(crop_size[1]//2, w-crop_size[1]//2, cols)
    yc = np.unique(yc.astype('int'))
    # Try again
    minc = min(len(xc), len(yc))
    if len(xc) > len(yc): 
        rows = num_crops // minc
        cols = num_crops // rows
    else:
        cols = num_crops // minc
        rows = num_crops // cols
    xc = np.linspace(crop_size[0]//2, h-crop_size[0]//2, rows)
    xc = np.unique(xc.astype('int'))
    yc = np.linspace(crop_size[1]//2, w-crop_size[1]//2, cols)
    yc = np.unique(yc.astype('int'))        
    centers = []
    for x in xc:
        for y in yc:
            centers += [(x, y)]
    crops = [image[x-crop_size[0]//2:x+crop_size[0]//2,
                   y-crop_size[1]//2:y+crop_size[1]//2]
             for x,y in centers]
    for c in crops: assert c.shape[:2] == crop_size, f'{c.shape[:2]} is not equal to {crop_size}'
    return crops


def resize_for_crop(image, crop_size):
    h, w = image.shape[:2]
    # Rotate image so that orientation matches crop orientation
    tall = crop_size[0] > crop_size[1]
    if tall:
        if w > h:
            image = image.swapaxes(0,1)
    else:
        if h > w:
            image = image.swapaxes(0,1)
    ratio = max(max(crop_size) / max(h,w), min(crop_size) / min(h,w))
    return zoom(image, [ratio, ratio, 1], order=1, prefilter=False)

    