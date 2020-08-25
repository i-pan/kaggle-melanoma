import imagesize
import glob

from tqdm import tqdm


train_images = glob.glob('../../data/jpeg/train/*.jpg')
test_images  = glob.glob('../../data/jpeg/test/*.jpg')

train_sizes = []
for imfile in tqdm(train_images, total=len(train_images)):
    train_sizes += [imagesize.get(imfile)]

test_sizes = []
for imfile in tqdm(test_images, total=len(test_images)):
    test_sizes += [imagesize.get(imfile)]

