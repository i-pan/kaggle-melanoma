import pandas as pd
import yaml
import cv2

from factory.builder import build_dataloader


with open('configs/bee/bee022.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


config['dataset']['params']['crop_tta'] = 16
df = pd.read_csv('../data/combined_train_cdeotte_meta.csv')

imgfiles = [f'../data/jpeg/train/{i}.jpg' for i in df['image_name']]
imgfiles = ['../data/jpeg/train/ISIC_0151200.jpg',
       '../data/jpeg/train/ISIC_0227038.jpg',
       '../data/jpeg/train/ISIC_0230209.jpg',
       '../data/jpeg/train/ISIC_0236778.jpg',
       '../data/jpeg/train/ISIC_0280749.jpg',
       '../data/jpeg/train/ISIC_0343061.jpg',
       '../data/jpeg/train/ISIC_0361529.jpg',
       '../data/jpeg/train/ISIC_0384214.jpg',
       '../data/jpeg/train/ISIC_0401250.jpg']
labels = [0] * len(imgfiles)
data_info = {
    'imgfiles': imgfiles,
    'labels': labels
}

loader = build_dataloader(config, data_info, 'predict')
loader.dataset.preprocessor = None

loader = iter(loader)
data = next(loader)
data[0].shape
images = data[0].numpy()[0,1].transpose(1,2,0)
images.shape

cv2.imwrite('/home/ianpan/test_crop_tta.png', images)

while data[0].size(0) == 1:
    data = next(loader)
