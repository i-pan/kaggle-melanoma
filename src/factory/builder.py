import numpy as np
import re
import factory.models 
import factory.losses
import factory.data.datasets as datasets
import factory.data.transforms as transforms 
import factory.optim as optim
import factory.train.scheduler as scheduler

from torch.utils.data import DataLoader
from functools import partial


def _check_key(k,d):
    return k not in d.keys() or type(d[k]) == type(None)


def build(lib, name, params):
    return getattr(lib, name)(**params) if params is not None else getattr(lib, name)()


def build_model(name, params):
    return build(factory.models, name, params)


def build_loss(name, params):
    return build(factory.losses, name, params)


def build_optimizer(name, model_params, params):
    return getattr(optim, name)(params=model_params, **params)


def build_scheduler(name, optimizer, cfg):
    # loader is only needed for OnecycleLR
    # So that I can specify steps_per_epoch = 0 for convenience
    params = cfg['scheduler']['params']

    # Some schedulers will require manipulation of config params
    # My specifications were to make it more intuitive for me
    if name == 'CosineAnnealingWarmRestarts':
        params = {
            # Use num_epochs from training parameters
            'T_0': int(cfg['train']['params']['num_epochs'] / params['num_snapshots']),
            'eta_min': params['final_lr']
        }

    if name == 'CosineAnnealingLR':
        params = {
            'T_max': int(cfg['train']['params']['num_epochs']*cfg['train']['params']['steps_per_epoch']),
            'eta_min': params['final_lr']
        }

    if name in ('OneCycleLR', 'CustomOneCycleLR'):
        # Use steps_per_epoch from training parameters
        steps_per_epoch = cfg['train']['params']['steps_per_epoch']
        params['steps_per_epoch'] = steps_per_epoch
        # And num_epochs
        params['epochs'] = cfg['train']['params']['num_epochs']
        # Use learning rate from optimizer parameters as initial learning rate
        init_lr  = cfg['optimizer']['params']['lr']
        final_lr = params.pop('final_lr')
        params['div_factor'] = params['max_lr'] / init_lr
        params['final_div_factor'] = init_lr / final_lr

    schedule = getattr(scheduler, name)(optimizer=optimizer, **params)
    
    # Some schedulers might need more manipulation after instantiation
    if name == 'CosineAnnealingWarmRestarts':
        schedule.T_cur = 0

    # Set update frequency
    if name in ('CosineAnnealingWarmRestarts', 'OneCycleLR', 'CustomOneCycleLR', 'CosineAnnealingLR'):
        schedule.update = 'on_batch'
    elif name in ('ReduceLROnPlateau'):
        schedule.update = 'on_valid'
    else:
        schedule.update = 'on_epoch'

    return schedule


def build_dataloader(cfg, data_info, mode):

    if cfg['transform']['pad_ratio']:
        pad = partial(transforms.pad_to_ratio, ratio=cfg['transform']['pad_ratio'])
    else:
        pad = None

    if 'crop_size' not in cfg['transform'].keys():
        crop = None
    else:
        crop = transforms.crop(x=cfg['transform']['crop_size'][0], 
                               y=cfg['transform']['crop_size'][1], 
                               # always use CenterCrop for prostate
                               # some edge artifact in CycleGAN-augmented images
                               test_mode=mode!='train')

    if cfg['transform']['resize_to']:
        resize = transforms.resize(x=cfg['transform']['resize_to'][0], 
                                   y=cfg['transform']['resize_to'][1])
    else:
        resize = None

    if mode == 'train':
        if cfg['transform']['augment']:
            data_aug = getattr(transforms, cfg['transform']['augment'])(**cfg['transform']['params'])
        else:
            data_aug = None
    else:
        data_aug = None

    preprocessor = transforms.Preprocessor(
        image_range=cfg['transform']['preprocess']['image_range'], 
        input_range=cfg['transform']['preprocess']['input_range'], 
        mean=cfg['transform']['preprocess']['mean'], 
        sdev=cfg['transform']['preprocess']['sdev'])

    # TODO: need some way of handling TTA
    # if mode == 'test' and cfg['test']['tta']:
    #   ...

    dset_params = cfg['dataset']['params'] if cfg['dataset']['params'] is not None else {}

    dset = getattr(datasets, cfg['dataset']['name'])(
        **data_info,
        resize=resize,
        crop=crop,
        pad=pad,
        transform=data_aug if mode == 'train' else None,
        preprocessor=preprocessor,
        test_mode=(mode != 'train'),
        **dset_params)


    if mode != 'train':
        dset.test_mode = True 


    if mode == 'train':
        sampler = None
        if 'sampler' in cfg['dataset'].keys():
            if _check_key('params', cfg['dataset']['sampler']):
                cfg['dataset']['sampler']['params'] = {}
            params = cfg['dataset']['sampler']['params']
            if type(cfg['dataset']['sampler']['name']) != type(None):
                sampler = getattr(datasets, cfg['dataset']['sampler']['name'])
                sampler = sampler(dataset=dset, **params)
        dgen_params = {
            'batch_size': cfg['train']['batch_size'],
            'num_workers': cfg['transform']['num_workers'],
            'shuffle': True if type(sampler) == type(None) else False,
            'drop_last': True,
        }
        if type(sampler) != type(None):
            print('Using sampler {} ...'.format(type(sampler)))
        if sampler: dgen_params['sampler'] = sampler
    elif mode == 'predict':
        dgen_params = {
            'batch_size': 1,
            'num_workers': cfg['transform']['num_workers'],
            'shuffle': False,
            'drop_last': False
        }
    else: 
        dgen_params = {
            'batch_size': cfg[mode.replace('valid','evaluation')]['batch_size'],
            'num_workers': cfg['transform']['num_workers'],
            'shuffle': False,
            'drop_last': False
        }

    loader = DataLoader(dset, **dgen_params)

    return loader