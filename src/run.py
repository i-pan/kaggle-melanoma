import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
import copy
import glob
import re
import os, os.path as osp

from collections import defaultdict
from tqdm import tqdm

try:
    from .factory import set_reproducibility
    from .factory import train as factory_train
    from .factory import evaluate as factory_evaluate
    from .factory import builder 
    from .factory.data.utils import cudaify
except:
    from factory import set_reproducibility
    import factory.train as factory_train
    import factory.evaluate as factory_evaluate
    import factory.builder as builder
    from factory.data.utils import cudaify


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('mode', type=str) 
    parser.add_argument('--gpu', type=lambda s: [int(_) for _ in s.split(',')] , default=[0])
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--load-previous', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--save-file', type=str)
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--checkpoint-dir', type=str)
    return parser.parse_args()


def create_logger(cfg, mode):
    logfile = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], 'log_{}.txt'.format(mode))
    if osp.exists(logfile): os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))
    return logger


def set_inference_batch_size(cfg):
    if 'evaluation' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['evaluation'].keys(): 
            cfg['evaluation']['batch_size'] = 2*cfg['train']['batch_size']

    if 'test' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['test'].keys(): 
            cfg['test']['batch_size'] = 2*cfg['train']['batch_size']

    if 'predict' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['predict'].keys(): 
            cfg['predict']['batch_size'] = 2*cfg['train']['batch_size']

    return cfg 


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.num_workers > 0:
        if 'transform' not in cfg.keys():
            cfg['transform'] = {}
        cfg['transform']['num_workers'] = args.num_workers

    if args.mode != 'predict_kfold':
        if args.backbone:
            cfg['model']['params']['backbone'] = args.backbone
            cfg['evaluation']['params']['save_checkpoint_dir'] = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], args.backbone)

    if args.batch_size:
        cfg['train']['batch_size'] = args.batch_size
        cfg['evaluation']['params']['save_checkpoint_dir'] = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], f'bs{args.batch_size}')

    if args.fold >= 0:
        if args.mode == 'train':
            cfg['dataset']['outer_only'] = True
            cfg['dataset']['outer_fold'] = args.fold
            cfg['evaluation']['params']['save_checkpoint_dir'] = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], f'fold{args.fold}')
            cfg['seed'] = int('{}{}'.format(cfg['seed'], args.fold))
        elif args.mode == 'test':
            cfg['dataset']['outer_only'] = True
            cfg['dataset']['outer_fold'] = args.fold
            cfg['test']['savefile'] = osp.join(osp.dirname(cfg['test']['savefile']), f'fold{args.fold}', cfg['test']['savefile'].split('/')[-1]) 

    if args.checkpoint:
        cfg['test']['checkpoint'] = args.checkpoint

    if args.load_previous:
        cfg['train']['load_previous'] = args.load_previous

    if args.eps:
        cfg['optimizer']['params']['eps'] = args.eps
        
    cfg = set_inference_batch_size(cfg)

    # We will set all the seeds we can, in vain ...
    set_reproducibility(cfg['seed'])
    # Set GPU
    if len(args.gpu) == 1:
        torch.cuda.set_device(args.gpu[0])

    if 'predict' in args.mode:
        eval(args.mode)(args, cfg)
        return

    if cfg['transform']['augment'] == 'grid_mask':
        assert 'grid_mask' in cfg['train']['params']

    if 'mixup' not in cfg['train']['params'].keys():
        cfg['train']['params']['mixup'] = None

    if 'cutmix' not in cfg['train']['params'].keys():
        cfg['train']['params']['cutmix'] = None

    # Make directory to save checkpoints
    if not osp.exists(cfg['evaluation']['params']['save_checkpoint_dir']): 
        os.makedirs(cfg['evaluation']['params']['save_checkpoint_dir'])

    # Load in labels with CV splits
    df = pd.read_csv(cfg['dataset']['csv_filename'])
    if cfg['dataset'].pop('exclude_unknown', False):
        df = df[df['diagnosis'] != 'unknown']
    
    if cfg['dataset'].pop('isic2020', False):
        df = df[df['isic'] == 2020]

    ofold = cfg['dataset']['outer_fold']
    ifold = cfg['dataset']['inner_fold']

    train_df, valid_df, test_df = get_train_valid_test(cfg, df, ofold, ifold)

    logger = create_logger(cfg, args.mode)
    logger.info('Saving to {} ...'.format(cfg['evaluation']['params']['save_checkpoint_dir']))

    if args.mode == 'find_lr':
        cfg['optimizer']['params']['lr'] = cfg['find_lr']['params']['start_lr']
        find_lr(args, cfg, train_df, valid_df)
    elif args.mode == 'train':
        train(args, cfg, train_df, valid_df)
    elif args.mode == 'test':
        test(args, cfg, train_df, test_df)


def get_train_valid_test(cfg, df, ofold, ifold):
    # Get train/validation set
    if cfg['dataset']['outer_only']: 
        print('<outer_only>')
        print('<outer_fold> {}'.format(ofold))
        # valid and test are essentially the same here
        train_df = df[df['outer'] != ofold]
        valid_df = df[df['outer'] == ofold]
        test_df  = df[df['outer'] == ofold]
    else:
        print('<inner_fold> {}'.format(ifold))
        print('<outer_fold> {}'.format(ofold))
        test_df = df[df['outer'] == ofold]
        df = df[df['outer'] != ofold]
        train_df = df[df['inner{}'.format(ofold)] != ifold]
        valid_df = df[df['inner{}'.format(ofold)] == ifold]
    return train_df, valid_df, test_df


def get_invfreq_weights(values, scale=None):
    logger = logging.getLogger('root')
    values, counts = np.unique(values, return_counts=True)
    num_samples = np.sum(counts)
    freqs = counts / float(num_samples)
    max_freq = np.max(freqs)
    invfreqs = max_freq / freqs
    if scale == 'log':
        logger.info('  Log scaling ...') 
        invfreqs = np.log(invfreqs+1)
    elif scale == 'sqrt':
        logger.info('  Square-root scaling ...')
        invfreqs = np.sqrt(invfreqs)
    invfreqs = invfreqs / np.min(invfreqs)
    return invfreqs


def setup(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    if isinstance(cfg['dataset']['data_dir'], list):
        data_dir_dict = {
            2019: cfg['dataset']['data_dir'][0],
            2020: cfg['dataset']['data_dir'][1]
        }
        if len(cfg['dataset']['data_dir']) == 3:
            data_dir_dict[2021] = cfg['dataset']['data_dir'][2]
        train_images = []
        for rownum, row in train_df.iterrows():
            data_dir = data_dir_dict[row.isic]
            imgfile = osp.join(data_dir, f'{row.image}.jpg')
            train_images += [imgfile]
        valid_images = []
        for rownum, row in valid_df.iterrows():
            data_dir = data_dir_dict[row.isic]
            imgfile = osp.join(data_dir, f'{row.image}.jpg')
            valid_images += [imgfile]
    else:
        train_images = [osp.join(cfg['dataset']['data_dir'], f'{_}.jpg') for _ in train_df['image'].values]
        valid_images = [osp.join(cfg['dataset']['data_dir'], f'{_}.jpg') for _ in valid_df['image'].values]
    train_data_info = {
        'imgfiles': train_images,
        'labels': train_df['label'].values
    }
    valid_data_info = {
        'imgfiles': valid_images,
        'labels': valid_df['label'].values
    }
    if cfg['dataset'].pop('meta', False):
        train_data_info['meta'] = [dict(age=row['age_cat'],sex=row['sex'],ant=row['anatom_site_general_challenge']) for rownum, row in train_df.iterrows()]
        valid_data_info['meta'] = [dict(age=row['age_cat'],sex=row['sex'],ant=row['anatom_site_general_challenge']) for rownum, row in valid_df.iterrows()]
    train_loader = builder.build_dataloader(cfg, data_info=train_data_info, mode='train')
    valid_loader = builder.build_dataloader(cfg, data_info=valid_data_info, mode='valid')

    ARC = False
    if cfg['model']['name'] in ('ArcNet', 'SiameseNet'):
        ARC = True
        if 'isic' in train_df.columns:
            mel_df = train_df[(train_df['label'] == 1) & (train_df['isic'] == 2020)]
        else:
            mel_df = train_df[train_df['label'] == 1]
        mel_df = mel_df.drop_duplicates()
        arc_data_info = {
            'imgfiles': [osp.join(cfg['dataset']['data_dir'], f'{_}.jpg') for _ in mel_df['image'].values],
            'labels': mel_df['label'].values        
        }
        arc_loader = builder.build_dataloader(cfg, data_info=arc_data_info, mode='predict')
        print(f'{len(arc_loader)} melanoma examples will be used as reference ...')

    OHEM = False
    if 'ohem' in cfg['train']['params'] and cfg['train']['params']['ohem']:
        print('Creating benign loader ...')
        OHEM = True
        benign_df = train_df[train_df['label'] == 0]
        benign_data_info = {
            'imgfiles': [osp.join(cfg['dataset']['data_dir'], f'{_}.jpg') for _ in benign_df['image'].values],
            'labels': benign_df['label'].values        
        }
        benign_loader = builder.build_dataloader(cfg, data_info=benign_data_info, mode='predict')

    # Adjust steps per epoch if necessary (i.e., equal to 0)
    # We assume if gradient accumulation is specified, then the user
    # has already adjusted the steps_per_epoch accordingly in the 
    # config file
    steps_per_epoch = cfg['train']['params']['steps_per_epoch']
    gradient_accmul = cfg['train']['params']['gradient_accumulation']
    if steps_per_epoch == 0:
        cfg['train']['params']['steps_per_epoch'] = len(train_loader)

    # Generic build function will work for model/loss
    logger.info('Building [{}] architecture ...'.format(cfg['model']['name']))
    if 'backbone' in cfg['model']['params'].keys():
        logger.info('  Using [{}] backbone ...'.format(cfg['model']['params']['backbone']))
    if 'pretrained' in cfg['model']['params'].keys():
        logger.info('  Pretrained weights : {}'.format(cfg['model']['params']['pretrained']))
    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    model = model.train().cuda()

    if cfg['loss']['params'] is None:
        cfg['loss']['params'] = {}

    if re.search(r'^OHEM', cfg['loss']['name']):
        cfg['loss']['params']['total_steps'] = cfg['train']['params']['num_epochs'] * cfg['train']['params']['steps_per_epoch']

    if cfg['loss']['name'] == 'CrossEntropyLoss':
        weighted = cfg['loss'].pop('weighted', False)
        if weighted:
            wts = get_invfreq_weights(train_data_info['labels'], scale=weighted)
            cfg['loss']['params']['weight'] = torch.tensor(wts)
            logger.info('Using the following class weights:')
            for i in range(len(wts)):
                logger.info(f'  Class {i} : {wts[i]:.4f}')

    criterion = builder.build_loss(cfg['loss']['name'], cfg['loss']['params'])
    optimizer = builder.build_optimizer(
        cfg['optimizer']['name'], 
        model.parameters(), 
        cfg['optimizer']['params'])
    scheduler = builder.build_scheduler(
        cfg['scheduler']['name'], 
        optimizer, 
        cfg=cfg)

    if len(args.gpu) > 1:
        print(f'DEVICES : {args.gpu}')
        model = nn.DataParallel(model, device_ids=args.gpu)
        if args.gpu[0] != 0:
            model.to(f'cuda:{model.device_ids[0]}')
            
    if ARC: valid_loader = (valid_loader, arc_loader)
    if OHEM: train_loader = (train_loader, benign_loader)
    
    return cfg, \
           train_loader, \
           valid_loader, \
           model, \
           optimizer, \
           criterion, \
           scheduler

def find_lr(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    logger.info('FINDING LR ...')

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    finder = factory_train.LRFinder(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_checkpoint_dir=cfg['evaluation']['params']['save_checkpoint_dir'],
        logger=logger,
        gradient_accumulation=cfg['train']['params']['gradient_accumulation'],
        mixup=cfg['train']['params']['mixup'],
        cutmix=cfg['train']['params']['cutmix'])

    finder.find_lr(**cfg['find_lr']['params'])

    logger.info('Results are saved in : {}'.format(osp.join(finder.save_checkpoint_dir, 'lrfind.csv')))

def train(args, cfg, train_df, valid_df):
    
    logger = logging.getLogger('root')

    logger.info('TRAINING : START')

    logger.info('TRAIN: n={}'.format(len(train_df)))
    valid_df = valid_df.drop_duplicates()
    logger.info('VALID: n={}'.format(len(valid_df)))

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    ARC = False
    if isinstance(valid_loader, tuple):
        valid_loader, arc_loader = valid_loader
        ARC = True

    if 'load_previous' in cfg['train'] and cfg['train']['load_previous']:
        print(f'Loading previously trained model from <{cfg["train"]["load_previous"]}> ...')
        weights = torch.load(cfg['train']['load_previous'], map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(weights)
        except:
            weights = {k.replace('module.', ''):v for k,v in weights.items()}
            model.load_state_dict(weights)

    model = model.train()

    evaluator = getattr(factory_evaluate, cfg['evaluation']['evaluator'])
    if ARC: 
        cfg['evaluation']['params']['arc_loader'] = arc_loader
    evaluator = evaluator(loader=valid_loader,
        **cfg['evaluation']['params'])

    trainer = getattr(factory_train, cfg['train']['trainer'])
    trainer = trainer(loader=train_loader,
        model=model,
        optimizer=optimizer,
        schedule=scheduler,
        criterion=criterion,
        evaluator=evaluator,
        logger=logger)
    trainer.train(**cfg['train']['params'])


def test(args, cfg, train_df, test_df):

    test_df = test_df.drop_duplicates()
    data_info = {  
        'imgfiles': [osp.join(cfg['dataset']['data_dir'], f'{_}.jpg') for _ in test_df['image'].values],
        'labels': test_df['label'].values
    }

    loader = builder.build_dataloader(cfg, data_info=data_info, mode='test')

    ARC = False
    if cfg['model']['name'] in ('ArcNet', 'SiameseNet'):
        ARC = True
        if 'use_benign' in cfg['test'] and cfg['test']['use_benign']:
            LBL = 0
        else:
            LBL = 1
        if 'isic' in train_df.columns:
            mel_df = train_df[(train_df['label'] == LBL) & (train_df['isic'] == 2020)]
        else:
            mel_df = train_df[train_df['label'] == LBL]
        mel_df = mel_df.drop_duplicates()
        arc_data_info = {
            'imgfiles': [osp.join(cfg['dataset']['data_dir'], f'{_}.jpg') for _ in mel_df['image'].values],
            'labels': mel_df['label'].values        
        }
        arc_loader = builder.build_dataloader(cfg, data_info=arc_data_info, mode='predict')
        print(f'{len(arc_loader)} melanoma examples will be used as reference ...')

    print('TESTING : START')
    print('TEST (N={})'.format(len(test_df)))
    print(f'Saving predictions to {cfg["test"]["savefile"]} ...')

    def create_model(cfg):
        model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
        print('Loading <{}> model from <{}> ...'.format(cfg['model']['name'], cfg['test']['checkpoint']))
        model.load_state_dict(torch.load(cfg['test']['checkpoint'], map_location=lambda storage, loc: storage))
        model = model.eval().cuda()
        return model

    model = create_model(cfg)

    if 'params' not in cfg['test'] or not isinstance(cfg['test']['params'], dict):
        cfg['test']['params'] = {}

    predictor = getattr(factory_evaluate, cfg['test']['predictor'])
    predictor = predictor(loader=loader,
        **cfg['test']['params'])
    if ARC: 
        predictor.arc_loader = arc_loader

    y_true, y_pred, _ = predictor.predict(model, criterion=None, epoch=None)

    save_dir = osp.dirname(cfg['test']['savefile'])
    if not osp.exists(save_dir): os.makedirs(save_dir)

    pickled = {
            'image_id': [_.split('/')[-1].split('.')[0] for _ in data_info['imgfiles']],
            'y_pred': y_pred,
            'y_true': y_true
        }
    with open(cfg['test']['savefile'], 'wb') as f:
        pickle.dump(pickled, f)


def predict(args, cfg):

    # Get data directory
    data_dir = cfg['dataset']['data_dir']

    if 'csv_filename' in cfg['dataset']:
        df = pd.read_csv(cfg['dataset']['csv_filename'])
        images = [osp.join(data_dir, f'{_}.jpg') for _ in df['image_name']]
    else:
        images = glob.glob(osp.join(osp.join(data_dir, '*')))

    data_info = {
        'imgfiles': images,
        'labels': [0] * len(images),
    }

    if cfg['dataset'].pop('meta', False):
        data_info['meta'] = [dict(age=row['age_cat'],sex=row['sex'],ant=row['anatom_site_general_challenge']) for rownum, row in df.iterrows()]

    if 'params' not in cfg['predict'] or not isinstance(cfg['predict']['params'], dict):
        cfg['predict']['params'] = {}

    loader = builder.build_dataloader(cfg, data_info=data_info, mode='predict')
    if 'arc' in cfg and cfg['arc']:
        assert 'arc_csvfile' in cfg['dataset']
        assert 'arc_datadir' in cfg['dataset']
        df = pd.read_csv(cfg['dataset']['arc_csvfile'])
        mel_df = df[df['label'] == 1].drop_duplicates()
        arc_info = {
            'imgfiles': [osp.join(cfg['dataset']['arc_datadir'], f'{_}.jpg') for _ in mel_df['image']],
            'labels': mel_df['label'].values
        }
        arc_loader = builder.build_dataloader(cfg, data_info=arc_info, mode='predict')

        cfg['predict']['params']['arc_loader'] = arc_loader

    print('PREDICTING : START')
    print('PREDICT (N={})'.format(len(images)))

    # Replace checkpoints, if necessary
    if 'model_checkpoints' in cfg and type(cfg['model_checkpoints']) != type(None):
        assert len(cfg['model_checkpoints']) == len(cfg['model_configs'])
        assert type(cfg['model_checkpoints']) == list
        replace_checkpoint_paths = True

    model_configs = []
    for cfg_ind, cfgfile in enumerate(cfg['model_configs']):
        with open(cfgfile) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        model_cfg['model']['params']['pretrained'] = None
        if replace_checkpoint_paths:
            model_cfg['test']['checkpoint'] = cfg['model_checkpoints'][cfg_ind]
        model_configs.append(model_cfg)

    def create_model(cfg):
        model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
        print('Loading <{}> model from <{}> ...'.format(cfg['model']['name'], cfg['test']['checkpoint']))
        checkpoint = cfg['test']['checkpoint']
        weights = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        weights = {k.replace('module.', '') : v for k,v in weights.items()}
        model.load_state_dict(weights)
        model = model.eval().cuda()
        return model

    models = [create_model(model_cfg) for ind, model_cfg in enumerate(model_configs)]

    predictor = getattr(factory_evaluate, cfg['predict']['predictor'])
    predictor = predictor(loader=loader,
        **cfg['predict']['params'])

    final_pred = []
    for m in models:
        if 'tta' in cfg and cfg['tta']:
            m.tta = True
        _, y_pred, _ = predictor.predict(m, criterion=None, epoch=None)
        final_pred += [y_pred]
    #final_pred = np.mean(np.asarray(final_pred), axis=0)

    save_dir = osp.dirname(cfg['predict']['savefile'])
    if not osp.exists(save_dir): os.makedirs(save_dir)

    pickled = {
            'image_id': [_.split('/')[-1] for _ in images],
            'label': final_pred
        }
    with open(cfg['predict']['savefile'], 'wb') as f:
        pickle.dump(pickled, f)


def predict_kfold(args, cfg):

    if args.save_file:
        cfg['predict']['savefile'] = args.save_file

    if args.model_config:
        cfg['model_configs'] = [args.model_config]

    if args.checkpoint_dir:
        # format will be config/backbone/fold/checkpoint.pth
        checkpoints = glob.glob(osp.join(args.checkpoint_dir, 'fold*', '*.PTH'))
        checkpoint_dict = defaultdict(list)
        for ckpt in checkpoints:
            fold = ckpt.split('/')[-2]
            checkpoint_dict[fold] += [ckpt]
        best_metric_per_fold = []
        for k,v in checkpoint_dict.items():
            metrics = [float(_.split('/')[-1].split('-')[-1].replace('.PTH', '')) for _ in v]
            checkpoint_dict[k] = v[np.argmax(metrics)]
            best_metric_per_fold += [np.max(metrics)]
        print(f'KFOLD CV: {np.mean(best_metric_per_fold):.4f}')
        cfg['model_checkpoints'] = [v for v in checkpoint_dict.values()]

    save_dir = osp.dirname(cfg['predict']['savefile'])
    if not osp.exists(save_dir): os.makedirs(save_dir)
    print(f'Saving predictions to {cfg["predict"]["savefile"]} ...')

    # Get data directory
    data_dir = cfg['dataset']['data_dir']

    if 'csv_filename' in cfg['dataset']:
        df = pd.read_csv(cfg['dataset']['csv_filename'])
        images = [osp.join(data_dir, f'{_}.jpg') for _ in df['image_name']]
    else:
        images = glob.glob(osp.join(osp.join(data_dir, '*')))

    data_info = {
        'imgfiles': images,
        'labels': [0] * len(images),
    }

    if cfg['dataset'].pop('meta', False):
        data_info['meta'] = [dict(age=row['age_cat'],sex=row['sex'],ant=row['anatom_site_general_challenge']) for rownum, row in df.iterrows()]

    if 'params' not in cfg['predict'] or not isinstance(cfg['predict']['params'], dict):
        cfg['predict']['params'] = {}

    print('PREDICTING : START')
    print('PREDICT (N={})'.format(len(images)))

    assert 'model_checkpoints' in cfg and isinstance(cfg['model_checkpoints'], list)
    assert len(np.unique(cfg['model_configs'])) == 1, 'Different model config files have been specified, please use `predict()`'
    with open(cfg['model_configs'][0]) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        if args.backbone:
            model_cfg['model']['params']['backbone'] = args.backbone
        model_cfg['model']['params']['pretrained'] = None

    model_cfg['transform']['augment'] = None
    model_cfg['transform']['params']  = None
    model_cfg['transform']['num_workers'] = args.num_workers
    model_cfg['dataset'] = cfg['dataset']
    loader = builder.build_dataloader(model_cfg, data_info=data_info, mode='predict')

    def create_model(cfg, checkpoint):
        model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
        print('Loading <{}> model from <{}> ...'.format(cfg['model']['name'], checkpoint))
        weights = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        weights = {k.replace('module.', '') : v for k,v in weights.items()}
        model.load_state_dict(weights)
        model = model.eval().cuda()
        return model

    models = [create_model(model_cfg, ckpt) for ckpt in cfg['model_checkpoints']]
    print(f'{len(models)} models will be used for inference ...')

    final_preds = []
    for batch, labels in tqdm(loader):
        model_preds = []
        batch, labels = cudaify(batch, labels)
        for m in models:
            with torch.no_grad():
                output = m(batch)
            if m.fc.out_features > 1:
                output = torch.softmax(output, dim=1)
            else:
                output = torch.sigmoid(output)
            model_preds += [output.cpu().numpy()]
        final_preds += [model_preds]

    pickled = {
            'image_id': [_.split('/')[-1] for _ in images],
            'label': final_preds
        }

    with open(cfg['predict']['savefile'], 'wb') as f:
        pickle.dump(pickled, f)


if __name__ == '__main__':
    main()












