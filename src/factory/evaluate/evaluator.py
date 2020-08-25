import torch
import pickle
import pandas as pd
import numpy as np
import os, os.path as osp
import re

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from .metrics import *
from ..data import cudaify


class Predictor(object):


    def __init__(self,
                 loader,
                 labels_available=True,
                 cuda=True,
                 debug=False,
                 arc_loader=None):

        self.loader = loader
        self.arc_loader = arc_loader
        self.labels_available = labels_available
        self.cuda = cuda
        self.debug = debug

        if self.loader.dataset.square_tta:
            assert self.loader.batch_size == 1

    @staticmethod
    def get_center(vectors):
        avg = np.mean(vectors, axis=0)
        if avg.ndim == 1:
            avg = avg / np.linalg.norm(avg)
        elif avg.ndim == 2:
            avg = avg / np.linalg.norm(avg, axis=1, keepdims=True)
        return avg


    def predict(self, model, criterion, epoch):
        self.epoch = epoch
        y_pred = []
        y_true = []
        losses = []
        if 'arcnet' in str(model).lower():
            melanoma = []
            features = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(self.arc_loader), total=len(self.arc_loader)):
                    # arc_loader should be a loader of MELANOMA IMAGES ONLY from training set
                    batch, labels = data
                    if self.cuda:
                        batch, labels = cudaify(batch, labels)
                    # Get feature
                    melanoma += [model(batch).cpu().numpy()]
                for i, data in tqdm(enumerate(self.loader), total=len(self.loader)):
                    # Validation loader 
                    if self.debug:
                        if i > 10:
                            break
                    batch, labels = data 
                    if self.cuda: 
                        batch, labels = cudaify(batch, labels)
                    features += [model(batch).cpu().numpy()]
                    losses += [0]
                    y_true += list(labels.cpu().numpy())
            if self.debug:
                y_true[-1] = 0
                y_true[-2] = 1
            melanoma = np.vstack(melanoma)
            features = np.vstack(features)
            # Get center of melanoma features
            melanoma = self.get_center(melanoma).reshape(1, -1)
            # Compute distances 
            distances = cosine_similarity(features, melanoma)
            if len(self.arc_loader) > 10000: 
                print('Using distance from benign ...')
                distances = -distances
            return y_true, distances, losses
        elif 'siamesenet' in str(model).lower():
            melanoma = []
            features = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(self.arc_loader), total=len(self.arc_loader)):
                    # arc_loader should be a loader of MELANOMA IMAGES ONLY from training set
                    batch, labels = data
                    if self.cuda:
                        batch, labels = cudaify(batch, labels)
                    # Get feature
                    melanoma += [model.extract_features(batch)]
                for i, data in tqdm(enumerate(self.loader), total=len(self.loader)):
                    # Validation loader 
                    if self.debug:
                        if i > 10:
                            break
                    batch, labels = data 
                    if self.cuda: 
                        batch, labels = cudaify(batch, labels)
                    features += [model.extract_features(batch)]
                    losses += [0]
                    y_true += list(labels.cpu().numpy())
                # Now that features are extracted, we must pass them to the head
                melanoma = torch.cat(melanoma)
                features = torch.cat(features)
                similarities = [model.forward_head(melanoma, torch.stack([features[i]]*melanoma.size(0), dim=0)).mean().item() for i in tqdm(range(features.size(0)), total=features.size(0))]
                # with open('/root/melanoma/src/sims.pkl', 'wb') as f:
                #     pickle.dump(similarities, f)
                # print(len(similarities))
            if self.debug:
                y_true[-1] = 0
                y_true[-2] = 1
            return y_true, similarities, losses            
        else:
            with torch.no_grad():
                for i, data in tqdm(enumerate(self.loader), total=len(self.loader)):
                    if self.debug:
                        if i > 10:
                            y_true[0] = 1
                            y_true[1] = 0
                            break
                    batch, labels = data
                    if self.cuda:
                        batch, labels = cudaify(batch, labels)
                    output = model(batch)
                    if criterion:
                        if 'onehot' in str(criterion).lower():
                            losses += [0]
                        else:
                            losses += [criterion(output, labels).item()]
                    if hasattr(model, 'module'):
                        num_classes = model.module.fc.out_features
                    else:
                        num_classes = model.fc.out_features
                    if num_classes == 2:
                        output = torch.softmax(output, dim=1)[:,1]
                    elif num_classes == 3:
                        output = torch.softmax(output, dim=1)
                    elif num_classes == 1:
                        output = torch.sigmoid(output)
                    output = output.cpu().numpy()
                    y_pred += list(output)
                    if self.labels_available:
                        y_true += list(labels.cpu().numpy())

            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_true)

            return y_true, y_pred, losses



class Evaluator(Predictor):


    def __init__(self,
                 loader,
                 metrics,
                 valid_metric,
                 mode,
                 improve_thresh,
                 prefix,
                 save_checkpoint_dir,
                 save_best,
                 early_stopping=np.inf,
                 thresholds=np.arange(0.05, 1.05, 0.05),
                 cuda=True,
                 debug=False,
                 arc_loader=None):
        
        super(Evaluator, self).__init__(
            loader=loader, 
            cuda=cuda,
            debug=debug,
            arc_loader=arc_loader)

        if type(metrics) is not list: metrics = list(metrics)
        # if type(valid_metric) == list:
        #     for vm in valid_metric: assert vm in metrics
        # else:
        #     assert valid_metric in metrics

        # List of strings corresponding to desired metrics
        # These strings should correspond to function names defined
        # in metrics.py
        self.metrics = metrics
        # valid_metric should be included within metrics
        # This specifies which metric we should track for validation improvement
        self.valid_metric = valid_metric
        # Mode should be one of ['min', 'max']
        # This determines whether a lower (min) or higher (max) 
        # valid_metric is considered to be better
        self.mode = mode
        # This determines by how much the valid_metric needs to improve
        # to be considered an improvement
        self.improve_thresh = improve_thresh
        # Specifies part of the model name
        self.prefix = prefix
        self.save_checkpoint_dir = save_checkpoint_dir
        # save_best = True, overwrite checkpoints if score improves
        # If False, save all checkpoints
        self.save_best = save_best
        self.metrics_file = os.path.join(save_checkpoint_dir, 'metrics.csv')
        if os.path.exists(self.metrics_file): os.system('rm {}'.format(self.metrics_file))
        # How many epochs of no improvement do we wait before stopping training?
        self.early_stopping = early_stopping
        self.stopping = 0
        self.thresholds = thresholds

        self.history = []
        self.epoch = None

        self.reset_best()


    def reset_best(self):
        self.best_model = None
        self.best_score = -np.inf


    def set_logger(self, logger):
        self.logger = logger
        self.print  = self.logger.info


    def validate(self, model, criterion, epoch):
        y_true, y_pred, losses = self.predict(model, criterion, epoch)
        valid_metric = self.calculate_metrics(y_true, y_pred, losses)
        self.save_checkpoint(model, valid_metric, y_true, y_pred)
        return valid_metric


    def generate_metrics_df(self):
        df = pd.concat([pd.DataFrame(d, index=[0]) for d in self.history])
        df.to_csv(self.metrics_file, index=False)


    # Used by Trainer class
    def check_stopping(self):
        return self.stopping >= self.early_stopping


    def check_improvement(self, score):
        # If mode is 'min', make score negative
        # Then, higher score is better (i.e., -0.01 > -0.02)
        multiplier = -1 if self.mode == 'min' else 1
        score = multiplier * score
        improved = score >= (self.best_score + self.improve_thresh)
        if improved:
            self.stopping = 0
            self.best_score = score
        else:
            self.stopping += 1
        return improved


    def save_checkpoint(self, model, valid_metric, y_true, y_pred):
        save_file = '{}_{}_VM-{:.4f}.pth'.format(self.prefix, str(self.epoch).zfill(3), valid_metric).upper()
        save_file = os.path.join(self.save_checkpoint_dir, save_file)
        if self.save_best:
            if self.check_improvement(valid_metric):
                if self.best_model is not None: 
                    os.system('rm {}'.format(self.best_model))
                self.best_model = save_file
                torch.save(model.state_dict(), save_file)
                # Save predictions
                with open(os.path.join(self.save_checkpoint_dir, 'valid_predictions.pkl'), 'wb') as f:
                    pickle.dump({'y_true': y_true, 'y_pred': y_pred}, f)
        else:
            torch.save(model.state_dict(), save_file)
            # Save predictions
            with open(os.path.join(self.save_checkpoint_dir, 'valid_predictions.pkl'), 'wb') as f:
                pickle.dump({'y_true': y_true, 'y_pred': y_pred}, f)
 

    def calculate_metrics(self, y_true, y_pred, losses):
        metrics_dict = {}
        metrics_dict['loss'] = np.mean(losses)
        for metric in self.metrics:
            if metric == 'loss': continue
            metric = eval(metric)
            metrics_dict.update(metric(y_true, y_pred, thresholds=self.thresholds))
        print_results = 'epoch {epoch} // VALIDATION'.format(epoch=self.epoch)
        if type(self.valid_metric) == list:
            valid_metric = np.mean([metrics_dict[vm] for vm in self.valid_metric])
        else:
            valid_metric = metrics_dict[self.valid_metric]
        metrics_dict['vm'] = valid_metric
        max_str_len = np.max([len(k) for k in metrics_dict.keys()])
        for key in metrics_dict.keys():
            self.print('{key} | {value:.5g}'.format(key=key.ljust(max_str_len), value=metrics_dict[key]))
        metrics_dict['epoch'] = int(self.epoch)
        self.history += [metrics_dict]
        self.generate_metrics_df()
        return valid_metric


