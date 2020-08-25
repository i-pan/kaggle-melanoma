import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime
import time
import logging
import pandas as pd
import numpy as np
import os

from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

global AMP_AVAIL
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAIL = True
except ImportError:
    print('Automatic mixed precision not available !')
    AMP_AVAIL = False

from tqdm import tqdm
from torch.utils import data

from ..data.utils import cudaify, _isnone
from ..models import *
from .fmix import fmix_apply
from .cutmix import cutmix_apply
from .cutmixup import cutmixup_apply


class TimeTracker(object):


    def __init__(self, length=100):
        self.length = length
        self.load_time = []
        self.step_time = []


    def set_time(self, t):
        self.load_time.append(t[0])
        self.step_time.append(t[1])


    def get_time(self):
        return (np.mean(self.load_time[-int(self.length):]),
                np.mean(self.step_time[-int(self.length):]))


class LossTracker(object): 


    def __init__(self, num_moving_average=1000): 
        self.losses  = defaultdict(list)
        self.history = defaultdict(list)
        self.avg = num_moving_average


    def set_loss(self, minibatch_loss):
        for k,v in minibatch_loss.items():
            self.losses[k].append(v) 


    def get_loss(self): 
        for k,v in self.losses.items():
            self.history[k].append(np.mean(v[-self.avg:]))
        return {k:v[-1] for k,v in self.history.items()}


    def reset(self): 
        self.losses = defaultdict(list)


    def get_history(self): 
        return self.history


class Step(object):


    def __init__(self, loader):
        super(Step, self).__init__()

        self.loss_tracker = LossTracker(num_moving_average=1000)
        self.time_tracker = TimeTracker(length=100)
        if type(loader) == tuple:
            self.full_loader = loader[1]
            self.loader = loader[0]
        else:
            self.full_loader = None
            self.loader = loader

        self.generator = self._data_generator()
        self.fitted = False

    # Wrap data loader in a generator ...
    def _data_generator(self):
        while 1:
            for data in self.loader:
                yield data


    # Move the model forward ...
    def _fetch_data(self): 
        batch, labels = next(self.generator)

        if self.cuda:
            batch, labels = cudaify(batch, labels)

        mixaug = []
        if not _isnone(self.cutmix): mixaug.append('cutmix')
        if not _isnone(self.mixup):  mixaug.append('mixup')

        if len(mixaug) == 0: 
            return (batch, labels)
        
        mix = np.random.choice(mixaug)

        alpha = getattr(self, mix)
        if mix == 'cutmix':
            batch, index, lam = cutmix_apply(batch, alpha)

        elif mix == 'mixup':
            lam = np.random.beta(alpha, alpha, batch.size(0))
            lam = np.max((lam, 1-lam), axis=0)
            index = torch.randperm(batch.size(0))
            lam = torch.Tensor(lam).cuda()
            for _ in range(batch.ndim-1):
                lam = lam.unsqueeze(-1)
            batch = lam * batch + (1-lam) * batch[index]

        labels_dict = {
            'y_true1': labels,
            'y_true2': labels[index],
            'lam': lam
        }

        return (batch, labels_dict)


    # With closure
    def _step(self):
        data_start = time.time()
        batch, labels = self._fetch_data()
        data_time = time.time() - data_start

        def closure():
            self.optimizer.zero_grad()
            output = self.model(batch)
            if self.loss_masking:
                loss = self.criterion(output, labels, batch['mask'])
            else:
                loss = self.criterion(output, labels)
            loss.backward() 
            self.loss_tracker.set_loss(loss.item())
            return loss 

        step_start = time.time()
        loss = self.optimizer.step(closure=closure)
        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))


    def _loss(self, output, labels):
        if 'dualloss' in str(self.criterion).lower():
            loss, mse_loss, bce_loss = self.criterion(output, labels)
            tracked_loss = {
                'loss': loss.item(), 
                'mse_loss': mse_loss.item(),
                'bce_loss': bce_loss.item()
            }
        elif 'arcceloss' in str(self.criterion).lower():
            loss, celoss, arcloss = self.criterion(output, labels)
            tracked_loss = {
                'loss': loss.item(),
                'ce': celoss.item(),
                'arc': arcloss.item()
            }
        else:
            loss = self.criterion(output, labels)
            tracked_loss = {'loss': loss.item()}
        return loss, tracked_loss


    # AMP currently does not support closure
    def _amp_step(self):
        data_start = time.time()
        batch, labels = self._fetch_data()
        data_time = time.time() - data_start

        step_start = time.time() 

        self.optimizer.zero_grad()
        with autocast():
            if isinstance(batch, list):
                # JSD 
                assert len(batch) == 3
                batch_all = torch.cat(batch, 0)
                logits_all = self.model(batch_all)
                output = torch.split(logits_all, batch[0].size(0))
            else:
                output = self.model(batch)
            loss, tracked_loss = self._loss(output, labels)
        self.scaler.scale(loss).backward()
        self.loss_tracker.set_loss(tracked_loss)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))


    def _amp_accumulate_step(self):
        data_start = time.time()
        batch, labels = self._fetch_data()
        data_time = time.time() - data_start
        if type(batch) == dict:
            batch_size = len(batch[[*batch][0]])
        else:
            batch_size = batch.size()[0]
        assert batch_size % self.gradient_accumulation == 0, f'Batch size <{batch_size}> must be multiple of gradient accumulation <{self.gradient_accumulation}>'
        splits = torch.split(torch.arange(batch_size), int(batch_size/self.gradient_accumulation))

        step_start = time.time() 

        self.optimizer.zero_grad()
        tracker_loss = 0.
        with autocast():
            for i in range(int(self.gradient_accumulation)):
                if isinstance(batch, dict):
                    output = self.model({k : v[splits[i]] for k,v in batch.items()})
                else:
                    output = self.model(batch[splits[i]])
                if self.mixup or self.cutmix:
                    loss = self.criterion(output, 
                        {k : v[splits[i]] for k,v in labels.items()})
                else:
                    loss = self.criterion(output, labels[splits[i]])
                loss = loss / self.gradient_accumulation
                tracker_loss += loss.item()
                self.scaler.scale(loss).backward()

        self.loss_tracker.set_loss({'loss': tracker_loss})
        self.scaler.step(self.optimizer)
        self.scaler.update()

        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))


    def _accumulate_step(self):

        data_start = time.time()
        batch, labels = self._fetch_data()
        data_time = time.time() - data_start 
        batch_size = batch.size()[0]
        splits = torch.split(torch.arange(batch_size), int(batch_size/self.gradient_accumulation))

        def closure(): 
            self.optimizer.zero_grad()
            tracker_loss = 0.
            for i in range(int(self.gradient_accumulation)):
                output = self.model(batch[splits[i]])
                if self.mixup or self.cutmix:
                    loss = self.criterion(output, 
                        self._separate_batch(labels, splits[i]))                    
                else:
                    loss = self.criterion(output, 
                        {k : v[splits[i]] for k,v in labels.items()})
                tracker_loss += loss.item()
                if i < (self.gradient_accumulation - 1):
                    retain = True
                else:
                    retain = False
                (loss / self.gradient_accumulation).backward()#retain_graph=retain) 
            self.loss_tracker.set_loss(tracker_loss / self.gradient_accumulation)

        step_start = time.time()
        loss = self.optimizer.step(closure=closure)
        step_time = time.time() - step_start

        self.time_tracker.set_time((data_time, step_time))        


    def train_step(self):
        if self.amp:
            self._amp_accumulate_step() if self.gradient_accumulation > 1 else self._amp_step()
        else:
            self._accumulate_step() if self.gradient_accumulation > 1 else self._step()

        if not self.fitted:
            print('Batch size successful !')
            self.fitted = True

        if self.grid_mask:
            # grid_mask is a float in (0, 1)
            # anneal from p_start to p_end in grid_mask*total_steps
            self.loader.dataset.transform.set_p(int(self.grid_mask * self.total_steps))


class Trainer(Step):


    def __init__(self, 
        loader,
        model, 
        optimizer,
        schedule, 
        criterion, 
        evaluator,
        logger):

        super(Trainer, self).__init__(loader=loader)

        self.model = model 
        self.optimizer = optimizer
        self.scheduler = schedule
        self.criterion = criterion
        self.evaluator = evaluator

        self.logger = logger
        self.print = self.logger.info
        self.evaluator.set_logger(self.logger)


    def check_end_train(self): 
        return self.current_epoch >= self.num_epochs


    def check_end_epoch(self):
        return (self.steps % self.steps_per_epoch) == 0 and (self.steps > 0)


    def check_validation(self):
        # We add 1 to current_epoch when checking whether to validate
        # because epochs are 0-indexed. E.g., if validate_interval is 2,
        # we should validate after epoch 1. We need to add 1 so the modulo
        # returns 0
        return self.check_end_epoch() and self.steps > 0 and ((self.current_epoch + 1) % self.validate_interval) == 0


    def scheduler_step(self):
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step(self.current_epoch + self.steps * 1./self.steps_per_epoch)
        else:
            self.scheduler.step()


    def print_progress(self):
        train_loss = self.loss_tracker.get_loss()
        loss_statement = ''
        for k,v in train_loss.items():
            loss_statement += '{}={:.4f} '.format(k,v)
        learning_rates = np.unique([_['lr'] for _ in self.optimizer.param_groups])
        lr_statement = 'lr='
        for lr in learning_rates:
            lr_statement += '{:.1e}/'.format(lr)
        lr_statement = lr_statement[:-1]

        self.print('epoch {epoch}, batch {batch}/{steps_per_epoch}: {loss_statement}(data: {load_time:.3f}s/batch, step: {step_time:.3f}s/batch, {lr_statement})'
                .format(epoch=str(self.current_epoch).zfill(len(str(self.num_epochs))), \
                        batch=str(self.steps).zfill(len(str(self.steps_per_epoch))), \
                        steps_per_epoch=self.steps_per_epoch, \
                        loss_statement=loss_statement, \
                        load_time=self.time_tracker.get_time()[0],
                        step_time=self.time_tracker.get_time()[1],
                        lr_statement=lr_statement))


    def init_training(self, 
                      gradient_accumulation, 
                      num_epochs,
                      steps_per_epoch,
                      validate_interval,
                      mixup,
                      cuda):

        self.gradient_accumulation = float(gradient_accumulation)
        self.num_epochs = num_epochs
        self.steps_per_epoch = len(self.loader) if steps_per_epoch == 0 else steps_per_epoch
        self.validate_interval = validate_interval
        self.mixup = mixup
        self.cuda = True

        self.total_steps = self.steps_per_epoch * self.num_epochs
        self.steps = 0 
        self.current_epoch = 0

        self.optimizer.zero_grad()


    @staticmethod
    def run_ohem(model, full_loader, loader, cuda=True):
        benign_preds = []
        for i, dat in tqdm(enumerate(full_loader), total=len(full_loader)):
            batch, labels = dat
            if cuda:
                batch, labels = cudaify(batch, labels)
            output = model(batch)
            output = torch.softmax(output, dim=1)
            benign_preds += [output.detach().cpu().numpy()]
        benign_preds = np.vstack(benign_preds)
        benign_preds = benign_preds[:,1]
        # Normalize
        benign_preds /= np.sum(benign_preds)
        loader.sampler.probas = {full_loader.dataset.imgfiles[i] : benign_preds[i] for i in range(len(benign_preds))}
        return loader


    def train(self, 
              gradient_accumulation,
              num_epochs, 
              steps_per_epoch, 
              validate_interval,
              verbosity=100,
              loss_masking=True,
              grid_mask=None,
              mixup=None,
              cutmix=None,
              minmix=None,
              fmix=None,
              cutmix_single=False,
              cutmix_target=False,
              cutmix_margin=0,
              cutminmix=False,
              cutmixup=None,
              cuda=True,
              amp=False,
              fgm=False,
              ohem=False): 
        # Epochs are 0-indexed
        self.init_training(gradient_accumulation, num_epochs, steps_per_epoch, validate_interval, mixup, cuda)
        self.grid_mask = grid_mask
        self.loss_masking = loss_masking
        self.cutmix = cutmix
        self.minmix = minmix
        self.cutmixup = cutmixup
        self.cutminmix = cutminmix
        self.cutmix_single = cutmix_single
        self.cutmix_target = cutmix_target
        self.cutmix_margin = cutmix_margin
        self.fmix = fmix
        self.amp = amp
        self.fgm = None
        if fgm: self.fgm = FGM(self.model)
        self.ohem = ohem
        if amp: 
            assert AMP_AVAIL, 'Automatic mixed precision training not available, `amp` must be `False`'
            self.scaler = GradScaler()
        start_time = datetime.datetime.now()
        if type(self.model) == nn.DataParallel:
            self.model.module._autocast = True
        else:
            self.model._autocast = True
        while 1: 
            self.train_step()
            self.steps += 1
            if self.scheduler.update == 'on_batch':
                 self.scheduler_step()
            # Check- print training progress
            if self.steps % verbosity == 0 and self.steps > 0:
                self.print_progress()
            # Check- run validation
            if self.check_validation():
                self.print('VALIDATING ...')
                validation_start_time = datetime.datetime.now()
                # Start validation
                self.model.eval()
                valid_metric = self.evaluator.validate(self.model, 
                    self.criterion, 
                    str(self.current_epoch).zfill(len(str(self.num_epochs))))
                if self.scheduler.update == 'on_valid':
                    self.scheduler.step(valid_metric)
                # End validation
                self.model.train()
                self.print('Validation took {} !'.format(datetime.datetime.now() - validation_start_time))
            # Check- end of epoch
            if self.check_end_epoch():
                if self.scheduler.update == 'on_epoch':
                    self.scheduler.step()
                self.current_epoch += 1
                self.steps = 0
                # RESET BEST MODEL IF USING COSINEANNEALINGWARMRESTARTS
                if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    if self.current_epoch % self.scheduler.T_0 == 0:
                        self.evaluator.reset_best()
                        # If specified- run inference on all benign cases from training data
                        if self.ohem:
                            self.print('Running inference on benign lesions ...')
                            self.model.eval()
                            self.loader = self.run_ohem(self.model, self.full_loader, self.loader)
                            self.model.train()
            #
            if self.evaluator.check_stopping(): 
                # Make sure to set number of epochs to max epochs
                # Remember, epochs are 0-indexed and we added 1 already
                # So, this should work (e.g., epoch 99 would now be epoch 100,
                # thus training would stop after epoch 99 if num_epochs = 100)
                self.current_epoch = num_epochs
            if self.check_end_train():
                # Break the while loop
                break
        self.print('TRAINING : END') 
        self.print('Training took {}\n'.format(datetime.datetime.now() - start_time))








