import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):


    def forward(self, p, t):
        return F.binary_cross_entropy_with_logits(p.float(), t.float())


class CrossEntropyLoss(nn.CrossEntropyLoss):


    def forward(self, p, t):
        t = t.view(-1)
        if self.weight:
            return F.cross_entropy(p.float(), t.long(), weight=self.weight.float().to(t.device))
        else:
            return F.cross_entropy(p.float(), t.long())


class OneHotCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class CrossEntropyJSD(nn.Module):

    def __init__(self, lam=12):
        self.lam = lam

    def forward(self, p, t):
        if isinstance(x, list):
            assert len(p) == 3
            loss = F.cross_entropy(p[0].float(), t.long())
            p_clean, p_aug1, p_aug2 = F.softmax(
              p[0], dim=1), F.softmax(
                  p[1], dim=1), F.softmax(
                      p[2], dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += self.lam * \
                    (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                     F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                     F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            return loss
        else:
            return F.cross_entropy(p.float(), t.long())


class LabelSmoothing(nn.Module):
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return F.cross_entropy(x, target.long())


class MixCE(nn.Module):


    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = F.cross_entropy(p.float(), t['y_true1'].long(), reduction='none')
        loss2 = F.cross_entropy(p.float(), t['y_true2'].long(), reduction='none')
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()


    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return F.cross_entropy(p.float(), t.long())



class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


    def forward(self, logits, labels):
        labels = F.one_hot(labels.long(), logits.size(1)).float().to(labels.device)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss


class ArcCELoss(nn.Module):

    def __init__(self, lam=0.2, s=30.0, m=0.5):
        super().__init__()
        self.arcface = ArcFaceLoss(s=s, m=m)
        self.lam = lam

    def forward(self, p, t):
        if isinstance(p, tuple):
            arcloss = self.arcface(p[1], t)
            celoss = F.cross_entropy(p[0], t.long())
            return celoss+self.lam*arcloss, celoss, arcloss
        else:
            return F.cross_entropy(p, t.long())


class OHEMCELoss(nn.Module):


    def __init__(self, total_steps, lowest_rate=1./8):
        super().__init__()
        self.total_steps = total_steps
        self.lowest_rate = lowest_rate
        self.steps = 0

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def calculate_rate(self):
        pct = float(self.steps) / self.total_steps
        self.steps += 1
        self.current_rate = self._annealing_cos(start=1.0, end=self.lowest_rate, pct=pct)

    def forward(self, y_pred, y_true):
        loss = F.cross_entropy(y_pred, y_true, reduction='none')
        B = y_pred.size(0)
        self.calculate_rate()
        loss, _ = loss.topk(k=int(self.current_rate * B), dim=0)
        return loss.mean()


class FocalLoss(nn.Module):
    #
    def __init__(self, alpha=1, gamma=2, logits=True, reduction=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
    #    
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)