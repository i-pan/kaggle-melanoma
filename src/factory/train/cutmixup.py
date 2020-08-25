import numpy as np
import torch


def rand_bbox_vector(size, lam):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(np.int)
    cut_h = (H * cut_rat).astype(np.int)
    # uniform
    cx = np.random.randint(0, W, B)
    cy = np.random.randint(0, H, B)
    #
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmixup_apply(batch, alpha):
    lam = np.random.beta(alpha, alpha, batch.size(0))
    lam = np.max((lam, 1.-lam), axis=0)
    index = torch.randperm(batch.size(0)) 
    x1, y1, x2, y2 = rand_bbox_vector(batch.size(), lam)
    for b in range(batch.size(0)):
        if np.random.binomial(1, 0.5):
            # Cutmix
            batch[b, :, x1[b]:x2[b], y1[b]:y2[b]] = batch[index[b], :, x1[b]:x2[b], y1[b]:y2[b]]
            lam[b] = 1. - ((x2[b] - x1[b]) * (y2[b] - y1[b]) / float((batch.size()[-1] * batch.size()[-2])))
        else:
            # Mixup
            batch[b] = lam[b]*batch[b] + (1.-lam[b])*batch[index[b]]
    return batch, index, torch.Tensor(lam).cuda()
