import numpy as np
import torch


def rand_bbox_vector(size, lam, margin=0):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(np.int)
    cut_h = (H * cut_rat).astype(np.int)
    # uniform
    if margin < 1 and margin > 0:
        w_margin = margin*W
        h_margin = margin*H
    else:
        w_margin = margin
        h_margin = margin

    cx = np.random.randint(0+w_margin, W-w_margin, B)
    cy = np.random.randint(0+h_margin, H-h_margin, B)
    #
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def rand_bbox_target(batch, size, lam):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(np.int)
    cut_h = (H * cut_rat).astype(np.int)
    # find points within the grapheme
    targets = [torch.stack(torch.where(batch[i,0] < torch.mean(batch[i,0])*0.5)) for i in range(B)]
    # uniform
    cxy = [t[:,np.random.choice(t.size(1))] if t.size(1) > 0 else torch.Tensor([np.random.randint(0, W), np.random.randint(0, H)]) for t in targets]
    cx = np.asarray([center[0] for center in cxy])
    # cx[cx-cut_w // 2 < 0] = cut_w[cx-cut_w // 2 < 0] // 2
    # cx[cx+cut_w // 2 > W] = cut_w[cx+cut_w // 2 > W] // 2
    cy = np.asarray([center[1] for center in cxy])
    # cy[cy-cut_h // 2 < 0] = cut_h[cy-cut_h // 2 < 0] // 2
    # cy[cy+cut_h // 2 > H] = cut_h[cy+cut_h // 2 > H] // 2
    #
    bbx1 = np.clip(cx - cut_w // 2, 0, W).astype('int')
    bby1 = np.clip(cy - cut_h // 2, 0, H).astype('int')
    bbx2 = np.clip(cx + cut_w // 2, 0, W).astype('int')
    bby2 = np.clip(cy + cut_h // 2, 0, H).astype('int')
    return bbx1, bby1, bbx2, bby2


def rand_bbox_single(size, lam, margin=0):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(margin, W-margin)
    cy = np.random.randint(margin, H-margin)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_apply(batch, alpha, single=False, target=False, margin=0, cutminmix=False):
    rand_bbox = rand_bbox_single if single else rand_bbox_vector
    batch_size = batch[0].size(0) if type(batch) == tuple else batch.size(0) 
    assert single + target < 2
    if single:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = np.max((lam, 1.-lam), axis=0)
    index = torch.randperm(batch_size)
    if target:
        x1, y1, x2, y2 = rand_bbox_target(batch, batch.size(), lam)
    else:
        x1, y1, x2, y2 = rand_bbox(batch.size(), lam, margin)
    if single:
        batch[index, :, x1:x2, y1:y2] = batch[index, :, x1:x2, y1:y2]
    else:
        for b in range(batch.size(0)):
            if cutminmix:
                batch[b, :, x1[b]:x2[b], y1[b]:y2[b]] = torch.min(batch[b, :, x1[b]:x2[b], y1[b]:y2[b]], batch[index[b], :, x1[b]:x2[b], y1[b]:y2[b]])
            else:
                batch[b, :, x1[b]:x2[b], y1[b]:y2[b]] = batch[index[b], :, x1[b]:x2[b], y1[b]:y2[b]]
    lam = 1. - ((x2 - x1) * (y2 - y1) / float((batch.size()[-1] * batch.size()[-2])))
    if not single:
        lam = torch.Tensor(lam).cuda()
    return batch, index, lam


if __name__ == 'main':

    import torch, numpy as np
    import matplotlib.pyplot as plt
    import cv2, glob

    imgs = glob.glob('*.png')
    images = np.asarray([cv2.imread(i) for i in imgs])
    #images = np.asarray([cv2.resize(i, (224,224)) for i in images])
    thresholded = [(images[i,...,0] < np.mean(images[i,...,0])*0.75) for i in range(len(images))]
    # for t in thresholded:
    #     plt.imshow(t); plt.show()
    batch = images.transpose(0,3,1,2)
    batch = torch.from_numpy(batch).float()

    lam = np.random.beta(1.0,1.0,len(batch))
    x1, y1, x2, y2 = rand_bbox_target(batch, batch.size(), lam)
    #x1, y1, x2, y2 = rand_bbox_vector(batch.size(), lam)
    index = torch.randperm(batch.size(0))

    for b in range(batch.size(0)):
        batch[b, :, x1[b]:x2[b], y1[b]:y2[b]] = 255 - batch[index[b], :, x1[b]:x2[b], y1[b]:y2[b]]

    for bind, b in enumerate(batch):
        print(lam[bind])
        plt.subplot(1,2,1)
        plt.imshow(b.numpy().transpose(1,2,0).astype('uint8')) 
        plt.subplot(1,2,2)
        plt.imshow(images[bind])
        plt.show()



