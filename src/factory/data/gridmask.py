import numpy as np

from scipy.ndimage.interpolation import rotate


class GridMask(object):


    def __init__(self, 
                 k, 
                 D, 
                 theta=360, 
                 mode=['topleft', 'botright'],
                 always_apply=True,
                 p_start=0,
                 p_end=0.8,
                 policy='linear'):
        self.k = k
        self.D = D
        self.theta = theta
        self.mode = mode
        self.always_apply = always_apply
        self.p_start = p_start
        self.p_end = p_end
        self.policy = 'linear'
        self.steps = 0
        self.p = p_start

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def set_p(self, total_steps):
        self.steps += 1
        pct = min(1.0, self.steps / float(total_steps))
        if self.policy == 'linear':
            self.p = self._annealing_linear(self.p_start, self.p_end, pct)
        elif self.policy == 'cosine':
            self.p = self._annealing_cos(self.p_start, self.p_end, pct)

    def apply(self, image):
        # Sample k if range is provided
        if isinstance(self.k, (tuple,list)):
            k = np.random.uniform(self.k[0], self.k[1])
        else:
            k = self.k        
        # Sample D if range is provided
        if isinstance(self.D, (tuple,list)):
            D = np.random.uniform(self.D[0], self.D[1])
        else:
            D = self.D
        if D <= 1: 
            D = D * np.min(image.shape[:2])
        D = int(D)
        dx = np.random.randint(D)
        dy = np.random.randint(D)
        dx = dy = 0
        rm = int(D * (1 - (1 - np.sqrt(1 - k))))
        _mode = np.random.choice(self.mode)
        mask = np.ones(image.shape[:2])
        for row in range(dx, mask.shape[0], D):
            for col in range(dy, mask.shape[1], D):
                if _mode == 'topleft':
                    row0, row1 = row, row+rm
                    col0, col1 = col, col+rm
                elif _mode == 'botright':
                    row0, row1 = row+(D-rm), row+D
                    col0, col1 = col+(D-rm), col+D
                mask[row0:row1+1, col0:col1+1] = 0
        if self.theta > 0:
            mask = rotate(mask, angle=np.random.uniform(-self.theta,self.theta), reshape=False, order=1, prefilter=False, mode='constant', cval=1)
        masked_image = image * np.expand_dims(mask, axis=-1)
        return {'image': masked_image}


    def __call__(self, image):
        if np.random.binomial(1, self.p) or self.always_apply:
            return self.apply(image)
        else:
            return {'image': image}
