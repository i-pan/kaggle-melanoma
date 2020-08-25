import random, torch, os, numpy as np
# From: https://github.com/liaopeiyuan/ml-arsenal-public/blob/master/reproducibility.py

def set_reproducibility(SEED):
    print("Fixing random seed for reproducibility ...")
    os.environ['PYTHONHASHSEED'] = '{}'.format(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print('\tSetting random seed to {} !'.format(SEED))
    print('')
    #
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print('PyTorch environment ...')
    print('\ttorch.__version__              =', torch.__version__)
    print('\ttorch.version.cuda             =', torch.version.cuda)
    print('\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    print('\n')