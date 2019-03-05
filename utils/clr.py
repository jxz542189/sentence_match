'''
2019.03.05 13:24
@author jxz
@reference [1] Smith, Leslie N. “Cyclical learning rates for training neural networks.” Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017
'''
import numpy as np


def cyclical_learning_rate(batch_step,
                           step_size,
                           base_lr=0.001,
                           max_lr=0.006,
                           mode='triangular',
                           gamma=0.999995):
    cycle = np.floor(1 + batch_step / (2. * step_size))
    x = np.abs(batch_step / float(step_size) - 2 * cycle + 1)

    lr_delta = (max_lr - base_lr) * np.maximum(0, (1 - x))

    if mode == 'triangular':
        pass
    elif mode == 'triangular2':
        lr_delta = lr_delta * 1 / (2. ** (cycle - 1))
    elif mode == 'exp_range':
        lr_delta = lr_delta * (gamma ** (batch_step))
    else:
        raise ValueError('mode must be "triangular", "triangular2", or "exp_range"')

    lr = base_lr + lr_delta
    return lr


