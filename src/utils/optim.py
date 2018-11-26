import itertools
import torch

class Optimizer(object):

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optim, max_grad_norm=0):
        self.optimizer = optim
        self.max_grad_norm = max_grad_norm

    def step(self):
        if self.max_grad_norm>0:
            params = itertools.chain.from_iterable(\
                [group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm(params, self.max_grad_norm)
        self.optimizer.step()

