from __future__ import print_function
import math
import torch.nn as nn
import numpy as np
import pdb

class Loss(object):
    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        self.acc_loss = 0
        self.norm_term = 0

    def reset(self):
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()
        
    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()

class NLLLoss(Loss):
    
    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0
        #weight = weight.cuda()
        super(NLLLoss, self).__init__(
              self._NAME,
              nn.NLLLoss(weight=weight, size_average=size_average))
    
    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()#.data[0]
        if self.size_average:
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        #print (outputs.size(), target.size())
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

