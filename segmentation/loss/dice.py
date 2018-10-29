#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

class DiceLoss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    """
    def __init__(self, size_average=True, reduce=True):
        super(DiceLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        """
        input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input
          """
        input = F.softmax(input)
        probs = input[:,1,:,:]


        # print(probs)
        probs = (probs.unsqueeze(1))
        # print(probs)
        target = (target.unsqueeze(1)).float()
        # assert probs.size() == target.size(), "Input sizes must be equal."
        # assert probs.dim() == 4, "Input must be a 4D Tensor."
        # uniques=np.unique(target.data.cpu().numpy())
        # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

        num=probs*target#b,c,h,w--p*g
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)
        

        den1=probs*probs#--p^2
        den1=torch.sum(den1,dim=3)#b,c,h
        den1=torch.sum(den1,dim=2)
        

        den2=target*target#--g^2
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c
        

        dice=2*(num/(den1+den2))
        # print(num)
        # print(den1)
        # print(den2)
        dice_eso=dice[:,:]#we ignore bg dice val, and take the fg
        # print(dice_eso)
        dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
        # print(dice_total)
        return dice_total


