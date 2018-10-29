"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

"""
Source code from: https://github.com/bermanmaxim/LovaszSoftmax AND Modify by GMX on 9/30/2018
ATTENTION:
The binary lovasz_hinge expects real-valued scores (positive scores correspond to foreground pixels).
The multiclass lovasz_softmax expect class probabilities (the maximum scoring category is predicted). First use a Softmax layer on the unnormalized scores.
"""


#from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse

class lovasz_base(nn.Module):
    def __init__(self):
        super(lovasz_base, self).__init__()

    def iou_binary(self, preds, labels, EMPTY=1., ignore=None, per_image=True):
        """
        IoU for foreground class
        binary: 1 foreground, 0 background
        """
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        for pred, label in zip(preds, labels):
            intersection = ((label == 1) & (pred == 1)).sum()
            union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
            if not union:
                iou = EMPTY
            else:
                iou = float(intersection) / union
            ious.append(iou)
        iou = mean(ious)  # mean accross images if per_image
        return 100 * iou

    def iou(self, preds, labels, C, EMPTY=1., ignore=None, per_image=False):
        """
        Array of IoU for each (non ignored) class
        """
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        for pred, label in zip(preds, labels):
            iou = []
            for i in range(C):
                if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                    intersection = ((label == i) & (pred == i)).sum()
                    union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                    if not union:
                        iou.append(EMPTY)
                    else:
                        iou.append(float(intersection) / union)
            ious.append(iou)
        ious = map(mean, zip(*ious))  # mean accross images if per_image
        return 100 * np.array(ious)

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        #intersection = gts - gt_sorted.float().cumsum(0)
        #union = gts + (1 - gt_sorted).float().cumsum(0)
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    # --------------------------- HELPER FUNCTIONS ---------------------------

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(np.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

class lovasz_softmax_loss(lovasz_base):
    def __init__(self):
        super(lovasz_softmax_loss, self).__init__()

    # --------------------------- MULTICLASS LOSSES ---------------------------


    def lovasz_softmax(self, probas, labels, only_present=False, per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          only_present: average only on classes present in ground truth
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """

        if per_image:
            loss = self.mean(self.lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                                            only_present=only_present)
                        for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), only_present=only_present)
        return loss

    def lovasz_softmax_flat(self, probas, labels, only_present=False):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          only_present: average only on classes present in ground truth
        """
        C = probas.size(1)
        losses = []
        for c in range(C):
            fg = (labels == c).float()  # foreground for class c
            if only_present and fg.sum() == 0:
                continue
            # errors = (Variable(fg) - probas[:, c]).abs()
            errors = (fg - probas[:, c]).abs()

            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            # losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
            losses.append(torch.dot(errors_sorted, self.lovasz_grad(fg_sorted)))

        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def xloss(self, logits, labels, ignore=None):
        """
        Cross entropy loss
        """

        return F.cross_entropy(logits, Variable(labels), ignore_index=255)

    def forward(self, output, target):
        return self.lovasz_softmax(F.softmax(output), target)

class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

class lovasz_hinge_loss(lovasz_base):
    def __init__(self):
        super(lovasz_hinge_loss, self).__init__()

    # --------------------------- BINARY LOSSES ---------------------------

    #def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
    def lovasz_hinge(self, logits, labels, per_image=False, ignore=None):

        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            loss = self.mean(self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                              for log, lab in zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss


    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        #errors = (1. - logits * Variable(signs))
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        #loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels


    def binary_xloss(self, logits, labels, ignore=None):
        """
        Binary Cross entropy loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          ignore: void class id
        """
        logits, labels = self.flatten_binary_scores(logits, labels, ignore)
        loss = StableBCELoss()(logits, Variable(labels.float()))
        return loss

    def forward(self, output, target):
        return self.lovasz_hinge(F.softmax(output), target)



