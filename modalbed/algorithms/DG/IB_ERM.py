import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from modalbed import networks
from modalbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj, Nonparametric,
            LARS,  SupConLossLambda
    )

from .. import Algorithm, ERM

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][1].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = []
        for x, y in minibatches:
            all_x.extend(x)
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + len(x)]
            logits = all_logits[all_logits_idx:all_logits_idx + len(x)]
            all_logits_idx += len(x)
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}
