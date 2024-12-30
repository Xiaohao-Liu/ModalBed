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

from .. import Algorithm
    
class Concat(Algorithm):
    """
    Facilitating Multimodal Classification via Dynamically Learning Modality Gap, LFM
    NIPS 24
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Concat, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        
        self.cls_f = networks.Classifier(
            self.featurizer.n_outputs * num_domains,
            num_classes,
            self.hparams['nonlinear_classifier'])
        
        featurizer_params = list(self.featurizer.parameters())
        cls_f_params = list(self.cls_f.parameters())

        all_params = featurizer_params + cls_f_params

        # self.optimizer = torch.optim.SGD(all_params, lr=self.hparams["lr"],
        #             momentum=self.hparams['momentum'],
        #             weight_decay=self.hparams['wc'])
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['wc']
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.hparams['patience'], 0.1)        
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        
    def update(self, minibatches, unlabeled=None, epoch = 0): 
        all_x = []
        all_y = []
        for x, y in minibatches:
            all_x.extend(x)
            all_y.append(y)
        all_y = torch.cat(all_y)
        
        self.optimizer.zero_grad()
        
        features = self.featurizer(all_x) # batch x dim
        
        num_sample = features.size(0) // self.num_domains
        modal_features = [features[torch.arange(num_sample)+i] for i in range(self.num_domains)]
        label = all_y[torch.arange(num_sample)*self.num_domains]
        
        logits_b = self.cls_f(torch.cat(modal_features, dim=1))

        loss = self.criterion(logits_b, label).mean()
        
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}
        
    def predict(self, x):
        features = self.featurizer(x) # batch x dim
        modal_features = [features for _ in range(self.num_domains)] # copy features to each domain
        logits = self.cls_f(torch.cat(modal_features, dim=1))
    
        return logits
    