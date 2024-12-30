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

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
def Alignment(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * (p + q)
    kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')
    js_score = 0.5 * (kl_p_m + kl_q_m)
    return js_score

def getAlpha_heritic(epoch, a = 0.3):
    alpha = np.exp(-a * epoch)
    alpha = np.clip(alpha, 0.05, 0.95)
    return [alpha,  1.0 -alpha]
    
class LFM(Algorithm):
    """
    Facilitating Multimodal Classification via Dynamically Learning Modality Gap, LFM
    NIPS 24
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LFM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        
        self.cls_f = networks.Classifier(
            self.featurizer.n_outputs * num_domains,
            num_classes,
            self.hparams['nonlinear_classifier'])
        
        featurizer_params = list(self.featurizer.parameters())
        cls_f_params = list(self.cls_f.parameters())

        all_params = featurizer_params + cls_f_params

        self.optimizer = torch.optim.SGD(all_params, lr=self.hparams["lr"],
                    momentum=self.hparams['momentum'],
                    weight_decay=self.hparams['wc'])
        
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

        loss_cls = self.criterion(logits_b, label).mean()
        
        alignments = [
                Alignment(modal_features[i], modal_features[j])
                for i in range(self.num_domains) 
                for j in range(i+1, self.num_domains)
                ]
        if len(alignments) > 0:
            loss_alignment = torch.mean(
                torch.stack(alignments)
                )
        else:
            loss_alignment = 0
        
        cls_k = getAlpha_heritic(epoch) 
        loss = cls_k[0] * loss_alignment + cls_k[1] * loss_cls
        
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item(), "cls_k": cls_k[0]}
        
    def predict(self, x):
        features = self.featurizer(x) # batch x dim
        modal_features = [features for _ in range(self.num_domains)] # copy features to each domain
        logits = self.cls_f(torch.cat(modal_features, dim=1))
        return logits
    