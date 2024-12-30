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

class OGM(Algorithm):
    """
    Balanced Multimodal Learning via On-the-fly Gradient Modulation.
    CVPR 2022
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(OGM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        
        self.cls_f = networks.Classifier(
            self.featurizer.n_outputs * num_domains,
            num_classes,
            self.hparams['nonlinear_classifier'])
        
        self.uni_fc = nn.ModuleList([nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs) for _ in range(num_domains)])
        
        featurizer_params = list(self.featurizer.parameters())
        cls_f_params = list(self.cls_f.parameters())

        all_params = featurizer_params + cls_f_params

        self.optimizer = torch.optim.SGD(all_params, lr=self.hparams["lr"],
                    momentum=self.hparams['momentum'],
                    weight_decay=self.hparams['wc'])
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.hparams['patience'], self.hparams['gamma'])        
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
        modal_features = [self.uni_fc[i](features[torch.arange(num_sample)+i]) for i in range(self.num_domains)]
        label = all_y[torch.arange(num_sample)*self.num_domains]

        logits_b = self.cls_f(torch.cat(modal_features, dim=1))

        loss = self.criterion(logits_b, label).mean()
        
        losses = []
        weight_size = self.cls_f.weight.size(1)
        unit_weight_size = weight_size//self.num_domains
        outs = []
        for idx, i in enumerate(modal_features):
            logits_i = torch.mm(
                i, self.cls_f.weight[:, int(unit_weight_size * idx) : int(unit_weight_size * (idx+1))].T
            ) + self.cls_f.bias
            losses.append(self.criterion(logits_i, label).mean())
            outs.append(logits_i)
            
        loss.backward()
        
        # modulation 
        scores = [
            sum([torch.softmax(out_, dim=1)[i][label[i]] for i in range(out_.size(0))])
            for out_ in outs
        ]
        
        ratios = [score / sum(scores) for score in scores]
        
        coeffs = []
        for ratio in ratios:
            if ratio > 1:
                coeffs.append(1 - torch.tanh(self.hparams["alpha"]  * torch.relu(ratio)))
            else:
                coeffs.append(1)
        
        for name, parms in self.uni_fc.named_parameters():
            layer = str(name).split('.')[0]
            idx = eval(layer)
            parms.grad = parms.grad * coeffs[idx]
        
        self.optimizer.step()
        
        return {"loss": loss.item()}
        
    def predict(self, x):
        features = self.featurizer(x) # batch x dim
        modal_features = [self.uni_fc[i](features) for i in range(self.num_domains)] # copy features to each domain
        logits = self.cls_f(torch.cat(modal_features, dim=1))
        return logits
    