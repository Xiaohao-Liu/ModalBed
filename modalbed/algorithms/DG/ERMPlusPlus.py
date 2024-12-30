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

class ERMPlusPlus(Algorithm,ErmPlusPlusMovingAvg):
    """
    Empirical Risk Minimization with improvements (ERM++)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self,input_shape, num_classes, num_domains,hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        if self.hparams["lars"]:
            self.optimizer = LARS(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        linear_parameters = []
        for n, p in self.network[1].named_parameters():
            linear_parameters.append(p)

        if self.hparams["lars"]:
            self.linear_optimizer = LARS(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        else:
            self.linear_optimizer = torch.optim.Adam(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )
        self.lr_schedule = []
        self.lr_schedule_changes = 0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience = 1)
        ErmPlusPlusMovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):

        if self.global_iter > self.hparams["linear_steps"]:
            selected_optimizer = self.optimizer
        else:
            selected_optimizer = self.linear_optimizer

        all_x = []
        all_y = []
        for x, y in minibatches:
            all_x.extend(x)
            all_y.append(y)
        all_y = torch.cat(all_y)
        loss = F.cross_entropy(self.network(all_x), all_y)

        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        self.update_sma()
        if not self.hparams["freeze_bn"]:
            self.network_sma.train()
            self.network_sma(all_x)

        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)

    def set_lr(self, eval_loaders_iid=None, schedule=None,device=None):
        with torch.no_grad():
             if self.global_iter > self.hparams["linear_steps"]:
                 if schedule is None:
                     self.network_sma.eval()
                     val_losses = []
                     for loader in eval_loaders_iid:
                         loss = 0.0
                         for x, y in loader:
                             x = x.to(device)
                             y = y.to(device)
                             loss += F.cross_entropy(self.network_sma(x),y)
                         val_losses.append(loss / len(loader ))
                     val_loss = torch.mean(torch.stack(val_losses))
                     self.scheduler.step(val_loss)
                     self.lr_schedule.append(self.scheduler._last_lr)
                     if len(self.lr_schedule) > 1:
                         if self.lr_schedule[-1] !=  self.lr_schedule[-2]:
                            self.lr_schedule_changes += 1
                     if self.lr_schedule_changes == 3:
                         self.lr_schedule[-1] = [0.0]
                     return self.lr_schedule
                 else:
                     self.optimizer.param_groups[0]['lr'] = (torch.Tensor(schedule[0]).requires_grad_(False))[0]
                     schedule = schedule[1:]
             return schedule
