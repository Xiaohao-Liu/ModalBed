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

class URM(ERM):
    """
    Implementation of Uniform Risk Minimization, as seen in Uniformly Distributed Feature Representations for
    Fair and Robust Learning. TMLR 2024 (https://openreview.net/forum?id=PgLbS5yp8n)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        ERM.__init__(self, input_shape, num_classes, num_domains, hparams)

        # setup discriminator model for URM adversarial training
        self._setup_adversarial_net()

        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def _modify_generator_output(self):
        print('--> Modifying encoder output:', self.hparams['urm_generator_output'])
        
        from modalbed.lib import wide_resnet
        # assert type(self.featurizer) in [networks.MLP, networks.MNIST_CNN, wide_resnet.Wide_ResNet, networks.ResNet]
        
        if self.hparams['urm_generator_output'] == 'tanh':
            self.featurizer.activation = nn.Tanh()

        elif self.hparams['urm_generator_output'] == 'sigmoid':
            self.featurizer.activation = nn.Sigmoid()
        
        elif self.hparams['urm_generator_output'] == 'identity':
            self.featurizer.activation = nn.Identity()

        elif self.hparams['urm_generator_output'] == 'relu':
            self.featurizer.activation = nn.ReLU()

        else:
            raise Exception('unrecognized output activation: %s' % self.hparams['urm_generator_output'])

    def _setup_adversarial_net(self):
        print('--> Initializing discriminator <--')        
        self.discriminator = self._init_discriminator()
        self.discriminator_loss = torch.nn.BCEWithLogitsLoss(reduction="mean") # apply on logit

        # featurizer optimized by self.optimizer only
        if self.hparams["urm_discriminator_optimizer"] == 'sgd':
            self.discriminator_opt = torch.optim.SGD(self.discriminator.parameters(), lr=self.hparams['urm_discriminator_lr'], \
                weight_decay=self.hparams['weight_decay'], momentum=0.9)
        elif self.hparams["urm_discriminator_optimizer"] == 'adam':
            self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams['urm_discriminator_lr'], \
                weight_decay=self.hparams['weight_decay'])
        else:
            raise Exception('%s unimplemented' % self.hparams["urm_discriminator_optimizer"])

        self._modify_generator_output()
        self.sigmoid = nn.Sigmoid() # to compute discriminator acc.
            
    def _init_discriminator(self):
        """
        3 hidden layer MLP
        """
        model = nn.Sequential()
        model.add_module("dense1", nn.Linear(self.featurizer.n_outputs, 100))
        model.add_module("act1", nn.LeakyReLU())

        for _ in range(self.hparams['urm_discriminator_hidden_layers']):            
            model.add_module("dense%d" % (2+_), nn.Linear(100, 100))
            model.add_module("act2%d" % (2+_), nn.LeakyReLU())

        model.add_module("output", nn.Linear(100, 1)) 
        return model

    def _generate_noise(self, feats):
        """
        If U is a random variable uniformly distributed on [0, 1), then (b-a)*U + a is uniformly distributed on [a, b).
        """
        if self.hparams['urm_generator_output'] == 'tanh':
            a,b = -1,1
        elif self.hparams['urm_generator_output'] == 'relu':
            a,b = 0,1
        elif self.hparams['urm_generator_output'] == 'sigmoid':
            a,b = 0,1
        else:
            raise Exception('unrecognized output activation: %s' % self.hparams['urm_generator_output'])

        uniform_noise = torch.rand(feats.size(), dtype=feats.dtype, layout=feats.layout, device=feats.device) # U~[0,1]
        n = ((b-a) * uniform_noise) + a # n ~ [a,b)
        return n

    def _generate_soft_labels(self, size, device, a ,b):
        # returns size random numbers in [a,b]
         uniform_noise = torch.rand(size, device=device) # U~[0,1]
         return ((b-a) * uniform_noise) + a

    def get_accuracy(self, y_true, y_prob):
        # y_prob is binary probability
        assert y_true.ndim == 1 and y_true.size() == y_prob.size()
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item() / y_true.size(0)

    def return_feats(self, x):
        return self.featurizer(x)

    def _update_discriminator(self, x, y, feats):
        # feats = self.return_feats(x)
        feats = feats.detach() # don't backbrop through encoder in this step
        noise = self._generate_noise(feats)
        
        noise_logits = self.discriminator(noise) # (N,1)
        feats_logits = self.discriminator(feats) # (N,1)

        # hard targets
        hard_true_y = torch.tensor([1] * noise.shape[0], device=noise.device, dtype=noise.dtype) # [1,1...1] noise is true
        hard_fake_y = torch.tensor([0] * feats.shape[0], device=feats.device, dtype=feats.dtype) # [0,0...0] feats are fake (generated)

        if self.hparams['urm_discriminator_label_smoothing']:
            # label smoothing in discriminator
            soft_true_y = self._generate_soft_labels(noise.shape[0], noise.device, 1-self.hparams['urm_discriminator_label_smoothing'], 1.0) # random labels in range
            soft_fake_y = self._generate_soft_labels(feats.shape[0], feats.device, 0, 0+self.hparams['urm_discriminator_label_smoothing']) # random labels in range
            true_y = soft_true_y
            fake_y = soft_fake_y
        else:
            true_y = hard_true_y
            fake_y = hard_fake_y

        noise_loss = self.discriminator_loss(noise_logits.squeeze(1), true_y) # pass logits to BCEWithLogitsLoss
        feats_loss = self.discriminator_loss(feats_logits.squeeze(1), fake_y) # pass logits to BCEWithLogitsLoss

        d_loss = 1*noise_loss + self.hparams['urm_adv_lambda']*feats_loss

        # update discriminator
        self.discriminator_opt.zero_grad()
        d_loss.backward()
        self.discriminator_opt.step()

    def _compute_loss(self, x, y):
        feats = self.return_feats(x)
        ce_loss = self.loss(self.classifier(feats), y).mean()

        # train generator/encoder to make discriminator classify feats as noise (label 1)
        true_y = torch.tensor(feats.shape[0]*[1], device=feats.device, dtype=feats.dtype)
        g_logits = self.discriminator(feats)
        g_loss = self.discriminator_loss(g_logits.squeeze(1), true_y) # apply BCEWithLogitsLoss to discriminator's logit output
        loss = ce_loss + self.hparams['urm_adv_lambda']*g_loss

        return loss, feats

    def update(self, minibatches, unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        
        all_x = []
        all_y = []
        for x, y in minibatches:
            all_x.extend(x)
            all_y.append(y)
        all_y = torch.cat(all_y)
            
        loss, feats = self._compute_loss(all_x, all_y)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self._update_discriminator(all_x, all_y, feats)
    
        return {'loss': loss.item()}
