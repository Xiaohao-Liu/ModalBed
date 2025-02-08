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

class ADRMX(Algorithm):
    '''ADRMX: Additive Disentanglement of Domain Features with Remix Loss from (https://arxiv.org/abs/2308.06624)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ADRMX, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.mix_num = 1
        self.scl_int = SupConLossLambda(lamda=0.5)
        self.scl_final = SupConLossLambda(lamda=0.5)

        self.featurizer_label = networks.Featurizer(input_shape, self.hparams)
        self.featurizer_domain = networks.Featurizer(input_shape, self.hparams)

        self.discriminator = networks.MLP(self.featurizer_domain.n_outputs,
            num_domains, self.hparams)

        self.classifier_label_1 = networks.Classifier(
            self.featurizer_label.n_outputs,
            num_classes,
            is_nonlinear=True)

        self.classifier_label_2 = networks.Classifier(
            self.featurizer_label.n_outputs,
            num_classes,
            is_nonlinear=True)

        self.classifier_domain = networks.Classifier(
            self.featurizer_domain.n_outputs,
            num_domains,
            is_nonlinear=True)


        self.network = nn.Sequential(self.featurizer_label, self.classifier_label_1)

        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters())),
            lr=self.hparams["lr"],
            betas=(self.hparams['beta1'], 0.9))

        self.opt = torch.optim.Adam(
            (list(self.featurizer_label.parameters()) +
             list(self.featurizer_domain.parameters()) +
             list(self.classifier_label_1.parameters()) +
                list(self.classifier_label_2.parameters()) +
                list(self.classifier_domain.parameters())),
            lr=self.hparams["lr"],
            betas=(self.hparams['beta1'], 0.9))
                                                    
    def update(self, minibatches, unlabeled=None):

        self.update_count += 1
        # all_x = torch.cat([x for x, _ in minibatches])
        # all_y = torch.cat([y for _, y in minibatches])
        all_x = []
        all_y = []
        for x, y in minibatches:
            all_x.extend(x)
            all_y.append(y)
        all_y = torch.cat(all_y)
        
        # import pdb; pdb.set_trace()

        feat_label = self.featurizer_label(all_x)
        feat_domain = self.featurizer_domain(all_x)
        feat_combined = feat_label - feat_domain
        # import pdb; pdb.set_trace()
        # get domain labels
        
        disc_labels = torch.cat([
            torch.full((len(x), ), i, dtype=torch.int64, device=all_y.device)
            for i, (x, _) in enumerate(minibatches)
        ])
        # predict domain feats from disentangled features
        disc_out = self.discriminator(feat_combined) 
        disc_loss = F.cross_entropy(disc_out, disc_labels) # discriminative loss for final labels (ascend/descend)

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        # alternating losses
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):
            # in discriminator turn
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'loss_disc': disc_loss.item()}
        else:
            # in generator turn

            domain_preds = self.classifier_domain(feat_domain)
            classifier_loss_domain = F.cross_entropy(domain_preds, disc_labels) # domain clf loss
            classifier_remixed_loss = 0

            # calculate CE and contrastive loss from x_label
            int_preds = self.classifier_label_1(feat_label)
            classifier_loss_int = F.cross_entropy(int_preds, all_y) # intermediate CE Loss
            cnt_loss_int = self.scl_int(feat_label, all_y, disc_labels)

            # calculate CE and contrastive loss from x_dinv
            final_preds = self.classifier_label_2(feat_combined)
            classifier_loss_final = F.cross_entropy(final_preds, all_y) # final CE Loss
            cnt_loss_final = self.scl_final(feat_combined, all_y, disc_labels)

            # remix strategy
            for i in range(self.num_classes):
                indices = torch.where(all_y == i)[0]
                for _ in range(self.mix_num):
                    # get two instances from same class with different domains
                    perm = torch.randperm(indices.numel())
                    if len(perm) < 2:
                        continue
                    idx1, idx2 = perm[:2]
                    # remix
                    remixed_feat = feat_combined[idx1] + feat_domain[idx2]
                    # make prediction
                    pred = self.classifier_label_1(remixed_feat.view(1,-1))
                    # accumulate the loss
                    classifier_remixed_loss += F.cross_entropy(pred.view(1, -1), all_y[idx1].view(-1))
            # normalize
            classifier_remixed_loss /= (self.num_classes * self.mix_num)

            # generator loss negates the discrimination loss (negative update)
            gen_loss = (classifier_loss_int +
                        classifier_loss_final +
                        self.hparams["dclf_lambda"] * classifier_loss_domain +
                        self.hparams["rmxd_lambda"] * classifier_remixed_loss +
                        self.hparams['cnt_lambda'] * (cnt_loss_int + cnt_loss_final) + 
                        (self.hparams['disc_lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.opt.zero_grad()
            gen_loss.backward()
            self.opt.step()

            return {'loss_total': gen_loss.item(), 
                'loss_cnt_int': cnt_loss_int.item(),
                'loss_cnt_final': cnt_loss_final.item(),
                'loss_clf_int': classifier_loss_int.item(), 
                'loss_clf_fin': classifier_loss_final.item(), 
                'loss_dmn': classifier_loss_domain.item(), 
                'loss_disc': disc_loss.item(),
                'loss_remixed': classifier_remixed_loss.item(),
                }
    
    def predict(self, x):
        return self.network(x)
