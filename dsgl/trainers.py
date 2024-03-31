from __future__ import print_function, absolute_import
import time
import torch
import torch.nn.functional as F
from .utils.meters import AverageMeter

class Trainer(object):
    def __init__(self, encoder, memory=None, distangle=None, re_crit=None, hard_weight=0.5):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.distangle = distangle
        self.re_crit = re_crit
        self.hard_weight = hard_weight
        # self.trip = TripletLoss(margin=0.3)

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, irre_labels, indexes = self._parse_data(inputs)  # obtain the label-irrelevant labels, how to utilize it in the whole process

            # loss = 0
            # forward
            f_out = self._forward(inputs)
            loss_m = self.memory(f_out, labels, irre_labels, indexes, epoch)
            K_order = irre_labels.size(1)
            loss_order = 0.

            if self.distangle is not None:
                hard_memory, clu_memory = self.memory.features.chunk(2, dim=0)
                self.distangle.clu_memory = clu_memory  # mean cluster feature
                for k_order in range(K_order):
                    loss_d = self.distangle(inputs=f_out, targets=labels, irre_targets=irre_labels[:, 0].view(-1, 1), indexes=indexes)
                    loss_order += loss_d
                loss_dis = loss_order * (1. / K_order)

                if True:
                    pos_samples, pos_indices = self.re_crit.get_positive_samples(indexes=indexes, targets=labels)
                    loss_kl, f_hat, p_i = self.re_crit.relative_entropy_criterion(model=self.encoder, f_out=f_out,
                                                                  pos_samples=pos_samples, protomemory=self.memory.features,
                                                                  pos_indices=pos_indices, instance_memory=self.distangle.ins_memory,
                                                                  targets=labels, irre_targets=irre_labels, indexes=indexes, epoch=epoch)  # irre_targets=irre_labels[:, 0].view(-1)

                loss = loss_m + loss_dis + loss_kl
            else:
                loss = loss_m

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, irre_pids,  _, indexes = inputs
        return imgs.cuda(), pids.cuda(), irre_pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)