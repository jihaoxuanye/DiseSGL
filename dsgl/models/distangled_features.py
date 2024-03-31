import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..utils.data import Preprocessor
from ..utils.data import transforms as T
from torch.autograd import Variable

def id_irrelevant_label(instance_feat, cluster_feat, pseudo_labels, k, temp=0.05):

    irre_labels = torch.zeros_like(torch.from_numpy(pseudo_labels)) - 1.
    irre_labels = irre_labels.view(-1, 1).repeat(1, k)

    hard_cluster_feats, mean_cluster_feats = cluster_feat.chunk(2, dim=0)  # mean_cluster_feats: (N_c*d)feats
    logits = instance_feat.mm(mean_cluster_feats.cpu().t())  # logits shape: N*N_c
    logits /= temp
    prob_logits = F.softmax(logits, dim=-1)

    for idx, lbl in enumerate(pseudo_labels):
        if lbl >= 0:
            prob_logits[idx, lbl] += -10.
            ir_lbl = torch.topk(prob_logits[idx], k=k, largest=True)[1]
            if False:
                c_ir_lbl = torch.argmax(prob_logits[idx])
            irre_labels[idx] = ir_lbl

    return irre_labels

class DistangledLearn(nn.Module):

    def __init__(self, ins_memory, labels, irre_labels, clu_memory=None, temp=0.05, tau=0.5):
        super(DistangledLearn, self).__init__()
        self.ins_memory = ins_memory
        self.labels = torch.from_numpy(labels).cuda()
        self.irre_labels = irre_labels.cuda()
        self.clu_memory = clu_memory
        self.temp = temp
        self.tau = tau

    def distangle_contrastive_feature(self, target, irre_target, ik):

        # positive distangle contrast features
        positive_samples_idx = torch.nonzero(self.labels == target).view(-1)
        positive_irre_samples = positive_samples_idx[torch.nonzero(self.irre_labels[positive_samples_idx, ik] != irre_target).view(-1)]
        if positive_irre_samples.size(0) == 0:
            m_pos = self.clu_memory[target].clone().detach().view(1, -1)  # self.ins_memory[positive_samples_idx]
        else:
            m_pos = self.ins_memory[positive_irre_samples]

        m_pos = torch.mean(m_pos, dim=0).detach()

        # negative distangle contrast features (multi-clusters)
        rele_samples_idx = torch.nonzero(self.irre_labels[:, ik] == irre_target).view(-1)
        negative_rele_samples = rele_samples_idx[torch.nonzero(self.labels[rele_samples_idx] != target).view(-1)]
        m_negs = []
        if negative_rele_samples.size(0) > 0:
            negatives_clusters = torch.unique(self.labels[negative_rele_samples])
            for neg_cluster in negatives_clusters:
                negative_cluster_rele = negative_rele_samples[torch.nonzero(self.labels[negative_rele_samples] == neg_cluster).view(-1)]
                m_neg = self.ins_memory[negative_cluster_rele]
                m_neg = torch.mean(m_neg, dim=0).detach()
                m_negs.append(m_neg)
        else:
            m_negs.append(self.clu_memory[irre_target].clone().detach())
            negatives_clusters = irre_target.view(1, -1)

        m_pos = m_pos.view(1, -1).cuda()  # 1 * 2048
        m_negs = torch.stack(m_negs, dim=0).cuda()  # n * 2048

        # return F.normalize(m_pos, dim=1), F.normalize(m_negs, dim=1)
        return m_pos, m_negs, negatives_clusters

    def distangle_proto(self, inputs, targets, irre_targets, indexes):

        dist_pos_distengle_proto, dist_neg_distengle_proto_list = dict(), dict()

        for k in range(irre_targets.size(1)):
            pos_distengle_proto = torch.zeros_like(inputs)
            neg_distengle_proto_list, neg_clusters_idx = list(), list()
            for indice, (target, irre_target, index) in enumerate(zip(targets, irre_targets[:, k], indexes)):
                dist_m_pos, dist_m_neg, neg_c = self.distangle_contrastive_feature(target=target, irre_target=irre_target, ik=k)

                pos_distengle_proto[indice] = dist_m_pos
                neg_distengle_proto_list.append(dist_m_neg)
                neg_clusters_idx.append(neg_c)

            dist_pos_distengle_proto[k] = pos_distengle_proto
            dist_neg_distengle_proto_list[k] = neg_distengle_proto_list

        return dist_pos_distengle_proto, dist_neg_distengle_proto_list, neg_clusters_idx

    def forward(self, inputs, targets, irre_targets, indexes):

        pos_distengle_proto, neg_distengle_proto_list, neg_clusters_idx = self.distangle_proto(
            inputs=inputs,
            targets=targets,
            irre_targets=irre_targets,
            indexes=indexes
        )

        for indice, index in enumerate(indexes):
            self.ins_memory[index] = inputs[indice].clone().detach()

        # pos_dist_prototypes = F.normalize(self.clu_memory.clone(), dim=1)
        k = irre_targets.size(1)
        pos_dist_prototypes = self.clu_memory.clone()
        delta_pos_distengle_proto = torch.zeros_like(pos_distengle_proto[0])
        for key in pos_distengle_proto:
            delta_pos_distengle_proto += pos_distengle_proto[key]
        pos_dist_prototypes[targets] = (1. - self.tau) * pos_dist_prototypes[targets] + (self.tau / k) * delta_pos_distengle_proto
        pos_dist_prototypes = F.normalize(pos_dist_prototypes, dim=1)   # normlization positive distengle prototype.

        outputs = inputs.mm(pos_dist_prototypes.t())
        outputs /= self.temp

        l_pos = torch.exp(outputs[:, targets].diag())
        l = torch.exp(outputs)
        l_ = l.sum(dim=1)

        for key in neg_distengle_proto_list:
            for ind in range(inputs.size(0)):
                neg_dist_proto = F.normalize(neg_distengle_proto_list[key][ind], dim=1)
                delta_outputs_dist_n = inputs[ind].view(1, -1).mm(neg_dist_proto.t())
                delta_outputs_dist_n /= self.temp
                delta_l_dist_n = torch.exp(delta_outputs_dist_n).sum()
                if False:
                    distengle_l = l[ind, neg_clusters_idx[ind]].sum()
                    delta_l_dist_ = torch.clamp(delta_l_dist_n - distengle_l, min=0.0)
                    l_[ind] += (self.tau / k) * delta_l_dist_
                    # check part
                    if l_[ind] < l_pos[ind]: print('Error')
                else:
                    l_[ind] += (self.tau / k) * delta_l_dist_n

        loss_dis = -1 * torch.log(l_pos / l_).mean()

        return loss_dis

# --------- calculate the relative entropy of a positive pairs --------

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

re_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

class RelativeEntropy(object):
    def __init__(self, index2targets, clusterset=None, uncertain=None, temp=0.05, irre_labels=None, labels=None, tau=0.8, tau_1=0.1, inner_iter=1):
        self.index2targets = index2targets.cuda()
        self.kl_div = nn.KLDivLoss(size_average=False, reduce=False)
        self.re_sampler = Preprocessor(dataset=clusterset, transform=re_transformer)
        self.labels = torch.from_numpy(labels).cuda()
        self.irre_labels = irre_labels.cuda()
        self.temp = temp
        self.tau = tau
        self.tau_1 = tau_1
        self.inner_iter = inner_iter
        if uncertain is not None:
            self.uncertain = uncertain

    def get_positive_samples(self, indexes, targets):

        batch_pospair = list()
        # targets_h = self.index2targets_h[indexes]

        for i, th in enumerate(targets):
            pos_idx = torch.nonzero(self.index2targets == th).view(-1)
            if pos_idx.size(0) > 1:
                ran_pos_idx = np.random.choice(pos_idx.cpu(), size=1)
                batch_pospair.append([indexes[i].item(), ran_pos_idx.item()])
            else:
                batch_pospair.append([indexes[i].item(), indexes[i].item()])

        batch_pospair = np.array(batch_pospair)
        positive_indices = batch_pospair[:, 1].tolist()  # (<64)
        positive_samples = self.re_sampler.sampler(positive_indices)
        positive_samples = torch.stack(positive_samples).cuda()

        return positive_samples, torch.tensor(positive_indices)

    def get_distengle_proto(self, f_out, f_hat, protomemory, indexes, pos_indices, instance_memory, targets, irre_targets, epoch):

        dist_proto, dist_proto_hat = protomemory.clone(), protomemory.clone()
        irre_targets_hat = self.irre_labels[pos_indices]

        for idx, (tgt, ir_tgt, ir_tgt_hat, pos_ind) in enumerate(zip(targets, irre_targets, irre_targets_hat, pos_indices)):

            positive_samples_idx = torch.nonzero(self.index2targets == tgt).view(-1)

            # generate distengle positive protomemory for x_i
            re_positive_samples_idx = positive_samples_idx[torch.nonzero(self.irre_labels[positive_samples_idx] != ir_tgt)].view(-1)
            if re_positive_samples_idx.size(0) > 0:
                m_pos = instance_memory[re_positive_samples_idx]
                d_pos = torch.mean(m_pos, dim=0)
            else:
                d_pos = protomemory[tgt].clone().detach()

            # d_pos = F.normalize(d_pos, dim=-1)
            dist_proto[tgt] = (1 - self.tau) * dist_proto[tgt] + self.tau * d_pos
            dist_proto[tgt] /= dist_proto[tgt].norm()

            if True:
                # generate distangle negative protomemory for x_j
                re_samples_idx = torch.nonzero(self.irre_labels == ir_tgt).view(-1)
                re_negative_samples_idx = re_samples_idx[torch.nonzero(self.index2targets[re_samples_idx] != tgt)].view(-1)
                negative_clusters_idx = self.index2targets[re_negative_samples_idx]
                negative_clusters_idx = torch.unique(negative_clusters_idx)
                for ind, nc in enumerate(negative_clusters_idx):
                    nc_samples = re_negative_samples_idx[torch.nonzero(self.index2targets[re_negative_samples_idx] == nc)].view(-1)
                    m_neg = instance_memory[nc_samples]
                    # mixing negative disentangle features in mean manner
                    d_neg = torch.mean(m_neg, dim=0)
                    dist_proto[nc] = (1 - self.tau_1) * dist_proto[nc] + self.tau_1 * d_neg
                    dist_proto[nc] /= dist_proto[nc].norm()

        return dist_proto, dist_proto_hat

    def relative_entropy_criterion(self, model, f_out, pos_samples, protomemory, pos_indices, instance_memory, targets, irre_targets, indexes, epoch, margin=0.8):

        f_hat = model(pos_samples)
        hard_proto, mean_proto = protomemory.chunk(2, dim=0)

        K_order = irre_targets.size(1)
        loss_kl_sum = 0.

        for k_order in range(K_order):
            dist_proto, dist_proto_hat = self.get_distengle_proto(f_out, f_hat, mean_proto, indexes, pos_indices,
                                                                  instance_memory, targets, irre_targets[:, k_order].view(-1), epoch)
            logits = f_out.mm(dist_proto.t()) / self.temp
            logits = F.softmax(logits, dim=1)

            # positive prob distribution
            logits_hat = f_hat.mm(mean_proto.t()) / self.temp
            logits_hat = F.log_softmax(logits_hat, dim=1)
            loss_kl_k = torch.mean(self.kl_div(logits_hat, logits).sum(dim=1))
            loss_kl_sum += loss_kl_k

        loss_kl = loss_kl_sum * (1. / K_order)

        return loss_kl, f_hat, logits