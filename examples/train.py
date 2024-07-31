# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
print(os.path.dirname(os.path.abspath(__file__)) + '/..')

from dsgl import datasets
from dsgl import models
from dsgl.models.cm import ClusterMemory
from dsgl.trainers import Trainer
from dsgl.evaluators import Evaluator, extract_features
from dsgl.utils.data import IterLoader
from dsgl.utils.data import transforms as T
from dsgl.utils.data.sampler import RandomMultipleGallerySampler
from dsgl.utils.data.preprocessor import Preprocessor
from dsgl.utils.logging import Logger
from dsgl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from dsgl.utils.faiss_rerank import compute_jaccard_distance
from dsgl.models.distangled_features import id_irrelevant_label, DistangledLearn, RelativeEntropy
# from dsgl.models.uncertains import UncertainSamples

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, train_type=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer, train_type=False),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    
    # Load from checkpoint
    if args.resume:
        global start_epoch
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model, strip='module.')
        start_epoch = checkpoint['epoch']
    
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = Trainer(model)

    for epoch in range(start_epoch, args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == start_epoch:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_outliers = (pseudo_labels == -1).sum()

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)

        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                if id != -1:
                    labels.append(id)
                else:
                    labels.append(num + outliers)
                    outliers += 1
            return torch.Tensor(labels).long()

        idx2labels = generate_pseudo_labels(pseudo_labels, num_cluster)  # index2target

        # obtain generation pseudo labels
        if epoch == 0:
            prior_idx2labels = torch.arange(idx2labels.size(0))
            prior_logits = features.mm(cluster_features.t())
        uncern = UncertainSamples(prelogits=prior_logits, prelabels=prior_idx2labels, labels=idx2labels, temp=args.temp)
        prior_idx2labels = idx2labels.clone()

        del cluster_loader

        # Create memory bank
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                                momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                num_instances=args.num_instances, hard_weight=0.5).cuda()  # hard_weight=1.0

        if args.memorybank=='DLhybrid':
            memory.features = F.normalize(cluster_features.repeat(2, 1), dim=1).cuda()
        elif args.memorybank=='DLhybrid_v2':
            memory.features = F.normalize(cluster_features.repeat(2, 1), dim=1).cuda()
        else:
            memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        id_irre_labels = id_irrelevant_label(instance_feat=features, cluster_feat=memory.features, pseudo_labels=pseudo_labels, k=2)  # label irre pseudo labels

        distangle = DistangledLearn(
            ins_memory=features.clone().cuda(),
            labels=pseudo_labels,
            irre_labels=id_irre_labels,
            clu_memory=cluster_features, tau=0.5)

        re_crit = RelativeEntropy(index2targets=idx2labels,
                                  clusterset=sorted(dataset.train), uncertain=uncern,
                                  irre_labels=id_irre_labels[:, 0].view(-1), labels=pseudo_labels)
        trainer.re_crit = re_crit

        del features

        if args.memorybank == 'DLhybrid':
            trainer.distangle = distangle
        elif args.memorybank == 'DLhybrid_v2':
            trainer.memory.distangle = distangle

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label, irre_label) in enumerate(zip(sorted(dataset.train), pseudo_labels, id_irre_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), irre_label, cid, i))

        print('==> Statistics for epoch {}: {} clusters, {} outliers'.format(epoch, num_cluster, num_outliers))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        # update prior_logits
        prior_logits = torch.mm(distangle.ins_memory, distangle.clu_memory.t())

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Disentangled Sample Guidance Learning for Unsupervised Person Re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--smooth', type=float, default=0, help="label smoothing")
    parser.add_argument('--hard-weight', type=float, default=0.5, help="hard weights")
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the memory bank")
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('-mb', '--memorybank', type=str, default='DLhybrid', choices=['DLhybrid', 'DLhybrid_v2'])

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, metavar='PATH', default='')
    main()
