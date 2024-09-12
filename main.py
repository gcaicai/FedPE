import sys
from kge_model import KGEModel
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import *
from fedpc import KGERunner

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import logging
import argparse
import json
import pickle
import numpy as np


def test_pretrain(args, all_data):
    data_len = len(all_data)

    embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
    kge_model = KGEModel(args, model_name=args.model)

    results = ddict(float)
    for i, data in enumerate(all_data):
        one_results = ddict(float)
        state = torch.load('state_new/fb15k237_fed3_TransE.best', map_location=args.gpu)
        rel_embed = state['rel_emb'].detach()
        ent_embed = state['ent_emb'].detach()

        train_dataset, valid_dataset, test_dataset, nrelation, nentity = get_task_dataset(data, args)
        test_dataloader_tail = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=max(1, args.num_cpu),
            collate_fn=TestDataset.collate_fn
        )

        client_res = ddict(float)
        for batch in test_dataloader_tail:
            # triplets, labels, mode = batch
            mode = args.model
            triplets, labels = batch
            # triplets, labels, mode = next(test_dataloader_list[i].__iter__())
            triplets, labels = triplets.to(args.gpu), labels.to(args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]

            pred = kge_model((triplets, None),
                              rel_embed,
                              ent_embed)
            b_range = torch.arange(pred.size()[0], device=args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]

            ranks = ranks.float()
            count = torch.numel(ranks)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            one_results['count'] += count
            one_results['mr'] += torch.sum(ranks).item()
            one_results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 3, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])
                one_results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in one_results.items():
            if k != 'count':
                one_results[k] = v / one_results['count']

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
            one_results['mrr'], one_results['hits@1'],
            one_results['hits@3'], one_results['hits@10']))

    for k, v in results.items():
        if k != 'count':
            results[k] = v / results['count']

    logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
        results['mrr'], results['hits@1'],
        results['hits@3'], results['hits@10']))

    return results


def init_dir(args):
    # state_new
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def init_logger(args):
    log_file = os.path.join(args.log_dir, args.name + '.log')

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='fed_data/FB15K237/FB15K237-Fed3.pkl', type=str)
    parser.add_argument('--name', default='fb15k237-fed3-fedpc-transe', type=str)
    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    parser.add_argument('--run_mode', default='FedPC', choices=['FedPC', 'Single'])
    parser.add_argument('--num_multi', default=3, type=int)

    parser.add_argument('--model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'])

    # one task hyperparam
    parser.add_argument('--one_client_idx', default=0, type=int)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--log_per_epoch', default=1, type=int)
    parser.add_argument('--check_per_epoch', default=10, type=int)

    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)

    # for FedE
    parser.add_argument('--num_client', default=3, type=int)
    parser.add_argument('--max_round', default=10000, type=int)
    parser.add_argument('--local_epoch', default=3, type=int)
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--log_per_round', default=1, type=int)
    parser.add_argument('--check_per_round', default=5, type=int)

    parser.add_argument('--early_stop_patience', default=5, type=int)
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--gpu', default='0', type=str) # default set as 0
    parser.add_argument('--num_cpu', default=10, type=int)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)

    # parser.add_argument('--negative_adversarial_sampling', default=True, type=bool)
    parser.add_argument('--seed', default=12345, type=int)

    args = parser.parse_args()
    args_str = json.dumps(vars(args))

    args.gpu = torch.device('cuda:' + args.gpu)
    # args.gpu = torch.device(("cuda:" + args.gpu) if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    init_dir(args)
    writer = SummaryWriter(os.path.join(args.tb_log_dir, args.name))
    args.writer = writer
    init_logger(args)
    logging.info(args_str)

    if args.run_mode == 'FedPC':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = KGERunner(args, all_data)
        learner.train()
    elif args.run_mode == 'Single':
        all_data = pickle.load(open(args.data_path, 'rb'))
        data = all_data[args.one_client_idx]
        learner = KGERunner(args, data)
        learner.train()
    # elif args.run_mode == 'test_pretrain':
    #     all_data = pickle.load(open(args.data_path, 'rb'))
    #     test_pretrain(args, all_data)
