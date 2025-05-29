import sys
from kge_model import KGEModel
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedpe import KGERunner
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import json
import pickle
import numpy as np


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

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='a+')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))

    logging.basicConfig(
        level=logging.INFO,  # 设置日志的最低级别为 INFO
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler]  # 使用自定义的文件日志 handler
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(console_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='fed_data/dynamic_datasets/FB15K237/FB15K237-Fed5.pkl', type=str)
    parser.add_argument('--name', default='fb15k237-fed5_dynamic-fedpe-transe', type=str)
    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    parser.add_argument('--run_mode', default='FedPE', choices=['FedPE'])
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

    parser.add_argument('--max_round', default=10000, type=int)
    parser.add_argument('--local_epoch', default=3, type=int)
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--log_per_round', default=1, type=int)
    parser.add_argument('--check_per_round', default=5, type=int)

    parser.add_argument('--early_stop_patience', default=5, type=int)
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--gpu', default='0', type=str)  # default set as 0
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)

    parser.add_argument('--seed', default=12345, type=int)

    args = parser.parse_args()
    args_str = json.dumps(vars(args))

    args.gpu = torch.device(("cuda:" + args.gpu) if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    init_dir(args)
    writer = SummaryWriter(os.path.join(args.tb_log_dir, args.name))
    args.writer = writer
    init_logger(args)
    logging.info(args_str)

    if args.run_mode == 'FedPE':
        fed_data = pickle.load(open(args.data_path, 'rb'))
        learner = KGERunner(args, fed_data)
        learner.train()

