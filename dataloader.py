import numpy as np
import torch
from collections import defaultdict as ddict
from torch.utils.data import Dataset


class TrainDataset_fedpe(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size

        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        # construct negative samples
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.isin(
                negative_sample,
                np.array(self.hr2t[(head, relation)]),
                invert=True
            )
            negative_sample = negative_sample[mask]

            if negative_sample.size > 0:
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, idx

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])

        return positive_sample, negative_sample, sample_idx


class TestDataset_fedpe(Dataset):
    def __init__(self, triples, all_true_triples, nentity, rel_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples

        self.nentity = nentity

        self.rel_mask = rel_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)

        return triple, trp_label

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = self.hr2t_all[(head, relation)]
        trp_label = self.get_label(label)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        y = np.zeros([self.nentity], dtype=np.float32)

        for e2 in label:
            y[e2] = 1.0

        return torch.FloatTensor(y)


def get_task_dataset_fedpe(data, args):
    train_edge_index = np.array([[], []], dtype=np.int64)
    train_edge_type = np.array([], dtype=np.int64)

    valid_edge_index = np.array([[], []], dtype=np.int64)
    valid_edge_type = np.array([], dtype=np.int64)

    test_edge_index = np.array([[], []], dtype=np.int64)
    test_edge_type = np.array([], dtype=np.int64)

    for d in data:
        train_edge_index = np.concatenate([train_edge_index, d['train']['edge_index']], axis=-1)
        valid_edge_index = np.concatenate([valid_edge_index, d['valid']['edge_index']], axis=-1)
        test_edge_index = np.concatenate([test_edge_index, d['test']['edge_index']], axis=-1)

        train_edge_type = np.concatenate([train_edge_type, d['train']['edge_type']], axis=-1)
        valid_edge_type = np.concatenate([valid_edge_type, d['valid']['edge_type']], axis=-1)
        test_edge_type = np.concatenate([test_edge_type, d['test']['edge_type']], axis=-1)

    train_triples = np.stack((train_edge_index[0],
                              train_edge_type,
                              train_edge_index[1])).T
    valid_triples = np.stack((valid_edge_index[0],
                              valid_edge_type,
                              valid_edge_index[1])).T
    test_triples = np.stack((test_edge_index[0],
                             test_edge_type,
                             test_edge_index[1])).T

    all_triples = np.concatenate([train_triples, valid_triples, test_triples])

    if 'FB15K237' in args.data_path:
        nentity = 14541
        nrelation = 237
    elif 'WN18RR' in args.data_path:
        nentity = 40943
        nrelation = 11
    elif 'NELL-995' in args.data_path:
        nentity = 75492
        nrelation = 200

    train_dataset = TrainDataset_fedpe(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset_fedpe(valid_triples, all_triples, nentity)
    test_dataset = TestDataset_fedpe(test_triples, all_triples, nentity)

    return train_dataset, valid_dataset, test_dataset, nrelation, nentity

