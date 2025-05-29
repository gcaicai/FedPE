import os
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import pickle
import random


def load_data(file_path):
    # get idx-->entity
    with open(os.path.join(file_path, 'entity2idEnc.txt'), errors='ignore') as f:
        entity2id = dict()
        lst = f.readlines()
        for line in lst:
            ent_id = line.strip('\n').split(' ')
            entity2id[int(ent_id[-1])] = ent_id[0]

    # get idx-->relation
    with open(os.path.join(file_path, 'relation2idEnc.txt'), errors='ignore') as f:
        relation2id = dict()
        lst = f.readlines()
        for line in lst:
            rel_id = line.strip('\n').split(' ')
            relation2id[int(rel_id[-1])] = rel_id[0]

    train_triples = read_triplets(os.path.join(file_path, 'train2id.txt'))
    valid_triples = read_triplets(os.path.join(file_path, 'valid2id.txt'))
    test_triples = read_triplets(os.path.join(file_path, 'test2id.txt'))

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triples)))
    print('num_valid_triples: {}'.format(len(valid_triples)))
    print('num_test_triples: {}'.format(len(test_triples)))

    return entity2id, relation2id, train_triples, valid_triples, test_triples


def read_triplets(file_path):
    triplets = []
    with open(file_path) as f:
        lst = f.readlines()
        lst.pop(0)
        for line in lst:
            head, tail, relation = line.strip('\n').split(' ')
            triplets.append((int(head), int(relation), int(tail)))
    return np.array(triplets)


def process_federated_dataset(read_path, num_client, write_path):
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data(read_path)

    triples = np.concatenate((train_triplets, valid_triplets), axis=0)
    triples = np.concatenate((triples, test_triplets), axis=0)

    np.random.shuffle(triples)

    client_triples = np.array_split(triples, num_client)

    for idx, val in enumerate(client_triples):
        client_triples[idx] = client_triples[idx].tolist()

    client_data = []

    for client_idx in tqdm(range(num_client)):
        triples_idx = client_triples[client_idx]

        ent_freq = ddict(int)
        rel_freq = ddict(int)

        for tri in triples_idx:
            h, r, t = tri
            ent_freq[h] += 1
            ent_freq[t] += 1
            rel_freq[r] += 1

        # reconstruct train:valid:test sets at the ratio of 8:1:1
        client_train_triples = []
        client_valid_triples = []
        client_test_triples = []

        for idx, tri in enumerate(triples_idx):
            h, r, t = tri
            # if the frequency of h r t >=2 , classify them into valid and test sets, else train set
            if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
                client_test_triples.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            else:
                client_train_triples.append(tri)

            if len(client_test_triples) > int(len(triples_idx) * 0.2):
                break

        client_train_triples.extend(triples_idx[idx + 1:])

        random.shuffle(client_test_triples)
        test_len = len(client_test_triples)

        client_valid_triples = client_test_triples[:int(test_len / 2)]  # valid set
        client_test_triples = client_test_triples[int(test_len / 2):]  # test set

        train_edge_index = np.array(client_train_triples)[:, [0, 2]].T
        train_edge_type = np.array(client_train_triples)[:, 1].T

        valid_edge_index = np.array(client_valid_triples)[:, [0, 2]].T
        valid_edge_type = np.array(client_valid_triples)[:, 1].T

        test_edge_index = np.array(client_test_triples)[:, [0, 2]].T
        test_edge_type = np.array(client_test_triples)[:, 1].T

        client_data_dict = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type},
                            'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type},
                            'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type}}

        client_data.append(client_data_dict)

    # save dataset
    pickle.dump(client_data, open(write_path, 'wb'))


def process_dynamic_dataset(read_path, num_client, write_path):
    all_data = pickle.load(open(read_path, 'rb'))

    client_data = []
    for client_idx in tqdm(range(num_client)):
        train_edge_index = all_data[client_idx]['train']['edge_index']
        train_edge_type = all_data[client_idx]['train']['edge_type']

        test_edge_index = all_data[client_idx]['test']['edge_index']
        test_edge_type = all_data[client_idx]['test']['edge_type']

        valid_edge_index = all_data[client_idx]['valid']['edge_index']
        valid_edge_type = all_data[client_idx]['valid']['edge_type']

        client_data_dict = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type},
                            'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type},
                            'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type}}

        client_data.append(client_data_dict)

    pickle.dump(client_data, open(write_path, 'wb'))


if __name__ == '__main__':
    random.seed(12345)

    federated = True  # choose to process federated datasets, otherwise process dynamic datasets

    if federated:
        '''process for writing federated datasets'''
        federated_read_path = './fed_data/FB15K237/'
        federated_num_client = 3  # the number of clients, choices = [3, 5, 10, or a specified number]
        federated_write_path = 'fed_data/FB15K237/FB15K237-Fed' + str(federated_num_client) + '.pkl'  # storage datapath

        process_federated_dataset(federated_read_path, federated_num_client, federated_write_path)
    else:
        '''process for writing dynamic federated datasets'''
        dynamic_read_path = 'fed_data/NELL-995/NELL-995-Fed10.pkl'
        dynamic_num_client = 5  # the number of clients, choices = [5, 6, 7, 8, 9, 10]
        dynamic_write_path = 'fed_data/dynamic_datasets/NELL-995/NELL-995-Fed' + str(dynamic_num_client) + '.pkl'

        process_dynamic_dataset(dynamic_read_path, dynamic_num_client, dynamic_write_path)

