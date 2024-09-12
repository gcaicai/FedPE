import os
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import pickle
import random


file_path='./fed_data/FB15K237/'
num_client = 3 # the number of clients
file_name='fed_data/FB15K237/FB15K237-Fed'+str(num_client)+'.pkl' # storage datapath


def load_data(file_path):
    print("load data from {}".format(file_path))

    # get idx-->entity
    with open(os.path.join(file_path, 'entity2idEnc.txt'),errors='ignore') as f:
        entity2id = dict()
        lst=f.readlines() # list[str]
        for line in lst:
            ent_id=line.strip('\n').split('\t')
            entity2id[int(ent_id[1])]=ent_id[0] # key(int)-->value(str)

    # get idx-->relation
    with open(os.path.join(file_path, 'relation2idEnc.txt'),errors='ignore') as f:
        relation2id = dict()
        lst=f.readlines() # list[str]
        for line in lst:
            rel_id=line.strip('\n').split('\t')
            relation2id[int(rel_id[1])]=rel_id[0]

    train_triplets = read_triplets(os.path.join(file_path, 'train2id.txt'))
    valid_triplets = read_triplets(os.path.join(file_path, 'valid2id.txt'))
    test_triplets = read_triplets(os.path.join(file_path, 'test2id.txt'))

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

def read_triplets(file_path):
    triplets = []
    with open(file_path) as f:
        lst=f.readlines()
        lst.pop(0)
        for line in lst:
            head,tail,relation=line.strip('\n').split(' ')
            triplets.append((int(head),int(relation),int(tail)))
    return np.array(triplets)

def check_datasets(train_triplets,valid_triplets,test_triplets,client_train_triples):
    a = train_triplets[:, 0]
    a = np.append(a, train_triplets[:, 2])
    a = np.unique(a)

    b = valid_triplets[:, 0]
    b = np.append(b, valid_triplets[:, 2])
    b = np.unique(b)

    c = test_triplets[:, 0]
    c = np.append(c, test_triplets[:, 2])
    c = np.unique(c)

    print(len(a), len(b), len(c))

    e = train_triplets[:, 1]
    e = np.unique(e)

    f = valid_triplets[:, 1]
    f = np.unique(f)

    g = test_triplets[:, 1]
    g = np.unique(g)

    print(len(e), len(f), len(g))

    h = np.setdiff1d(b, a)
    h = np.append(h, np.setdiff1d(c, a))
    h = np.unique(h)
    unique_entity=len(h) + len(a)
    print(unique_entity) # global unique entities


if __name__ == '__main__':
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data(file_path)

    random.seed(12345)

    # integrate train set, valid set, test set
    triples = np.concatenate((train_triplets, valid_triplets), axis=0)
    triples = np.concatenate((triples, test_triplets), axis=0)

    np.random.shuffle(triples)

    client_triples = np.array_split(triples, num_client)

    for idx, val in enumerate(client_triples):
        client_triples[idx] = client_triples[idx].tolist()

    client_data = []

    for client_idx in tqdm(range(num_client)):
        all_triples = client_triples[client_idx]

        triples_reidx = []
        ent_reidx = dict()
        rel_reidx = dict()
        entidx = 0
        relidx = 0

        ent_freq = ddict(int)
        rel_freq = ddict(int)

        # reconstruct indexes
        for tri in all_triples:
            h, r, t = tri
            ent_freq[h] += 1
            ent_freq[t] += 1
            rel_freq[r] += 1

            if h not in ent_reidx.keys():
                ent_reidx[h] = entidx
                entidx += 1
            if t not in ent_reidx.keys():
                ent_reidx[t] = entidx
                entidx += 1
            if r not in rel_reidx.keys():
                rel_reidx[r] = relidx
                relidx += 1
            #  (h,r,t) real indexes  ,  (ent_reidx[h], rel_reidx[r], ent_reidx[t]) reconstructing indexes
            triples_reidx.append([h, r, t, ent_reidx[h], rel_reidx[r], ent_reidx[t]]) # list

        # reconstruct train:valid:test sets at the ratio of 8:1:1
        client_train_triples = []
        client_valid_triples = []
        client_test_triples = []

        random.shuffle(triples_reidx)

        for idx, tri in enumerate(triples_reidx):
            h, r, t, _, _, _ = tri
            # if the frequency of h r t >=2 , classify them into valid and test sets, else train set
            if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
                client_test_triples.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            else:
                client_train_triples.append(tri)

            if len(client_test_triples) > int(len(triples_reidx) * 0.2):
                break

        client_train_triples.extend(triples_reidx[idx + 1:])

        random.shuffle(client_test_triples)
        test_len = len(client_test_triples)

        client_valid_triples = client_test_triples[:int(test_len / 2)]  # valid set
        client_test_triples = client_test_triples[int(test_len / 2):]  # test set

        train_edge_index_ori = np.array(client_train_triples)[:, [0, 2]].T
        train_edge_type_ori = np.array(client_train_triples)[:, 1].T
        train_edge_index = np.array(client_train_triples)[:, [3, 5]].T
        train_edge_type = np.array(client_train_triples)[:, 4].T

        valid_edge_index_ori = np.array(client_valid_triples)[:, [0, 2]].T
        valid_edge_type_ori = np.array(client_valid_triples)[:, 1].T
        valid_edge_index = np.array(client_valid_triples)[:, [3, 5]].T
        valid_edge_type = np.array(client_valid_triples)[:, 4].T

        test_edge_index_ori = np.array(client_test_triples)[:, [0, 2]].T
        test_edge_type_ori = np.array(client_test_triples)[:, 1].T
        test_edge_index = np.array(client_test_triples)[:, [3, 5]].T
        test_edge_type = np.array(client_test_triples)[:, 4].T

        client_data_dict = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type,
                                      'edge_index_ori': train_edge_index_ori, 'edge_type_ori': train_edge_type_ori},
                            'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type,
                                     'edge_index_ori': test_edge_index_ori, 'edge_type_ori': test_edge_type_ori},
                            'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type,
                                      'edge_index_ori': valid_edge_index_ori, 'edge_type_ori': valid_edge_type_ori}}

        client_data.append(client_data_dict)

    # save dataset
    pickle.dump(client_data, open(file_name, 'wb'))

    # Check the statistics of dataset
    # check_datasets(train_triplets,valid_triplets,test_triplets)

