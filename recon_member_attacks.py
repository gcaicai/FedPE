import pickle
import numpy as np
import random

random.seed(10)
np.random.seed(10)
p = 1


def calculate_fedpe(data, victim_id, adver_id):
    # ERR
    syn_ent_list = []
    victim_ent = np.unique(data[victim_id]['train']['edge_index'])
    adver_ent = np.unique(data[adver_id]['train']['edge_index'])

    for i in range(len(victim_ent)):
        syn_ent_list.append(random.choice(adver_ent))

    err = sum(first == second for (first, second) in zip(syn_ent_list, victim_ent)) / len(victim_ent)
    print('err =', err)

    # RRR
    syn_rel_list = []
    victim_rel = np.unique(data[victim_id]['train']['edge_type'])
    adver_rel = np.unique(data[adver_id]['train']['edge_type'])

    for i in range(len(victim_rel)):
        syn_rel_list.append(random.choice(adver_rel))

    rrr = sum(first == second for (first, second) in zip(syn_rel_list, victim_rel)) / len(victim_rel)
    print('rrr =', rrr)

    # TRR
    adver_triple_all = np.array([data[adver_id]['train']['edge_index'][0],
                                 data[adver_id]['train']['edge_type'],
                                 data[adver_id]['train']['edge_index'][1]])
    len_adver_triple = len(adver_triple_all[1])
    len_victim_triple = len(data[victim_id]['train']['edge_type'])
    victim_triple_all = []
    for i in range(len_victim_triple):
        h = data[victim_id]['train']['edge_index'][0][i]
        r = data[victim_id]['train']['edge_type'][i]
        t = data[victim_id]['train']['edge_index'][1][i]
        victim_triple_all.append([h, r, t])

    trr_list = []
    lack_head_list = []
    lack_tail_list = []

    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        h, r, t = triple[0], triple[1], triple[2]
        for j in range(len(adver_rel)):
            trr_list.append([h, adver_rel[j], t])
            lack_head_list.append([adver_ent[j], r, t])
            lack_tail_list.append([h, r, adver_ent[j]])

    syn_trr_list = set(tuple(row) for row in trr_list) & set(tuple(row) for row in victim_triple_all)
    syn_lack_head_list = set(tuple(row) for row in lack_head_list) & set(tuple(row) for row in victim_triple_all)
    syn_lack_tail_list = set(tuple(row) for row in lack_tail_list) & set(tuple(row) for row in victim_triple_all)

    trr = len(syn_trr_list) / len_victim_triple
    print('trr =', trr)

    lhr = len(syn_lack_head_list) / len_victim_triple
    print('lhr =', lhr)

    ltr = len(syn_lack_tail_list) / len_victim_triple
    print('ltr =', ltr)


if __name__ == '__main__':
    data_path = 'fed_data/FB15K237/FB15K237-Fed10.pkl'
    all_data = pickle.load(open(data_path, 'rb'))

    victim_id = 0
    adver_id = 9

    # fedpe
    calculate_fedpe(all_data, victim_id, adver_id)

