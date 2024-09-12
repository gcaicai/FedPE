# import torch
import pickle
import numpy as np
import random
from scipy import spatial

random.seed(10)
np.random.seed(10)
p = 1

def calculate_fedpc(data, victim_id, adver_id):
    # ERR
    syn_ent_list = []
    victim_ent = np.unique(data[victim_id]['train']['edge_index_ori'])
    adver_ent = np.unique(data[adver_id]['train']['edge_index_ori'])

    for i in range(len(victim_ent)):
        syn_ent_list.append(random.choice(adver_ent))

    err = sum(first == second for (first, second) in zip(syn_ent_list, victim_ent)) / len(victim_ent)
    print('err=', err)

    # RRR
    syn_rel_list = []
    victim_rel = np.unique(data[victim_id]['train']['edge_type_ori'])
    adver_rel = np.unique(data[adver_id]['train']['edge_type_ori'])

    for i in range(len(victim_rel)):
        syn_rel_list.append(random.choice(adver_rel))

    rrr = sum(first == second for (first, second) in zip(syn_rel_list, victim_rel)) / len(victim_rel)
    print('rrr=', rrr)

    # TRR
    adver_triple_all = np.array([data[adver_id]['train']['edge_index_ori'][0],
                                 data[adver_id]['train']['edge_type_ori'],
                                 data[adver_id]['train']['edge_index_ori'][1]])  # client1 训练集 (h,r,t) 真实索引
    len_adver_triple = len(adver_triple_all[1])
    len_victim_triple = len(data[victim_id]['train']['edge_type_ori'])
    victim_triple_all = []
    for i in range(len_victim_triple):
        h = data[victim_id]['train']['edge_index_ori'][0][i]
        r = data[victim_id]['train']['edge_type_ori'][i]
        t = data[victim_id]['train']['edge_index_ori'][1][i]
        victim_triple_all.append([h, r, t])

    syn_trr_list = []

    # 对于 h t 属于 敌手，推断二者之间的关系 r'，推断成功的概率
    #
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]  # 取出 client1 (h r t) 真实索引 的第 i条 (h r t)
        h, t = triple[0], triple[2]
        for j in range(len(adver_rel)):
            rx = adver_rel[j]
            if [h, rx, t] in victim_triple_all:
                syn_trr_list.append([h, rx, t])

    trr = len(syn_trr_list) / len_victim_triple
    print('trr=', trr)

    # lack head
    syn_lack_head_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        r, t = triple[1], triple[2]
        for j in range(len(adver_ent)):
            hx = adver_ent[j]
            if [hx, r, t] in victim_triple_all:
                syn_lack_head_list.append([hx, r, t])

    lhr = len(syn_lack_head_list) / len_victim_triple
    print('lhr=', lhr)

    # lack relation / TRR
    print('lrr=', trr)

    # lack tail
    syn_lack_tail_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        h, r = triple[0], triple[1]
        for j in range(len(adver_ent)):
            tx = adver_ent[j]
            if [h, r, tx] in victim_triple_all:
                syn_lack_tail_list.append([h, r, tx])

    ltr = len(syn_lack_tail_list) / len_victim_triple
    print('ltr=', ltr)


def calculate_fedr(emb,data,victim_id,adver_id):
    # the adversary knows global relations
    # ERR
    ent_embed = emb['ent_embed']
    rel_embed = emb['rel_embed']

    victim_ent = np.unique(data[victim_id]['train']['edge_index'])
    victim_ent_embed_dict = {}
    value = ent_embed[victim_id]
    for idx,ent in enumerate(victim_ent):
        victim_ent_embed_dict[ent] = value[idx]

    victim_mapping = dict(zip(data[victim_id]['train']['edge_index'][0], data[victim_id]['train']['edge_index_ori'][0]))
    victim_mapping.update(dict(zip(data[victim_id]['train']['edge_index'][1], data[victim_id]['train']['edge_index_ori'][1])))

    victim_ent_embed_dict_mapped = dict((victim_mapping[key], value) for (key, value) in victim_ent_embed_dict.items())
    victim_ent_pool_mapped = [victim_mapping[i] for i in victim_ent]

    adver_mapping = dict(zip(data[adver_id]['train']['edge_index'][0], data[adver_id]['train']['edge_index_ori'][0]))
    adver_mapping.update(dict(zip(data[adver_id]['train']['edge_index'][1], data[adver_id]['train']['edge_index_ori'][1])))

    adver_ent = np.unique(data[adver_id]['train']['edge_index'])

    adver_ent_pool = np.random.choice(adver_ent, int(p * len(adver_ent)), replace=False)

    syn_ent_list = []
    for i in adver_ent_pool:
        adver_ent_embed = ent_embed[adver_id][i]
        count = 0
        loss_bound = 0
        ent_idx = []

        for j in victim_ent_embed_dict_mapped:
            loss = spatial.distance.cosine(adver_ent_embed.detach().numpy(), victim_ent_embed_dict_mapped[j].detach().numpy())
            if count == 0:  # first round
                loss_bound = loss
                ent_idx.append(j)
                count += 1
            else:
                if loss < loss_bound:
                    loss_bound = loss
                    ent_idx.append(j)
        syn_ent_list.append(ent_idx[-1]) # global index of the entity

    tru_ent_list = [adver_mapping[i] for i in adver_ent_pool]

    # calculate the number of correct reconstruction
    err=sum(first == second for (first, second) in zip(syn_ent_list, tru_ent_list)) / len(adver_ent)
    print('err=',err)

    # RRR
    rrr=1.0
    print('rrr=',rrr)

    # TRR
    adver_rel=np.unique(data[adver_id]['train']['edge_type_ori'])
    adver_rel_embed_dict = {}
    for idx,rel in enumerate(adver_rel):
        adver_rel_embed_dict[rel]=rel_embed[idx]

    syn_rel_list = []
    for i in adver_rel:
        adver_rel_embed = adver_rel_embed_dict[i]
        count = 0
        loss_bound = 0
        rel_idx = []
        for j in adver_rel:
            if i != j:
                victim_rel_embed = adver_rel_embed_dict[j]
                loss = spatial.distance.cosine(adver_rel_embed.detach().numpy(), victim_rel_embed.detach().numpy())
                if count == 0:  # first round
                    loss_bound = loss
                    rel_idx.append(j)
                    count += 1
                else:
                    if loss < loss_bound:
                        loss_bound = loss
                        rel_idx.append(j)
        syn_rel_list.append(rel_idx[-1])  # global index of the entity

    adver_triple_all = np.array([data[adver_id]['train']['edge_index_ori'][0],
                              data[adver_id]['train']['edge_type_ori'],
                              data[adver_id]['train']['edge_index_ori'][1]])  # client1 训练集 (h,r,t) 真实索引
    len_adver_triple = len(adver_triple_all[1])
    len_victim_triple=len(data[victim_id]['train']['edge_type_ori'])
    victim_triple_all = []
    for i in range(len_victim_triple):
        h = data[victim_id]['train']['edge_index_ori'][0][i]
        r = data[victim_id]['train']['edge_type_ori'][i]
        t = data[victim_id]['train']['edge_index_ori'][1][i]
        victim_triple_all.append([h,r,t])

    syn_trr_list = []

    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]  # 取出 client1 (h r t) 真实索引 的第 i条 (h r t)
        h, r, t = triple[0], triple[1], triple[2]
        rx = syn_rel_list[r]
        if [h,rx,t] in victim_triple_all:
            syn_trr_list.append([h,rx,t])

    trr = len(syn_trr_list) / len_victim_triple
    print('trr=',trr)

    # lack head
    syn_lack_head_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        h, r, t = triple[0], triple[1], triple[2]
        idx = tru_ent_list.index(h)
        hx = syn_ent_list[idx]
        if [hx,r,t] in victim_triple_all:
            syn_lack_head_list.append([hx,r,t])

    lhr = len(syn_lack_head_list) / len_victim_triple
    print('lhr=',lhr)

    # lack relation / TRR
    print('lrr=',trr)

    # lack tail
    syn_lack_tail_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        h, r, t = triple[0], triple[1], triple[2]
        idx = tru_ent_list.index(t)
        tx = syn_ent_list[idx]
        if [h,r,tx] in victim_triple_all:
            syn_lack_tail_list.append([h,r,tx])

    ltr = len(syn_lack_tail_list) / len_victim_triple
    print('ltr=',ltr)


def calculate_fedm(data, victim_id, adver_id):
    # the adversary knows global entities and relations
    # ERR
    adver_ent = np.unique(data[adver_id]['train']['edge_index_ori'])
    victim_ent = np.unique(data[victim_id]['train']['edge_index_ori'])
    ent_inter = np.intersect1d(adver_ent, victim_ent)
    err = len(ent_inter) / len(victim_ent)
    print('err=', err)

    # RRR
    adver_rel = np.unique(data[adver_id]['train']['edge_type_ori'])
    victim_rel = np.unique(data[victim_id]['train']['edge_type_ori'])
    rel_inter = np.intersect1d(adver_rel, victim_rel)
    rrr = len(rel_inter) / len(victim_rel)
    print('rrr=', rrr)

    # TRR
    adver_triple_all = np.array([data[adver_id]['train']['edge_index_ori'][0],
                                 data[adver_id]['train']['edge_type_ori'],
                                 data[adver_id]['train']['edge_index_ori'][1]])
    len_adver_triple = len(adver_triple_all[1])
    len_victim_triple = len(data[victim_id]['train']['edge_type_ori'])
    victim_triple_all = []
    for i in range(len_victim_triple):
        h = data[victim_id]['train']['edge_index_ori'][0][i]
        r = data[victim_id]['train']['edge_type_ori'][i]
        t = data[victim_id]['train']['edge_index_ori'][1][i]
        victim_triple_all.append([h, r, t])

    syn_trr_list = []

    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        h, t = triple[0], triple[2]
        if h in ent_inter and t in ent_inter:
            for j in range(len(rel_inter)):
                rx = rel_inter[j]
                if [h, rx, t] in victim_triple_all:
                    syn_trr_list.append([h, rx, t])

    trr = len(syn_trr_list) / len_victim_triple
    print('trr=', trr)

    # lack head
    syn_lack_head_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]  # 取出 client1 (h r t) 真实索引 的第 i条 (h r t)
        r, t = triple[1], triple[2]
        if r in rel_inter and t in ent_inter:
            for j in range(len(ent_inter)):
                hx = ent_inter[j]
                if [hx, r, t] in victim_triple_all:
                    syn_lack_head_list.append([hx, r, t])

    lhr = len(syn_lack_head_list) / len_victim_triple
    print('lhr=', lhr)

    # lack relation / TRR
    print('lrr=', trr)

    # lack tail
    syn_lack_tail_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]  # 取出 client1 (h r t) 真实索引 的第 i条 (h r t)
        h, r = triple[0], triple[1]
        if h in ent_inter and r in rel_inter:
            for j in range(len(ent_inter)):
                tx = ent_inter[j]
                if [h, r, tx] in victim_triple_all:
                    syn_lack_tail_list.append([h, r, tx])

    ltr = len(syn_lack_tail_list) / len_victim_triple
    print('ltr=', ltr)


def calculate_fedcke(data, victim_id, adver_id):
    # the adversary knows the union set of each two client's intersection about global entities and realtions
    # ERR
    client_num = len(data)

    ent_union = np.array([], dtype=int)
    rel_union = np.array([], dtype=int)

    for i in range(client_num):
        for j in range(i + 1, client_num):
            # intersection --> unionset
            entity_i = np.unique(data[i]['train']['edge_index_ori'])
            entity_j = np.unique(data[j]['train']['edge_index_ori'])
            relation_i = np.unique(data[i]['train']['edge_type_ori'])
            relation_j = np.unique(data[j]['train']['edge_type_ori'])

            entity_inter = np.intersect1d(entity_i, entity_j)  # 默认升序排列
            relation_inter = np.intersect1d(relation_i, relation_j)

            ent_union = np.union1d(ent_union, entity_inter)
            rel_union = np.union1d(rel_union, relation_inter)

    adver_ent = np.unique(data[adver_id]['train']['edge_index_ori'])
    victim_ent = np.unique(data[victim_id]['train']['edge_index_ori'])

    adver_rel = np.unique(data[adver_id]['train']['edge_type_ori'])
    victim_rel = np.unique(data[victim_id]['train']['edge_type_ori'])

    ent_inter = np.intersect1d(victim_ent, ent_union) # adversary knowledge

    err = len(ent_inter) / len(victim_ent)
    print('err=', err)

    # RRR
    rel_inter = np.intersect1d(victim_rel, rel_union) # adversary knowledge
    rrr = len(rel_inter) / len(victim_rel)
    print('rrr=', rrr)

    # TRR
    adver_triple_all = np.array([data[adver_id]['train']['edge_index_ori'][0],
                                 data[adver_id]['train']['edge_type_ori'],
                                 data[adver_id]['train']['edge_index_ori'][1]])
    len_adver_triple = len(adver_triple_all[1])
    len_victim_triple = len(data[victim_id]['train']['edge_type_ori'])
    victim_triple_all = []
    for i in range(len_victim_triple):
        h = data[victim_id]['train']['edge_index_ori'][0][i]
        r = data[victim_id]['train']['edge_type_ori'][i]
        t = data[victim_id]['train']['edge_index_ori'][1][i]
        victim_triple_all.append([h, r, t])

    syn_trr_list = []

    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        h, t = triple[0], triple[2]
        if h in ent_union and t in ent_union:
            for j in range(len(rel_inter)):
                rx = rel_inter[j]
                if [h, rx, t] in victim_triple_all:
                    syn_trr_list.append([h, rx, t])
                    print([h, rx, t])

    trr = len(syn_trr_list) / len_victim_triple
    print('trr=', trr)

    # lack head
    syn_lack_head_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        r, t = triple[1], triple[2]
        if r in rel_union and t in ent_union:
            for j in range(len(ent_inter)):
                hx = ent_inter[j]
                if [hx, r, t] in victim_triple_all:
                    syn_lack_head_list.append([hx, r, t])
                    print([hx, r, t])

    lhr = len(syn_lack_head_list) / len_victim_triple
    print('lhr=', lhr)

    # lack relation / TRR
    print('lrr=', trr)

    # lack tail
    syn_lack_tail_list = []
    for i in range(len_adver_triple):
        triple = adver_triple_all[:, i]
        h, r = triple[0], triple[1]
        if h in ent_union and r in rel_union:
            for j in range(len(ent_inter)):
                tx = ent_inter[j]
                if [h, r, tx] in victim_triple_all:
                    syn_lack_tail_list.append([h, r, tx])
                    print([h, r, tx])

    ltr = len(syn_lack_tail_list) / len_victim_triple
    print('ltr=', ltr)


if __name__ == '__main__':
    # emb_best_path = './state_new/fb15k237-fed3_fedcke_transe.best'
    # emb = torch.load(emb_best_path, map_location=torch.device('cpu'))

    data_path = 'fed_data/FB15K237/FB15K237-Fed3.pkl'
    data = pickle.load(open(data_path, 'rb'))

    victim_id = 0
    adver_id = 2

    # fedpc
    calculate_fedpc(data, victim_id, adver_id)

    # fedr
    # calculate_fedr(emb, data, victim_id, adver_id)

    # fedm
    # calculate_fedm(data, victim_id, adver_id)

    # fedcke
    # calculate_fedcke(data, victim_id, adver_id)


