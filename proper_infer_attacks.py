from wlgk import wlgk
import networkx as nx
import numpy as np
import pickle

SMALL_VALUE = 1e-5


def test_wlgk_on_undirected_graphs(sample_graphs):
    G1, G2 = sample_graphs
    similarity = wlgk(G1, G2)

    # ensure it returns a value between 0 and 1 (inclusive)
    assert 0 - SMALL_VALUE <= similarity <= 1 + SMALL_VALUE
    print('similarity =', similarity)


def construct_c0_graph(all_data):
    head_index_arr = all_data[0]['train']['edge_index'][0]
    tail_index_arr = all_data[0]['train']['edge_index'][1]
    edge_index_arr = all_data[0]['train']['edge_type']

    G = nx.Graph()
    for i in range(len(edge_index_arr)):
        h, r, t = head_index_arr[i], edge_index_arr[i], tail_index_arr[i]
        G.add_edge(int(h), int(t), weight=int(r))

    return G


def construct_fedpe_c9_graph(all_data):
    # get global intersection of 10 clients
    head_inter = all_data[0]['test']['edge_index'][0]
    tail_inter = all_data[0]['test']['edge_index'][1]

    for i in range(1, len(all_data)):
        head_index_ori_arr = all_data[i]['test']['edge_index'][0]
        tail_index_ori_arr = all_data[i]['test']['edge_index'][1]
        head_inter = np.intersect1d(head_inter, head_index_ori_arr)
        tail_inter = np.intersect1d(tail_inter, tail_index_ori_arr)

    head_index_ori_arr9 = all_data[9]['test']['edge_index'][0]
    tail_index_ori_arr9 = all_data[9]['test']['edge_index'][1]
    edge_index_ori_arr9 = all_data[9]['test']['edge_type']

    G = nx.Graph()
    for i in range(len(edge_index_ori_arr9)):
        h, r, t = head_index_ori_arr9[i], edge_index_ori_arr9[i], tail_index_ori_arr9[i]
        if h in head_inter and t in tail_inter:
            G.add_edge(int(h), int(t), weight=int(r))
    return G


def construct_global_graph(data):
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

    G = nx.Graph()
    for tri in all_triples:
        h, r, t = tri[0], tri[1], tri[2]
        G.add_edge(int(h), int(t), weight=int(r))

    return G


if __name__ == '__main__':
    data_path = './fed_data/FB15K237/FB15K237-Fed10.pkl'
    all_data = pickle.load(open(data_path, 'rb'))

    g_all = construct_global_graph(all_data)

    g_c0 = construct_c0_graph(all_data)

    # take c9 as the adversary
    g_c9 = construct_fedpe_c9_graph(all_data)

    graphs = (g_c0, g_c9)  # FedFE
    # graphs = (g_all, g_c9)

    test_wlgk_on_undirected_graphs(graphs)


''' FedPE
c0-c9        similarity = 0.3885427660797244  # 0.37855036358548644
all-c9       similarity = 0.3650083523148118  # 0.07143361219033542
'''
