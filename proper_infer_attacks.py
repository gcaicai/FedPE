from wlgk import wlgk
import igraph as ig
import networkx as nx
import numpy as np
import pickle

SMALL_VALUE = 1e-5

def test_wlgk_on_undirected_graphs(sample_graphs):
    G1, G2 = sample_graphs
    similarity = wlgk(G1, G2)

    # ensure it returns a value between 0 and 1 (inclusive)
    assert 0 - SMALL_VALUE <= similarity <= 1 + SMALL_VALUE
    print('similarity=',similarity)

def construct_c0_graph(all_data):
    head_index_ori_arr = all_data[0]['train']['edge_index_ori'][0]
    tail_index_ori_arr = all_data[0]['train']['edge_index_ori'][1]
    edge_index_ori_arr = all_data[0]['train']['edge_type_ori']

    G=nx.Graph()
    for i in range(len(edge_index_ori_arr)):
        h, r, t = head_index_ori_arr[i], edge_index_ori_arr[i], tail_index_ori_arr[i]
        G.add_edge(int(h), int(t), weight=int(r))
    return G

def construct_fedpc_c9_graph(all_data):
    # get global intersection of 10 clients
    head_inter=all_data[0]['test']['edge_index_ori'][0]
    tail_inter=all_data[0]['test']['edge_index_ori'][1]

    for i in range(1,len(all_data)):
        head_index_ori_arr = all_data[i]['test']['edge_index_ori'][0]
        tail_index_ori_arr = all_data[i]['test']['edge_index_ori'][1]
        head_inter=np.intersect1d(head_inter,head_index_ori_arr)
        tail_inter=np.intersect1d(tail_inter,tail_index_ori_arr)

    head_index_ori_arr9 = all_data[9]['test']['edge_index_ori'][0]  # get head entities
    tail_index_ori_arr9 = all_data[9]['test']['edge_index_ori'][1]  # get tail entities
    edge_index_ori_arr9 = all_data[9]['test']['edge_type_ori']  # get edges

    G=nx.Graph()
    for i in range(len(edge_index_ori_arr9)):
        h, r, t = head_index_ori_arr9[i], edge_index_ori_arr9[i], tail_index_ori_arr9[i]
        if h in head_inter and t in tail_inter:
            G.add_edge(int(h), int(t), weight=int(r))
    return G

def construct_global_graph(global_data_path):
    all_data=pickle.load(open(global_data_path, 'rb'))

    head_index_ori_arr0 = all_data[0]['test']['edge_index_ori'][0]
    tail_index_ori_arr0 = all_data[0]['test']['edge_index_ori'][1]
    edge_index_ori_arr0 = all_data[0]['test']['edge_type_ori']

    head_index_ori_arr1 = all_data[1]['test']['edge_index_ori'][0]
    tail_index_ori_arr1 = all_data[1]['test']['edge_index_ori'][1]
    edge_index_ori_arr1 = all_data[1]['test']['edge_type_ori']

    head_index_ori_arr2 = all_data[2]['test']['edge_index_ori'][0]
    tail_index_ori_arr2 = all_data[2]['test']['edge_index_ori'][1]
    edge_index_ori_arr2 = all_data[2]['test']['edge_type_ori']

    head_arr=np.concatenate((head_index_ori_arr0,head_index_ori_arr1,head_index_ori_arr2))
    rel_arr=np.concatenate((edge_index_ori_arr0,edge_index_ori_arr1,edge_index_ori_arr2))
    tail_arr=np.concatenate((tail_index_ori_arr0,tail_index_ori_arr1,tail_index_ori_arr2))

    G=nx.Graph()
    for i in range(len(rel_arr)):
        h, r, t = head_arr[i], rel_arr[i], tail_arr[i]
        G.add_edge(int(h), int(t), weight=int(r))
    return G


if __name__ == '__main__':
    file_path_entity2id='./fed_data/FB15K237/entity2id.txt'
    global_data_path='./fed_data/FB15K237/FB15K237-Fed3.pkl'
    data_path='./fed_data/FB15K237/FB15K237-Fed10.pkl'

    all_data = pickle.load(open(data_path, 'rb'))

    g_c0=construct_c0_graph(all_data)

    g_all=construct_global_graph(global_data_path)

    # take c9 as the adversary
    g_fedpc=construct_fedpc_c9_graph(all_data)

    graphs=(g_fedpc,g_all)

    test_wlgk_on_undirected_graphs(graphs)

'''
c0-c9        similarity= 0.3885427660797244
all-c9       similarity= 0.3650083523148118
'''
