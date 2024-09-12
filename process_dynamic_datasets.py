
import pickle

if __name__ == '__main__':
    num_client=6
    file_name = 'fed_data/dynamic_datasets/FB15K237/FB15K237-Fed' + str(num_client) + '.pkl'

    data_path='fed_data/FB15K237/FB15K237-Fed10.pkl'
    all_data=pickle.load(open(data_path, 'rb'))

    client_data=[]
    for i in range(num_client):
        train_edge_index=all_data[i]['train']['edge_index']
        train_edge_type=all_data[i]['train']['edge_type']
        train_edge_index_ori=all_data[i]['train']['edge_index_ori']
        train_edge_type_ori=all_data[i]['train']['edge_type_ori']

        test_edge_index=all_data[i]['test']['edge_index']
        test_edge_type=all_data[i]['test']['edge_type']
        test_edge_index_ori=all_data[i]['test']['edge_index_ori']
        test_edge_type_ori=all_data[i]['test']['edge_type_ori']

        valid_edge_index=all_data[i]['valid']['edge_index']
        valid_edge_type=all_data[i]['valid']['edge_type']
        valid_edge_index_ori=all_data[i]['valid']['edge_index_ori']
        valid_edge_type_ori=all_data[i]['valid']['edge_type_ori']

        client_data_dict = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type,
                                      'edge_index_ori': train_edge_index_ori, 'edge_type_ori': train_edge_type_ori},
                            'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type,
                                     'edge_index_ori': test_edge_index_ori, 'edge_type_ori': test_edge_type_ori},
                            'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type,
                                      'edge_index_ori': valid_edge_index_ori, 'edge_type_ori': valid_edge_type_ori}}
        client_data.append(client_data_dict)

    pickle.dump(client_data, open(file_name, 'wb'))
