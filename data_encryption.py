import numpy as np
import pickle
import copy

from pypbc import *
import hashlib
import unittest

Hash = hashlib.sha512

stored_params = """type a
q 8780710799663312522437781984754049815806883199414208211028653399266475630880222957078625179422662221423155858769582317459277713367317481324925129998224791
h 12016012264891146079388821366740534204802954401251311822919615131047207289359704531102844802183906537786776
r 730750818665451621361119245571504901405976559617
exp2 159
exp1 107
sign1 1
sign0 1
"""

# initialize public parameters
params = Parameters(param_string=stored_params)  # type a
pairing = Pairing(params)

g = Element.random(pairing, G1)  # random select generator g


class PKG(object):
    def __init__(self):
        self.__sk = []
        self.__pk = []
        self.__aggPK = []
        self.__aggPK_all = []

    def static(self, client_num):
        sk_lst = []
        pk_lst = []
        aggPK_lst = []

        # generate private keys and public keys
        for i in range(client_num):
            ski = Element.random(pairing, Zr)  # 私钥是一个素数域 Zp内的随机数
            pki = Element(pairing, G1, value=g ** ski)  # 公钥 pk = g^sk
            sk_lst.append(ski)
            pk_lst.append(pki)

        # calculate aggregated keys
        for i in range(client_num):
            aggPKi = g
            for j in range(client_num):
                if i != j:
                    aggPKi = Element(pairing, G1, value=aggPKi ** sk_lst[j])
            aggPK_lst.append(aggPKi)

        self.__sk = sk_lst
        self.__pk = pk_lst
        self.__aggPK = aggPK_lst
        self.__aggPK_all = Element(pairing, G1, value=self.__aggPK[0] ** self.__sk[0])
        return pk_lst, aggPK_lst

    def dynamic(self):
        # generate private keys and public keys
        ski = Element.random(pairing, Zr)  # random select a prime number
        pki = Element(pairing, G1, value=g ** ski)  # pk = g^sk
        rdi = Element.random(pairing, Zr)
        gammai = Element(pairing, G1, value=ski * rdi)  # gamma = sk * rd

        self.__sk.append(ski)
        self.__pk.append(pki)

        # calculate aggregated keys
        aggPKi = Element(pairing, G1, value=self.__aggPK_all * rd)  # g^((sk1*sk2*sk3)*rd4)
        self.__aggPK.append(aggPKi)

        self.__aggPK_all = Element(pairing, G1, value=self.__aggPK_all * gammai)

        return pki, aggPKi, gammai

class Server(object):
    def __init__(self):
        pass

    # encrypt data after new client joining
    def process_data(self, client, gammai):
        ent_enc_data, rel_enc_data = client.get_enc_data()
        # recalculate e(x,y)^gammi
        ent_enc_data_new = [data ** gammai for idx, data in enumerate(ent_enc_data)]
        rel_enc_data_new = [data ** gammai for idx, data in enumerate(rel_enc_data)]

        client.update_enc_data(ent_enc_data, rel_enc_data, ent_enc_data_new, rel_enc_data_new)


class Client(object):
    def __init__(self):
        self.ent2id_enc_all = []
        self.rel2id_enc_all = []

    def get_enc_data(self):
        ent_enc_data = copy.deepcopy(self.ent2id_enc_all)
        rel_enc_data = copy.deepcopy(self.rel2id_enc_all)
        ent_enc_data.pop(len(ent_enc_data)-1)
        rel_enc_data.pop(len(rel_enc_data)-1)
        ent_enc_data = list(np.unique(ent_enc_data))
        rel_enc_data = list(np.unique(rel_enc_data))

        return ent_enc_data, rel_enc_data

    def update_enc_data(self, ent_enc_data, rel_enc_data, ent_enc_data_new, rel_enc_data_new):
        for i in range(len(self.ent2id_enc_all)-1):
            for j in range(len(ent_enc_data)):
                if ent_enc_data[j] in self.ent2id_enc_all[i]:
                    idx = self.ent2id_enc_all[i].index(ent_enc_data[j])
                    self.ent2id_enc_all[i][idx] = ent_enc_data_new[j]

        for i in range(len(self.rel2id_enc_all)-1):
            for j in range(len(rel_enc_data)):
                if rel_enc_data[j] in self.rel2id_enc_all[i]:
                    idx = self.rel2id_enc_all[i].index(rel_enc_data[j])
                    self.rel2id_enc_all[i][idx] = rel_enc_data_new[j]

    # both suitable for static client and dynamic client
    def process_data(self, client_id, pki, aggPKi):

        data_path = 'fed_data/FB15K237/FB15K237-Fed10.pkl'  # need to modify the file path
        all_data = pickle.load(open(data_path, 'rb'))

        ent_unique = np.unique(np.concatenate((
            all_data[client_id]['train']['edge_index_ori'],
            all_data[client_id]['test']['edge_index_ori'],
            all_data[client_id]['valid']['edge_index_ori']
        )))

        rel_unique = np.unique(np.concatenate((
            all_data[client_id]['train']['edge_type_ori'],
            all_data[client_id]['test']['edge_type_ori'],
            all_data[client_id]['valid']['edge_type_ori']
        )))

        ent2id_lst = []
        rel2id_lst = []

        ent_file_path = 'fed_data/FB15K237/entity2id.txt'  # need to modify the file path
        rel_file_path = 'fed_data/FB15K237/relation2id.txt'  # need to modify the file path
        with open(ent_file_path) as f:
            for line in f:
                ent, idx = line.strip().split('\t')
                ent2id_lst.append(ent)

        with open(rel_file_path) as f:
            for line in f:
                rel, idx = line.strip().split('\t')
                rel2id_lst.append(rel)

        ent2id_enc_lst = []
        rel2id_enc_lst = []

        for i in range(len(ent2id_lst)):
            if ent2id_lst[i] in ent_unique:
                # if entities of client i exist, then encrypting
                enc_left = Element.from_hash(pairing, G1, value=pki ** Hash(str(ent2id_lst[i]).encode('utf-8')).hexdigest())
                enc = pairing.apply(enc_left, aggPKi)
                ent2id_enc_lst.append(enc)
            else:
                ent2id_enc_lst.append(-1)

        for i in range(len(rel2id_lst)):
            if rel2id_lst[i] in rel_unique:
                # if relations of client i exist, then encrypting
                enc_left = Element.from_hash(pairing, G1, value=pki ** Hash(str(rel2id_lst[i]).encode('utf-8')).hexdigest())
                enc = pairing.apply(enc_left, aggPKi)
                rel2id_enc_lst.append(enc)
            else:
                rel2id_enc_lst.append(-1)

        self.ent2id_enc_all.append(ent2id_enc_lst)
        self.rel2id_enc_all.append(rel2id_enc_lst)

    # after all client's encryption
    def merge_data(self):
        ent2id_enc_lst = self.ent2id_enc_all[0]
        rel2id_enc_lst = self.rel2id_enc_all[0]

        # process entity and relation data
        for i in range(1, len(self.ent2id_enc_all)):
            for j in range(len(ent2id_enc_lst)):
                ent2id_enc_lst[j].append(self.ent2id_enc_all[i][j])

        for i in range(1, len(self.rel2id_enc_all)):
            for j in range(len(rel2id_enc_lst)):
                rel2id_enc_lst[j].append(self.rel2id_enc_all[i][j])

        for i in range(len(ent2id_enc_lst)):
            for j, ele in enumerate(ent2id_enc_lst[i]):
                if ele == -1:
                    ent2id_enc_lst[i].remove(-1)

            ent2id_enc_lst[i] = list(np.unique(ent2id_enc_lst[i]))
            ent2id_enc_str = ''

            for j in range(len(ent2id_enc_lst[i])):
                ent2id_enc_str = ent2id_enc_str+str(ent2id_enc_lst[i][j]) + '\t'
            ent2id_enc_str = ent2id_enc_str+str(i) + '\n'
            ent2id_enc_lst[i] = ent2id_enc_str

        for i in range(len(rel2id_enc_lst)):
            for j,ele in enumerate(rel2id_enc_lst[i]):
                if ele == -1:
                    rel2id_enc_lst[i].remove(-1)

            rel2id_enc_lst[i] = list(np.unique(rel2id_enc_lst[i]))
            rel2id_enc_str = ''

            for j in range(len(rel2id_enc_lst[i])):
                rel2id_enc_str = rel2id_enc_str+str(rel2id_enc_lst[i][j]) + '\t'
            rel2id_enc_str = rel2id_enc_str+str(i) + '\n'
            rel2id_enc_lst[i] = rel2id_enc_str

        # write files
        ent_enc_file_path = 'fed_data/FB15K237/entity2idEnc.txt'
        rel_enc_file_path = 'fed_data/FB15K237/relation2idEnc.txt'

        with open(ent_enc_file_path, 'w') as f:
            for item in range(len(ent2id_enc_lst)):
                f.write(ent2id_enc_lst[item])

        with open(rel_enc_file_path, 'w') as f:
            for item in range(len(rel2id_enc_lst)):
                f.write(rel2id_enc_lst[item])


'''Note:
    different keys and generator g lead to different encrypted results, 
    but this factor will not affect the method, details and analysis can be found in FedPE.
    pypbc library: https://github.com/debatem1/pypbc
'''
if __name__ == '__main__':

    pkg = PKG()
    pk_lst, aggPK_lst = pkg.static(5)  # default there are 5 clients, generate initial keys information

    server = Server()

    client = Client()
    # encrypt data for 5 clients, you can execute this process step-by-step
    for i in range(5):
        client.process_data(i, pk_lst[i], aggPK_lst[i])

    # client.merge_data() # if there is no more client joining

    # if there are new clients
    for i in range(5):
        # for one client
        client_id = 5+i

        pki, aggPKi, gammai = pkg.dynamic()

        client.process_data(client_id, pki, aggPKi)
        server.process_data(client, gammai)

    client.merge_data()
