import networkx as nx
from collections import defaultdict
import random
from tqdm import tqdm
import json
import argparse

class UG:
    def __init__(self, addr_kg_train, addr_kg_test, addr_tx, addr_emb, nb_path=10, cutoff=3, using_reverse=True):
        self.using_reverse = using_reverse
        self.nb_path = nb_path
        self.cutoff = cutoff

        self.G_kg, self.e2ind_kg, self.ind2e_kg, self.ep2lst_r_kg = self.create_graph([addr_kg_train], self.using_reverse)
        self.G_tx, self.e2ind_tx, self.ind2e_tx, self.ep2lst_r_tx = self.create_graph([addr_tx], self.using_reverse)
        self.G_hy, self.e2ind_hy, self.ind2e_hy, self.ep2lst_r_hy = self.create_graph([addr_kg_train, addr_tx], self.using_reverse)

        self.dataset = {}
        
        all_w_train, all_r_train, self.all_triple, self.data_train = self.create_path_dataset(addr_kg_train, self.nb_path, self.cutoff)
        all_w_test, all_r_test, _, self.data_test = self.create_path_dataset(addr_kg_test, self.nb_path, self.cutoff)

        all_w = []
        all_w.extend(all_w_train)
        all_w.extend(all_w_test)
        all_w = sorted(list(set(all_w)))
        all_w.append('PADDING')
        word2id = {w:ind for ind, w in enumerate(all_w)}

        all_r = []
        all_r.extend(all_r_train)
        all_r.extend(all_r_test)
        all_r = list(set(all_r))
        all_r = ['NA'] + all_r
        relation2id = {r:ind for ind, r in enumerate(all_r)}
        self.dataset['relation2id'] = relation2id
        
        self.dataset['word2id'] = word2id
        self.dataset['triples'] = self.all_triple
        self.dataset['train'] = self.data_train
        self.dataset['test'] = self.data_test
        
        word2vec = self.load_word_emb(addr_emb)
        self.dataset['word2vec'] = word2vec

    def extract_ug_paths(self, lst_ep, nb_path, cutoff):
        all_data = []
        for h, t in lst_ep:
            data = {}
            paths_kg = self.__search_path('KG', self.G_kg, self.e2ind_kg, self.ind2e_kg, self.ep2lst_r_kg, nb_path,
                                          source=h, target=t, cutoff=cutoff)
            paths_tx = self.__search_path('TX', self.G_tx, self.e2ind_tx, self.ind2e_tx, self.ep2lst_r_tx, nb_path,
                                          source=h, target=t, cutoff=cutoff)
            paths_hy = self.__search_path('HY', self.G_hy, self.e2ind_hy, self.ind2e_hy, self.ep2lst_r_hy, nb_path,
                                          source=h, target=t, cutoff=cutoff)
            if paths_hy != []:
                data['e1_id'] = h
                data['e2_id'] = t
                data['e1_word'] = h
                data['e2_word'] = t
                data['paths'] = [path.split() for path in paths_hy]
                data['path_e1_e2_positions'] = [[0, len(path.split())-1] for path in paths_hy]
                
                if paths_tx != []:
                    for path in paths_tx:
                        data['paths'].append(path.split())
                        data['path_e1_e2_positions'].append([0, len(path.split())-1])

                if paths_kg != []:
                    for path in paths_kg:
                        data['paths'].append(path.split())
                        data['path_e1_e2_positions'].append([0, len(path.split())-1])
                
            else:
                continue
            all_data.append(data)
        return all_data
                    
        
    def load_word_emb(self, addr_emb):
        w2v = {}
        with open(addr_emb, 'r') as fle:
            for line in fle.readlines()[1:]:
                try:
                    contents = line.split()
                    w = contents[0]
                    vec = [float(ele) for ele in contents[1:]]
                except IndexError:
                    continue
                w2v[w] = vec
        return w2v
                    
        
    def create_graph(self, lst_addr, using_reverse):
        G = nx.Graph()
        all_e = set()
        ep2lst_r = defaultdict(list)
        for addr in lst_addr:
            with open(addr, 'r') as fle:
                for line in fle:
                    line = line.strip('\n')
                    try:
                        h, r, t = line.split('\t')[:3]
                    except ValueError:
                        continue

                    ep = (h, t)
                    ep2lst_r[ep].append(r)

                    if using_reverse:
                        ep_ = (t, h)
                        r_ = 'reverse %s' % r
                        ep2lst_r[ep_].append(r_)

                    all_e.add(h)
                    all_e.add(t)

        e2ind = {e:ind for ind, e in enumerate(list(all_e))}
        ind2e = {ind:e for e, ind in e2ind.items()}

        lst_edge = []
        for h, t in ep2lst_r.keys():
            try:
                hi = e2ind[h]
                ti = e2ind[t]
            except KeyError:
                continue

            edge = (hi, ti)
            lst_edge.append(edge)

        G.add_edges_from(lst_edge)
        return G, e2ind, ind2e, ep2lst_r

    def create_path_dataset(self, addr_in, nb_path, cutoff):
        all_w = []
        all_r = set()
        all_triple = []
        all_data = []
        
        with open(addr_in, 'r') as fle:
            lst_line = fle.readlines()
            pbar = tqdm(lst_line, desc="Loading UG:")
            for line in pbar:
                line = line.strip('\n')
                data = {}
                try:
                    h, r, t = line.split('\t')
                except ValueError:
                    continue

                all_r.add(r)
                triple = [h, t, r]
                all_triple.append(triple)
                
                paths_kg = self.__search_path('KG', self.G_kg, self.e2ind_kg, self.ind2e_kg, self.ep2lst_r_kg, nb_path,
                                              source=h, target=t, cutoff=cutoff)
                paths_tx = self.__search_path('TX', self.G_tx, self.e2ind_tx, self.ind2e_tx, self.ep2lst_r_tx, nb_path,
                                              source=h, target=t, cutoff=cutoff)
                paths_hy = self.__search_path('HY', self.G_hy, self.e2ind_hy, self.ind2e_hy, self.ep2lst_r_hy, nb_path,
                                              source=h, target=t, cutoff=cutoff)

                if paths_hy != []:
                    data['e1_id'] = h
                    data['e2_id'] = t
                    data['e1_word'] = h
                    data['e2_word'] = t
                    data['relation'] = r
                    if paths_kg != []:
                        data['kg_paths'] = [path.split() for path in paths_kg]
                        for path in paths_kg:
                            all_w.extend(path.split())
                        data['kg_path_e1_e2_positions'] = [[0, len(path.split())-1] for path in paths_kg]
                    else:
                        data['kg_paths'] = [['PADDING']]
                        data['kg_path_e1_e2_positions'] = [[0, 0]]
                    
                    if paths_tx != []:
                        data['textual_paths'] = [path.split() for path in paths_tx]
                        for path in paths_tx:
                            all_w.extend(path.split())
                        data['textual_path_e1_e2_positions'] = [[0, len(path.split())-1] for path in paths_tx]
                    else:
                        data['textual_paths'] = [['PADDING']]
                        data['textual_path_e1_e2_positions'] = [[0, 0]]

                    data['hybrid_paths'] = [path.split() for path in paths_hy]
                    for path in paths_hy:
                        all_w.extend(path.split())
                    data['hybrid_path_e1_e2_positions'] = [[0, len(path.split())-1] for path in paths_hy]

                else:
                    data = {}

                if data != {}:
                    all_data.append(data)

        all_w = list(set(all_w))
        all_r = list(all_r)
        return all_w, all_r, all_triple, all_data
        

    def __search_path(self, G_tp, G, e2ind, ind2e, ep2lst_r, nb_path, source, target, cutoff=3):
        all_mul_hop = []
        try:
            source_ind = e2ind[source]
            target_ind = e2ind[target]
        except KeyError:
            return all_mul_hop
        
        paths = nx.all_simple_paths(G, source=source_ind, target=target_ind, cutoff=cutoff)
        all_mul_hop = []
        paths = list(paths)
        random.seed(123)
        random.shuffle(paths)
        if paths != []:
            for path in paths[:nb_path]:
                if G_tp in ['KG', 'HY'] and len(path) == 2:
                    continue
                #e.g., path: [1, 10645, 4419, 1825, 4]
                hop_start = '<hop%s>'
                hop_end = '</hop%s>'
                hopi = 1
                hops = [hop_start % hopi, source]
                for i1, e1 in enumerate(path[:-1]):
                    e2 = path[i1+1]
                    try:
                        es1 = ind2e[e1]
                        es2 = ind2e[e2]
                        ep = (es1, es2)
                        lst_r = ep2lst_r[ep]
                    except KeyError:
                        break

                    if lst_r != []:
                        hop = random.sample(lst_r, 1)[0]
                        hops.append(hop)
                        hops.append(es2)
                        hops.append(hop_end % hopi)
                        hopi += 1
                        if i1 + 1 < len(path) - 1:
                            hops.append(hop_start % hopi)
                            hops.append(es2)

                if hops != [hop_start % 1, source]:
                    mul_hop = ' '.join(hops)
                    all_mul_hop.append(mul_hop)

        return all_mul_hop

    
    
