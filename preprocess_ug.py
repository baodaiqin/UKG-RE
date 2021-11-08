import json
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from time import time
from collections import defaultdict
import random
from load_ug import UG

import settings as conf

def pos_embed(x, maxlen):
    return max(0, min(x + maxlen, maxlen + maxlen + 1))

def indexing_paths(paths, posis, total, fixlen, word2id, relation2id):
    paths_label = np.zeros((total), dtype = np.int32)
    paths_head = np.zeros((total), dtype = np.int32)
    paths_tail = np.zeros((total), dtype = np.int32)
    paths_word = np.zeros((total, fixlen), dtype = np.int32)
    paths_posi1 = np.zeros((total, fixlen), dtype = np.int32)
    paths_posi2 = np.zeros((total, fixlen), dtype = np.int32)
    instance_triple = []
    instance_scope = []

    pbar = tqdm(paths.items())
    pathi = -1
    for triple, lst_path in pbar:
        ei1, ei2, rel = triple
        lst_posi = posis[triple]

        headi = word2id[ei1]
        taili = word2id[ei2]
        try:
            reli = relation2id[rel]
        except KeyError:
            reli = relation2id['NA']
        
        for path, posi in zip(lst_path, lst_posi):
            e1posi, e2posi = posi
            pathi += 1
            
            paths_label[pathi] = reli
            paths_head[pathi] = headi
            paths_tail[pathi] = taili
            
            if instance_triple == [] or instance_triple[len(instance_triple) - 1] != triple:
                instance_triple.append(triple)
                instance_scope.append([pathi,pathi])
            instance_scope[len(instance_triple) - 1][1] = pathi
            
            for i in range(fixlen):
                paths_word[pathi][i] = word2id['BLANK']
                paths_posi1[pathi][i] = pos_embed(i - e1posi, fixlen)
                paths_posi2[pathi][i] = pos_embed(i - e2posi, fixlen)
                
            for i, word in enumerate(path):
                if i >= fixlen:
                    break
                elif not word in word2id:
                    paths_word[pathi][i] = word2id['UNK']
                else:
                    paths_word[pathi][i] = word2id[word]

    return paths_head, paths_tail, paths_label, paths_word, paths_posi1, paths_posi2, np.array(instance_scope)

def preprocess_ug_paths(bags, dir_out, fixlen, word2id, relation2id):
    instance_scope = []
    instance_scope_kg = []
    instance_scope_tx = []
    instance_scope_hy = []

    paths = defaultdict(list)
    paths_kg = defaultdict(list)
    paths_tx = defaultdict(list)
    paths_hy = defaultdict(list)

    posis = defaultdict(list)
    posis_kg = defaultdict(list)
    posis_tx = defaultdict(list)
    posis_hy = defaultdict(list)

    total = 0
    total_kg = 0
    total_tx = 0
    total_hy = 0
    
    for bag in bags:
        e1_id = bag['e1_id']
        e2_id = bag['e2_id']
        e1_word = bag['e1_word']
        e2_word = bag['e2_word']
        relation = bag['relation']

        triple = (e1_id, e2_id, relation)
        
        try:
            kg_paths = bag['kg_paths']
            kg_paths_e1_e2_posi = bag['kg_path_e1_e2_positions']
        except KeyError:
            kg_paths = [['PADDING']]
            kg_paths_e1_e2_posi = [[0, 0]]
            
        try:
            tx_paths = bag['textual_paths']
            tx_paths_e1_e2_posi = bag['textual_path_e1_e2_positions']
        except KeyError:
            tx_paths = [['PADDING']]
            tx_paths_e1_e2_posi = [[0, 0]]
            
        try:
            hy_paths = bag['hybrid_paths']
            hy_paths_e1_e2_posi = bag['hybrid_path_e1_e2_positions']
        except KeyError:
            hy_paths = [['PADDING']]
            hy_paths_e1_e2_posi = [[0, 0]]

        paths[triple].extend(kg_paths)
        paths[triple].extend(tx_paths)
        paths[triple].extend(hy_paths)
        nb_paths = len(kg_paths) + len(tx_paths) + len(hy_paths)
        total += nb_paths
        
        paths_kg[triple].extend(kg_paths)
        paths_tx[triple].extend(tx_paths)
        paths_hy[triple].extend(hy_paths)
        total_kg += len(kg_paths)
        total_tx += len(tx_paths)
        total_hy += len(hy_paths)

        posis[triple].extend(kg_paths_e1_e2_posi)
        posis[triple].extend(tx_paths_e1_e2_posi)
        posis[triple].extend(hy_paths_e1_e2_posi)

        posis_kg[triple].extend(kg_paths_e1_e2_posi)
        posis_tx[triple].extend(tx_paths_e1_e2_posi)
        posis_hy[triple].extend(hy_paths_e1_e2_posi)

    print('Indexing ug paths ...:')
    paths_head, paths_tail, paths_label, paths_word, paths_posi1, paths_posi2, instance_scope = indexing_paths(paths, posis, total, fixlen, word2id, relation2id)
    print('Indexing kg paths ...:')
    paths_head_kg, paths_tail_kg, paths_label_kg, paths_word_kg, paths_posi1_kg, paths_posi2_kg, instance_scope_kg = indexing_paths(paths_kg, posis_kg, total_kg, fixlen, word2id, relation2id)
    print('Indexing textual paths ...:')
    paths_head_tx, paths_tail_tx, paths_label_tx, paths_word_tx, paths_posi1_tx, paths_posi2_tx, instance_scope_tx = indexing_paths(paths_tx, posis_tx, total_tx, fixlen, word2id, relation2id)
    print('Indexing hybrid paths ...:')
    paths_head_hy, paths_tail_hy, paths_label_hy, paths_word_hy, paths_posi1_hy, paths_posi2_hy, instance_scope_hy = indexing_paths(paths_hy, posis_hy, total_hy, fixlen, word2id, relation2id)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    np.save(dir_out + '/' + 'head', paths_head)
    np.save(dir_out + '/' + 'tail', paths_tail)
    np.save(dir_out + '/' + 'label', paths_label)
    np.save(dir_out + '/' + 'word', paths_word)
    np.save(dir_out + '/' + 'posi1', paths_posi1)
    np.save(dir_out + '/' + 'posi2', paths_posi2)
    np.save(dir_out + '/' + 'scope', instance_scope)

    np.save(dir_out + '/' + 'head_kg', paths_head_kg)
    np.save(dir_out + '/' + 'tail_kg', paths_tail_kg)
    np.save(dir_out + '/' + 'label_kg', paths_label_kg)
    np.save(dir_out + '/' + 'word_kg', paths_word_kg)
    np.save(dir_out + '/' + 'posi1_kg', paths_posi1_kg)
    np.save(dir_out + '/' + 'posi2_kg', paths_posi2_kg)
    np.save(dir_out + '/' + 'scope_kg', instance_scope_kg)

    np.save(dir_out + '/' + 'head_tx', paths_head_tx)
    np.save(dir_out + '/' + 'tail_tx', paths_tail_tx)
    np.save(dir_out + '/' + 'label_tx', paths_label_tx)
    np.save(dir_out + '/' + 'word_tx', paths_word_tx)
    np.save(dir_out + '/' + 'posi1_tx', paths_posi1_tx)
    np.save(dir_out + '/' + 'posi2_tx', paths_posi2_tx)
    np.save(dir_out + '/' + 'scope_tx', instance_scope_tx)

    np.save(dir_out + '/' + 'head_hy', paths_head_hy)
    np.save(dir_out + '/' + 'tail_hy', paths_tail_hy)
    np.save(dir_out + '/' + 'label_hy', paths_label_hy)
    np.save(dir_out + '/' + 'word_hy', paths_word_hy)
    np.save(dir_out + '/' + 'posi1_hy', paths_posi1_hy)
    np.save(dir_out + '/' + 'posi2_hy', paths_posi2_hy)
    np.save(dir_out + '/' + 'scope_hy', instance_scope_hy)

def prepocess_kg_triple(triples, word2id, relation2id, dir_out):
    total = len(triples)
    all_hi = []
    all_ti = []
    d_tup = {}
    kg_tup = []
    kg_tup_neg = []

    for h, t, rel in triples:
        try:
            hi = word2id[h]
            ti = word2id[t]
            reli = relation2id[rel]
        except KeyError:
            continue
        d_tup[(hi, ti, reli)] = True
        all_hi.append(hi)
        all_ti.append(ti)
        tup = [hi, ti, reli]
        kg_tup.append(tup)

    all_hi = list(set(all_hi))
    all_ti = list(set(all_ti))
    for hi, ti, reli in kg_tup:
        neg_hi = hi
        neg_ti = ti
        neg_tup = [neg_hi, neg_ti, reli]
        while True:
            ht_prob = np.random.binomial(1, 0.5)
            if ht_prob:
                neg_hi = random.choice(all_hi)
            else:
                neg_ti = random.choice(all_ti)
            neg_tup[0] = neg_hi
            neg_tup[1] = neg_ti
            try:
                d_tup[tuple(neg_tup)]
            except KeyError:
                break
        kg_tup_neg.append(neg_tup)

    kg_tup = np.array(kg_tup, dtype=np.int32)
    kg_tup_neg = np.array(kg_tup_neg, dtype=np.int32)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    np.save(dir_out + '/' + 'kg', kg_tup)
    np.save(dir_out + '/' + 'kg_neg', kg_tup_neg)

def preprocess_vec(word2id, word2vec, dir_out):
    nb_voc = len(word2id)
    vec_dim = len(list(word2vec.items())[0][1])
    vec = np.ones((nb_voc, vec_dim), dtype = np.float32)
    for word, wordi in tqdm(word2id.items()):
        try:
            wordv = word2vec[word]
            vec[wordi] = wordv
        except KeyError:
            continue
        
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    np.save(dir_out + '/' + 'vec', vec)
    
def preprocess(fixlen, sf_kg_train, sf_kg_test, sf_tx, sf_emb, nb_path, cutoff,
               dir_out_train, dir_out_test):

    ug = UG(sf_kg_train, sf_kg_test, sf_tx, sf_emb, nb_path, cutoff)
    data = ug.dataset

    try:
        word2id = data['word2id']
        if 'UNK' not in word2id:
            word2id['UNK'] = len(word2id)
        if 'BLANK' not in word2id:
            word2id['BLANK'] = len(word2id)
        if 'PADDING' not in word2id:
            word2id['PADDING'] = len(word2id)
    except KeyError:
        print('There is no word2id!')
        raise

    try:
        word2vec = data['word2vec']
    except KeyError:
        print('There is no word2vec!')
        raise

    try:
        relation2id = data['relation2id']
    except KeyError:
        print('There is no relation2id!')
        raise

    try:
        triples = data['triples']
    except KeyError:
        print('There is no triples!')
        raise

    for ent1, ent2, rel in triples:
        if ent1 not in word2id:
            word2id[ent1] = len(word2id)
        if ent2 not in word2id:
            word2id[ent2] = len(word2id)

    if not os.path.exists(dir_out_train):
        os.makedirs(dir_out_train)
    with open(dir_out_train + '/' + 'relation2id.json', 'w') as fle_rel2id:
        json.dump(relation2id, fle_rel2id)
    with open(dir_out_train + '/' + 'word2id.json', 'w') as fle_w2id:
        json.dump(word2id, fle_w2id)
            
    print('preprocessing word2vec ...')
    preprocess_vec(word2id, word2vec, dir_out_train)
        
    print('preprocessing triples ...')
    prepocess_kg_triple(triples, word2id, relation2id, dir_out_train)
                
    bags_train = data['train']
    bags_test = data['test']
    print('preprocessing training data ...')
    preprocess_ug_paths(bags_train, dir_out_train, fixlen, word2id, relation2id)
    print('preprocessing testing data ...')
    preprocess_ug_paths(bags_test, dir_out_test, fixlen, word2id, relation2id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_path', type=int, default=10, required=True, help='the maximum number of paths given an entity pair and one type of graph')
    parser.add_argument('--cutoff', type=int, default=3, required=True, help='Depth to stop the search')

    args = parser.parse_args()
    preprocess(conf.MAX_LENGTH,
               conf.ADDR_KG_Train,
               conf.ADDR_KG_Train,
               conf.ADDR_TX,
               conf.ADDR_EMB,
               args.nb_path,
               args.cutoff,
               conf.DIR_TRAIN,
               conf.DIR_TEST)
