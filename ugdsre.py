import tensorflow as tf
import numpy as np
import os
import network_ug as network
import json
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class model(object):
    def __init__(self, train_dir, model_add):
        with open(train_dir + '/' + 'word2id.json', 'r') as fle_w2id:
            self.word2id = json.load(fle_w2id)
        with open(train_dir + '/' + 'relation2id.json', 'r') as fle_rel2id:
            self.relation2id = json.load(fle_rel2id)
            self.id2relation = {ind:rel for rel, ind in relation2id.items()}
        with open(train_dir + '/' + 'vec.npy', 'r') as fle_vec:
            self.vec = np.load(fle_vec)
        with open(train_dir + '/' + 'hyper_params.json', 'r') as fle_hypa:
            self.hypa = json.load(fle_hypa)

        FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_integer('rel_total', self.hypa['rel_total'],'total of relations')
        tf.app.flags.DEFINE_integer('max_length', self.hypa['max_length'],'the maximum number of words in a path')
        tf.app.flags.DEFINE_integer('posi_num', self.hypa[max_length] * 2 + 1,'number of position embedding vectors')
        tf.app.flags.DEFINE_integer('num_classes', self.hypa['rel_total'],'maximum of relations')
        tf.app.flags.DEFINE_integer('hidden_size', self.hypa['hidden_size'],'hidden feature size')
        tf.app.flags.DEFINE_integer('posi_size', self.hypa['posi_size'],'position embedding size')
        tf.app.flags.DEFINE_float('weight_decay', self.hypa['weight_decay'],'weight_decay')
        tf.app.flags.DEFINE_float('keep_prob', self.hypa['keep_prob'],'dropout rate')
        tf.app.flags.DEFINE_float('seed', 123,'random seed')
        tf.app.flags.DEFINE_integer('testing_batch_size', 1,'batch size for testing')
        if 'ranking' in model_add:
            tf.app.flags.DEFINE_float('strategy', 'ranking','training strategy, none, pretrain, ranking, pretrain+ranking')
        else:
            tf.app.flags.DEFINE_float('strategy', 'none','training strategy, none, pretrain, ranking, pretrain+ranking')
        tf.app.flags.DEFINE_float('rank_topn', 30,'top n complex or simple paths')
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model = network.CNN(is_training = False, word_embeddings = self.vec)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_add)
        
    def pos_embed(self, x, maxlen):
        return max(0, min(x + maxlen, maxlen + maxlen + 1))

    def complexity_features(self, array):
        return np.array([[np.count_nonzero(ele), np.unique(ele).size] for ele in array]).astype(np.float32)

    def indexing_paths(self, paths, posis, total):
        fixlen = self.hypa['max_length']
        paths_head = np.zeros((total), dtype = np.int32)
        paths_tail = np.zeros((total), dtype = np.int32)
        paths_word = np.zeros((total, fixlen), dtype = np.int32)
        paths_posi1 = np.zeros((total, fixlen), dtype = np.int32)
        paths_posi2 = np.zeros((total, fixlen), dtype = np.int32)
        instance_ep = []
        instance_scope = []
        
        pbar = tqdm(paths.items())
        pathi = -1
        for ep, lst_path in pbar:
            ei1, ei2 = ep
            lst_posi = posis[ep]

            headi = self.word2id[ei1]
            taili = self.word2id[ei2]
        
            for path, posi in zip(lst_path, lst_posi):
                e1posi, e2posi = posi
                pathi += 1
                
                paths_label[pathi] = reli
                paths_head[pathi] = headi
                paths_tail[pathi] = taili
            
                if instance_ep == [] or instance_triple[len(instance_ep) - 1] != ep:
	            instance_ep.append(ep)
	            instance_scope.append([pathi,pathi])
	        instance_scope[len(instance_ep) - 1][1] = pathi
                
                for i in range(fixlen):
                    paths_word[pathi][i] = self.word2id['BLANK']
                    paths_posi1[pathi][i] = self.pos_embed(i - e1posi, fixlen)
                    paths_posi2[pathi][i] = self.pos_embed(i - e2posi, fixlen)
                
                for i, word in enumerate(path):
                    if i >= fixlen:
                        break
                    elif not word in self.word2id:
                        paths_word[pathi][i] = self.word2id['UNK']
                    else:
                        paths_word[pathi][i] = self.word2id[word]

        return paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, np.array(instance_scope), self.complexity_features(paths_words)

    def preprocess_ug_paths(self, bags):
        instance_scope = []
        
        paths = defaultdict(list)
        posis = defaultdict(list)
        total = 0
        
        for bag in bags:
            try:
                e1_id = bag['e1_id']
                e2_id = bag['e2_id']
                e1_word = bag['e1_word']
                e2_word = bag['e2_word']
                ep = (e1_id, e2_id)
                
                ug_paths = bag['paths']
                ug_paths_e1_e2_posi = bag['path_e1_e2_positions']
            except KeyError:
                print('please check the format of the input dataset!')
                raise
            
            paths[ep].extend(ug_paths)
            nb_paths = len(ug_paths)
            total += nb_paths
            posis[ep].extend(ug_paths_e1_e2_posi)
        
        paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, instance_scope = self.indexing_paths(paths, posis, total)
        return paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, instance_scope
        
    def infer(self, bags):
        """
        Args:
            bags: [{'e1_id': , ..., 'ug_paths': , ...}, ...]
        """
        paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, paths_scope, paths_comp_fea = self.preprocess_ug_paths(bags)
        feed_dict = {
            model.head_index: paths_head,
	    model.tail_index: paths_tail,
	    model.word: paths_word,
	    model.posi1: paths_posi1,
	    model.posi2: paths_posi2,
	    model.scope: scope,
            model.comp_fea: paths_comp_fea
        }
    
        tf.app.flags.DEFINE_integer('testing_batch_size', len(bags),'batch size for testing')
    
        output, test_att, test_pred, test_sc = self.sess.run([self.model.test_output, self.model.test_att, self.model.test_pred, self.model.test_sc], feed_dict)
        test_pred_rel = [self.id2relation[i] for i in test_pred]
        return [(rel, sc) for rel, sc in zip(test_pred_rel, test_pred_sc)]
    
    
    
