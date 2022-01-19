import tensorflow as tf
import numpy as np
import time, datetime, sys, os, json, argparse
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score
import threading
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict

import settings as conf
import network_ug as network
from kg_dataset_transe import KnowledgeGraph
from metrics import metrics

from load_ug import UG

class Model:

    def __init__(self, dir_train=conf.DIR_TRAIN, dir_test=conf.DIR_TEST, model_dir=conf.MODEL_DIR,
                 nb_batch_triple=conf.NB_BATCH_TRIPLE, batch_size=conf.BATCH_SIZE, testing_batch_size=conf.TESTING_BATCH_SIZE,
                 max_epoch=conf.MAX_EPOCH, max_length=conf.MAX_LENGTH, hidden_size=conf.HIDDEN_SIZE, posi_size=conf.POSI_SIZE,
                 learning_rate=conf.LR, learning_rate_kgc=conf.LR_KGC, keep_prob=conf.KEEP_PROB, margin=conf.MARGIN,
                 strategy=conf.STRATEGY, checkpoint_every=conf.CHECKPOINT_EVERY, result_dir=conf.RESULT_DIR, p_at_n=conf.P_AT_N,
                 load_graph=True, addr_kg_train=conf.ADDR_KG_Train, addr_kg_test=conf.ADDR_KG_Test, addr_tx=conf.ADDR_TX, addr_emb=conf.ADDR_EMB):

        if load_graph:
            self.G_ug = UG(addr_kg_train, addr_kg_test, addr_tx, addr_emb)

        self.dir_train = dir_train
        self.dir_test = dir_test
        self.model_dir = model_dir
        self.nb_batch_triple = nb_batch_triple
        self.batch_size = batch_size
        self.testing_batch_size = testing_batch_size
        self.max_epoch = max_epoch
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.posi_size = posi_size
        self.learning_rate = learning_rate
        self.learning_rate_kgc = learning_rate_kgc
        self.keep_prob = keep_prob
        self.margin = margin
        self.strategy=strategy
        self.checkpoint_every = checkpoint_every
        self.rank_topn = conf.RANK_TOPN
        self.result_dir=result_dir
        self.p_at_n = p_at_n
        
        self.seed = conf.SEED
        with open(self.dir_train + '/' + 'relation2id.json', 'r') as fle_rl2id:
            self.relation2id = json.load(fle_rl2id)
        self.num_classes = len(self.relation2id)
        self.rel_total = len(self.relation2id)
        self.id2relation = {ind:rl for rl, ind in self.relation2id.items()}
        with open(self.dir_train + '/' + 'word2id.json', 'r') as fle_w2id:
            self.word2id = json.load(fle_w2id)
        self.word_vec = np.load(self.dir_train + '/' + 'vec.npy')

        self.posi_num = self.max_length * 2 + 1
        
        self.data = None
        self.KG = None

    def train(self):
        strategy = self.strategy
        tf.reset_default_graph()
        self.data, self.KG = self.__load_data()

        para_info = [['bags', len(self.data['instance_scope'])],
                     ['paths', len(self.data['train_word'])],
                     ['relations', self.num_classes],
                     ['word size', len(self.data['word_vec'][0])],
                     ['position size', self.posi_size],
                     ['hidden size', self.hidden_size]]
        print(tabulate(para_info, headers=['Name', 'Numb.'], tablefmt='orgtbl'))

        print('building network...')

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        model, global_step, optimizer, grads_and_vars, train_op, global_step_kgc, optimizer_kgc, grads_and_vars_kgc, train_op_kgc = self.__model_init(strategy=self.strategy)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)
        
        def train_kgc(coord):
            batch_size = int(self.KG.n_triplet / self.nb_batch_triple)
            times_kg = 0
            while not coord.should_stop():
                times_kg += 1
                res = 0.0
                pos_batch_gen = self.KG.next_pos_batch(batch_size)
                neg_batch_gen = self.KG.next_neg_batch(batch_size)
                for batchi in range(int(self.nb_batch_triple)):
                    pos_batch = next(pos_batch_gen)
                    neg_batch = next(neg_batch_gen)
                    ph = pos_batch[:, 0]
                    pt = pos_batch[:, 1]
                    pr = pos_batch[:, 2]

                    nh = neg_batch[:, 0]
                    nt = neg_batch[:, 1]
                    nr = neg_batch[:, 2]

                    res += self.__train_step_kgc(sess, model, train_op_kgc, global_step_kgc, ph, pt, pr, nh, nt, nr)

        def train_nn(coord):
            stack_output = []
            stack_label = []
            train_order = range(len(self.data['instance_scope']))
            
            if strategy in ['none', 'pretrain', 'pretrain_ranking', 'locloss']:
                if strategy == 'pretrain_ranking':
                    print('pretrain phase in pretrain_ranking starts!')
                pbar = tqdm(range(self.max_epoch), desc="Training progress")
                for one_epoch in pbar:
                    np.random.shuffle(train_order)
                    s1 = 0.0
                    s2 = 0.0
                    tot1 = 0.0
                    tot2 = 1.0
                    losstot = 0.0

                    for i in range(int(len(train_order)/float(self.batch_size))):
                        [label, weights, scope, scope_kg, scope_tx, scope_hy,
                         train_head_batch, train_head_kg_batch, train_head_tx_batch, train_head_hy_batch,
                         train_tail_batch, train_tail_kg_batch, train_tail_tx_batch, train_tail_hy_batch,
                         index, index_kg, index_tx, index_hy,
                         scope_3tp, train_head_3tp_batch, train_tail_3tp_batch,
                         train_word_3tp, train_posi1_3tp, train_posi2_3tp,
                         train_label_3tp, train_comp_fea_3tp] = self.__load_data_batch(i, train_order)
                        
                        label_ = np.zeros((self.batch_size, self.num_classes))
                        label_[np.arange(self.batch_size), label] = 1

                        if strategy in ['none']:
                            output, loss, correct_predictions = self.__train_step(sess, model, train_op, global_step,
                                                                                  train_head_batch, train_tail_batch, self.data['train_word'][index,:],
                                                                                  self.data['train_posi1'][index,:], self.data['train_posi2'][index,:],
                                                                                  self.data['train_label'][index],
                                                                                  label_, np.array(scope), weights, self.data['train_comp_fea'][index,:])

                        elif strategy in ['locloss']:
                            output, loss, correct_predictions = self.__train_step(sess, model, train_op, global_step,
                                                                                  train_head_3tp_batch, train_tail_3tp_batch, train_word_3tp,
                                                                                  train_posi1_3tp, train_posi2_3tp,
                                                                                  train_label_3tp,
                                                                                  label_, np.array(scope_3tp), weights, train_comp_fea_3tp)

                        elif strategy in ['pretrain', 'pretrain_ranking']:
                            
                            output, loss, correct_predictions = self.__train_step(sess, model, train_op, global_step,
                                                                                  train_head_tx_batch, train_tail_tx_batch, self.data['train_word_tx'][index_tx,:],
                                                                                  self.data['train_posi1_tx'][index_tx,:], self.data['train_posi2_tx'][index_tx,:],
                                                                                  self.data['train_label_tx'][index_tx],
                                                                                  label_, np.array(scope_tx), weights, self.data['train_comp_fea_tx'][index_tx,:])
                            
                            output, loss, correct_predictions = self.__train_step(sess, model, train_op, global_step,
                                                                                  train_head_hy_batch, train_tail_hy_batch, self.data['train_word_hy'][index_hy,:],
                                                                                  self.data['train_posi1_hy'][index_hy,:], self.data['train_posi2_hy'][index_hy,:],
                                                                                  self.data['train_label_hy'][index_hy],
                                                                                  label_, np.array(scope_hy), weights, self.data['train_comp_fea_hy'][index_hy,:])
                            
                            output, loss, correct_predictions = self.__train_step(sess, model, train_op, global_step,
                                                                                  train_head_kg_batch, train_tail_kg_batch, self.data['train_word_kg'][index_kg,:],
                                                                                  self.data['train_posi1_kg'][index_kg,:], self.data['train_posi2_kg'][index_kg,:],
                                                                                  self.data['train_label_kg'][index_kg],
                                                                                  label_, np.array(scope_kg), weights, self.data['train_comp_fea_kg'][index_kg,:])
                            
                            output, loss, correct_predictions = self.__train_step(sess, model, train_op, global_step,
                                                                                  train_head_batch, train_tail_batch, self.data['train_word'][index,:],
                                                                                  self.data['train_posi1'][index,:], self.data['train_posi2'][index,:],
                                                                                  self.data['train_label'][index],
                                                                                  label_, np.array(scope), weights, self.data['train_comp_fea'][index,:])
                            
                        losstot += loss
                        s1, s2, tot1, tot2 = self.__cal_accuracy(s1, s2, tot1, tot2, label, correct_predictions)
                        loss_info = {'loss': loss, 'not NA accuracy': s2 / tot2}
                        pbar.set_postfix(loss_info)
                        current_step = tf.train.global_step(sess, global_step)

                    if (one_epoch + 1) % self.checkpoint_every == 0 and (one_epoch + 1) >= (self.max_epoch - 4):
                        path = saver.save(sess, self.model_dir+'/'+self.strategy, global_step=current_step)
                        description = 'have saved model to '+path
                        pbar.set_description(description)

            if strategy in ['ranking', 'pretrain_ranking']:
                if strategy == 'pretrain_ranking':
                    print('ranking phase in pretrain_ranking starts!')

                pbar = tqdm(range(self.max_epoch), desc="Training progress")
                for one_epoch in pbar:
                    np.random.shuffle(train_order)
                    s1 = 0.0
                    s2 = 0.0
                    tot1 = 0.0
                    tot2 = 1.0
                    losstot = 0.0
                    for i in range(int(len(train_order)/float(self.batch_size))):
                        [label, weights, scope, scope_kg, scope_tx, scope_hy,
                         train_head_batch, train_head_kg_batch, train_head_tx_batch, train_head_hy_batch,
                         train_tail_batch, train_tail_kg_batch, train_tail_tx_batch, train_tail_hy_batch,
                         index, index_kg, index_tx, index_hy] = self.__load_data_batch(i, train_order)

                        label_ = np.zeros((self.batch_size, self.num_classes))
                        label_[np.arange(self.batch_size), label] = 1

                        output, loss, correct_predictions = self.__train_step(sess, model, train_op, global_step,
                                                                              train_head_batch, train_tail_batch, self.data['train_word'][index,:],
                                                                              self.data['train_posi1'][index,:], self.data['train_posi2'][index,:],
                                                                              self.data['train_label'][index],
                                                                              label_, np.array(scope), weights, self.data['train_comp_fea'][index,:])

                        losstot += loss
                        s1, s2, tot1, tot2 = self.__cal_accuracy(s1, s2, tot1, tot2, label, correct_predictions)
                        loss_info = {'loss': loss, 'not NA accuracy': s2 / tot2}
                        pbar.set_postfix(loss_info)
                        current_step = tf.train.global_step(sess, global_step)

                    if (one_epoch + 1) % self.checkpoint_every == 0 and (one_epoch + 1) >= (self.max_epoch - 4):
                        path = saver.save(sess, self.model_dir+'/'+self.strategy, global_step=current_step)
                        description = 'have saved model to '+path
                        pbar.set_description(description)

            coord.request_stop()

        coord = tf.train.Coordinator()
        threads = []
        threads.append(threading.Thread(target=train_kgc, args=(coord,)))
        threads.append(threading.Thread(target=train_nn, args=(coord,)))
        for t in threads: t.start()
        coord.join(threads)


    def infer(self, lst_ep, nb_path, cutoff, model_dir):
        """
        extract relations from a given list of entity pair (i.e., (h, t)) with the help of a previously trained model
        
        @param lst_ep:  [(h1, t1), (h2, t2), ...]
        """

        if os.path.isdir(model_dir):
            if model_dir.endswith('/'):
                pass
            else:
                model_dir = model_dir + '/'
        else:
            print('the directory does not exist!')
            raise
        
        tf.reset_default_graph()
        
        bags = self.G_ug.extract_ug_paths(lst_ep, nb_path=nb_path, cutoff=cutoff)
        
        paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, paths_scope, paths_comp_fea, lst_ep, ep2paths = self.__preprocess_ug_paths(bags)

        model = network.CNN(is_training=False, word_embeddings=self.word_vec, max_length=self.max_length,
                            num_classes=self.num_classes, hidden_size=self.hidden_size, posi_size=self.posi_size,
                            margin=self.margin, rel_total=self.rel_total, seed=self.seed,
                            batch_size=self.batch_size, testing_batch_size=self.testing_batch_size, rank_topn=self.rank_topn, strategy=self.strategy,
                            is_inferring=True, nb_bags=len(bags))
        
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint_fle = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint_fle)

        
        feed_dict = {
            model.head_index: paths_head,
            model.tail_index: paths_tail,
            model.word: paths_word,
            model.posi1: paths_posi1,
            model.posi2: paths_posi2,
            model.scope: paths_scope,
            model.comp_fea: paths_comp_fea
            }

        test_pred, test_sc, test_att = sess.run([model.test_pred, model.test_sc, model.test_att], feed_dict)
        test_pred_rel = [self.id2relation[i] for i in test_pred]
        all_results = []
        for epi, ep in enumerate(lst_ep):
            rel = test_pred_rel[epi]
            sc = test_sc[epi]
            e1, e2 = ep

            att_path = test_att[paths_scope[epi]:paths_scope[epi+1]]
            
            result = {}
            result['triple_sc'] = (e1, rel, e2, sc)

            sum_att = 0.0
            for att, path in zip(att_path, ep2paths[ep]):
                path = ' '.join(path)
                path_att = (path, att)
                result.setdefault('path_att', []).append(path_att)
                sum_att += att
                
            all_results.append(result)

        return all_results

    
    def test(self):
        tf.reset_default_graph()
        self.data_test, _ = self.__load_data(is_training=False)
        
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        
        model = network.CNN(is_training=False, word_embeddings=self.word_vec, max_length=self.max_length,
                            num_classes=self.num_classes, hidden_size=self.hidden_size, posi_size=self.posi_size,
                            margin=self.margin, rel_total=self.rel_total, seed=self.seed,
                            batch_size=self.batch_size, testing_batch_size=self.testing_batch_size, rank_topn=self.rank_topn, strategy=self.strategy)
        
        sess.run(tf.global_variables_initializer())

        def test_step(head, tail, word, posi1, posi2, label_index, label, scope, comp_fea):
            feed_dict = {
                model.head_index: head,
                model.tail_index: tail,
                model.word: word,
                model.posi1: posi1,
                model.posi2: posi2,
                model.label_index: label_index,
                model.label: label,
                model.scope: scope,
                model.comp_fea: comp_fea
                }
            output, test_att, test_pred = sess.run([model.test_output, model.test_att, model.test_pred], feed_dict)
            return output, test_att, test_pred 
        
        saver = tf.train.Saver()
        checkpoint_fle = tf.train.latest_checkpoint(self.model_dir + '/')
        saver.restore(sess, checkpoint_fle)
        
        stack_output = []
        stack_label = []
        stack_att = []
        stack_pred = []
        stack_true = []
        stack_scope = []

        instance_scope = self.data_test["instance_scope"]

        iteration = len(instance_scope)/self.testing_batch_size
        for i in range(iteration):
            temp_str= 'running '+str(i)+'/'+str(iteration)+'...'
            sys.stdout.write(temp_str+'\r')
            sys.stdout.flush()
            input_scope = instance_scope[i * self.testing_batch_size:(i+1)*self.testing_batch_size]
            index = []
            scope = [0]
            label = []

            for num in input_scope:
	        index = index + range(num[0], num[1] + 1)
	        label.append(self.data_test["test_label"][num[0]])
	        scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
            
            label_ = np.zeros((self.testing_batch_size, self.num_classes))
	    label_[np.arange(self.testing_batch_size), label] = 1

            output, test_att, test_pred = test_step(self.data_test["test_head"][index], self.data_test["test_tail"][index], self.data_test["test_word"][index,:],
                                                    self.data_test["test_posi1"][index,:], self.data_test["test_posi2"][index,:], self.data_test["test_label"][index],
                                                    label_, np.array(scope), self.data_test["test_comp_fea"][index,:])
            stack_output.append(output)
	    stack_label.append(label_)
            stack_att.append(test_att)
            stack_pred.append(test_pred)
            stack_true.extend(label)
            stack_scope.extend(input_scope)

        stack_output = np.concatenate(stack_output, axis=0)
        stack_label = np.concatenate(stack_label, axis=0)
        stack_att = np.concatenate(stack_att, axis=0)
        stack_pred = np.concatenate(stack_pred, axis=0)
        stack_true = np.array(stack_true)
        stack_scope = np.array(stack_scope)
        
        exclude_na_flatten_output = stack_output[:,1:]
        exclude_na_flatten_label = stack_label[:,1:]
        
        score = metrics(stack_label, stack_output)
        performances = score.precision_at_k(self.p_at_n)
        
        y_pred = np.argmax(stack_output, axis=1)
        y_true = np.argmax(stack_label, axis=1)

        #print('Accuracy: %s' % accuracy_score(y_true, y_pred))
        
        pr = precision_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        rc = recall_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        f1 = f1_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro')
        average_precision = average_precision_score(exclude_na_flatten_label,exclude_na_flatten_output, average = "micro")
        
        performances.append(['F1',  '%.3f' % f1])
        performances.append(['AUC',  '%.3f' % average_precision])
    
        print(tabulate(performances, headers=['Metric', 'Score'], tablefmt='orgtbl'))
    
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
    
        np.save(self.result_dir+'/'+self.strategy+'_prob'+'.npy', exclude_na_flatten_output)
        np.save(self.result_dir+'/'+self.strategy+'_label'+'.npy',exclude_na_flatten_label)
        
        
    def __train_step_kgc(self, sess, model, train_op_kgc, global_step_kgc, pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
        feed_dict = {
            model.pos_h: pos_h_batch,
            model.pos_t: pos_t_batch,
            model.pos_r: pos_r_batch,
            model.neg_h: neg_h_batch,
            model.neg_t: neg_t_batch,
            model.neg_r: neg_r_batch
            }
        _, step, loss = sess.run([train_op_kgc, global_step_kgc, model.loss_kgc], feed_dict)
        return loss

    
    def __train_step(self, sess, model, train_op, global_step, head, tail, word, posi1, posi2, label_index, label, scope, weights, comp_fea):
        feed_dict = {
            model.head_index: head,
            model.tail_index: tail,
            model.word: word,
            model.posi1: posi1,
            model.posi2: posi2,
            model.label_index: label_index,
            model.label: label,
            model.scope: scope,
            model.keep_prob: self.keep_prob,
            model.weights: weights,
            model.comp_fea: comp_fea
            }
        _, step, loss, output, correct_predictions = sess.run([train_op, global_step, model.loss, model.output, model.correct_predictions], feed_dict)
        return output, loss, correct_predictions


    def __load_data(self, is_training=True):

        def cal_reltot(train_label):
            reltot = {}
            for index, i in enumerate(train_label):
                if not i in reltot:
                    reltot[i] = 1.0
                else:
                    reltot[i] += 1.0
            for i in reltot:
                reltot[i] = 1/(reltot[i] ** (0.05))
            return reltot
        
        def complexity_features(array):
            return np.array([[np.count_nonzero(ele), np.unique(ele).size] for ele in array]).astype(np.float32)

        data = {}

        if is_training:
            export_path = self.dir_train + '/'
            tag = 'train'
        else:
            export_path = self.dir_test + '/'
            tag = 'test'
            
        print('reading data ...')
        data['word_vec'] = np.load(self.dir_train + '/' + 'vec.npy')

        data['instance_scope'] = np.load(export_path + 'scope.npy')
        data['instance_scope_kg'] = np.load(export_path + 'scope_kg.npy')
        data['instance_scope_tx'] = np.load(export_path + 'scope_tx.npy')
        data['instance_scope_hy'] = np.load(export_path + 'scope_hy.npy')

        data['%s_label' % tag] = np.load(export_path + 'label.npy')
        data['reltot'] = cal_reltot(data['%s_label' % tag])
        data['%s_label_kg' % tag] = np.load(export_path + 'label_kg.npy')
        data['%s_label_tx' % tag] = np.load(export_path + 'label_tx.npy')
        data['%s_label_hy' % tag] = np.load(export_path + 'label_hy.npy')

        data['%s_word' % tag] = np.load(export_path + 'word.npy')
        data['%s_posi1' % tag] = np.load(export_path + 'posi1.npy')
        data['%s_posi2' % tag] = np.load(export_path + 'posi2.npy')

        data['%s_word_kg' % tag] = np.load(export_path + 'word_kg.npy')
        data['%s_posi1_kg' % tag] = np.load(export_path + 'posi1_kg.npy')
        data['%s_posi2_kg' % tag] = np.load(export_path + 'posi2_kg.npy')

        data['%s_word_tx' % tag] = np.load(export_path + 'word_tx.npy')
        data['%s_posi1_tx' % tag] = np.load(export_path + 'posi1_tx.npy')
        data['%s_posi2_tx' % tag] = np.load(export_path + 'posi2_tx.npy')

        data['%s_word_hy' % tag] = np.load(export_path + 'word_hy.npy')
        data['%s_posi1_hy' % tag] = np.load(export_path + 'posi1_hy.npy')
        data['%s_posi2_hy' % tag] = np.load(export_path + 'posi2_hy.npy')

        data['%s_head' % tag] = np.load(export_path + 'head.npy')
        data['%s_tail' % tag] = np.load(export_path + 'tail.npy')

        data['%s_comp_fea' % tag] = complexity_features(data['%s_word' % tag])
        data['%s_comp_fea_kg' % tag] = complexity_features(data['%s_word_kg' % tag])
        data['%s_comp_fea_tx' % tag] = complexity_features(data['%s_word_tx' % tag])
        data['%s_comp_fea_hy' % tag] = complexity_features(data['%s_word_hy' % tag])

        KG = KnowledgeGraph(self.dir_train + '/')
        
        print('reading finished')
        return data, KG

    
    def __load_data_batch(self, i, train_order):
        input_scope = np.take(self.data['instance_scope'], train_order[i * self.batch_size:(i+1)*self.batch_size], axis=0)
        input_scope_kg = np.take(self.data['instance_scope_kg'], train_order[i * self.batch_size:(i+1)*self.batch_size], axis=0)
        input_scope_tx = np.take(self.data['instance_scope_tx'], train_order[i * self.batch_size:(i+1)*self.batch_size], axis=0)
        input_scope_hy = np.take(self.data['instance_scope_hy'], train_order[i * self.batch_size:(i+1)*self.batch_size], axis=0)

        index = []
        scope = [0]
        index_kg = []
        index_tx = []
        index_hy = []
        scope_kg = [0]
        scope_tx = [0]
        scope_hy = [0]
        
        scope_3tp = [0]
        
        label = []
        weights = []
        train_head_kg_batch = []
        train_tail_kg_batch = []
        train_head_tx_batch = []
        train_tail_tx_batch = []
        train_head_hy_batch = []
        train_tail_hy_batch = []

        train_head_3tp_batch = []
        train_tail_3tp_batch = []

        train_word_3tp = []
        train_posi1_3tp = []
        train_posi2_3tp = []
        train_label_3tp = []
        train_comp_fea_3tp = []

        for num, num_kg, num_tx, num_hy in zip(input_scope, input_scope_kg, input_scope_tx, input_scope_hy):
            index = index + range(num[0], num[1] + 1)
            label.append(self.data['train_label'][num[0]])
            scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
            weights.append(self.data['reltot'][self.data['train_label'][num[0]]])
            index_kg = index_kg + range(num_kg[0], num_kg[1] + 1)
            scope_kg.append(scope_kg[len(scope_kg)-1] + num_kg[1] - num_kg[0] + 1)
            index_tx = index_tx + range(num_tx[0], num_tx[1] + 1)
            scope_tx.append(scope_tx[len(scope_tx)-1] + num_tx[1] - num_tx[0] + 1)
            index_hy = index_hy + range(num_hy[0], num_hy[1] + 1)
            scope_hy.append(scope_hy[len(scope_hy)-1] + num_hy[1] - num_hy[0] + 1)

            scope_3tp.append(scope_3tp[len(scope_3tp)-1] + num_kg[1] - num_kg[0] + 1)#e.g., [0, 3]
            scope_3tp.append(scope_3tp[len(scope_3tp)-1] + num_tx[1] - num_tx[0] + 1)#e.g., [0, 3, 7]
            scope_3tp.append(scope_3tp[len(scope_3tp)-1] + num_hy[1] - num_hy[0] + 1)#e.g., [0, 3, 7, 13]
            
            train_head_kg_batch += [self.data['train_head'][num[0]]]*len(range(num_kg[0], num_kg[1] + 1))
            train_tail_kg_batch += [self.data['train_tail'][num[0]]]*len(range(num_kg[0], num_kg[1] + 1))
            train_head_tx_batch += [self.data['train_head'][num[0]]]*len(range(num_tx[0], num_tx[1] + 1))
            train_tail_tx_batch += [self.data['train_tail'][num[0]]]*len(range(num_tx[0], num_tx[1] + 1))
            train_head_hy_batch += [self.data['train_head'][num[0]]]*len(range(num_hy[0], num_hy[1] + 1))
            train_tail_hy_batch += [self.data['train_tail'][num[0]]]*len(range(num_hy[0], num_hy[1] + 1))

            train_head_3tp_batch += [self.data['train_head'][num[0]]]*len(range(num_kg[0], num_kg[1] + 1))
            train_head_3tp_batch += [self.data['train_head'][num[0]]]*len(range(num_tx[0], num_tx[1] + 1))
            train_head_3tp_batch += [self.data['train_head'][num[0]]]*len(range(num_hy[0], num_hy[1] + 1))
            train_tail_3tp_batch += [self.data['train_tail'][num[0]]]*len(range(num_kg[0], num_kg[1] + 1))
            train_tail_3tp_batch += [self.data['train_tail'][num[0]]]*len(range(num_tx[0], num_tx[1] + 1))
            train_tail_3tp_batch += [self.data['train_tail'][num[0]]]*len(range(num_hy[0], num_hy[1] + 1))
            
            train_word_3tp.extend(self.data['train_word_kg'][num_kg[0]:num_kg[1] + 1])
            train_word_3tp.extend(self.data['train_word_tx'][num_tx[0]:num_tx[1] + 1])
            train_word_3tp.extend(self.data['train_word_hy'][num_hy[0]:num_hy[1] + 1])
            train_posi1_3tp.extend(self.data['train_posi1_kg'][num_kg[0]:num_kg[1] + 1])
            train_posi1_3tp.extend(self.data['train_posi1_tx'][num_tx[0]:num_tx[1] + 1])
            train_posi1_3tp.extend(self.data['train_posi1_hy'][num_hy[0]:num_hy[1] + 1])
            train_posi2_3tp.extend(self.data['train_posi2_kg'][num_kg[0]:num_kg[1] + 1])
            train_posi2_3tp.extend(self.data['train_posi2_tx'][num_tx[0]:num_tx[1] + 1])
            train_posi2_3tp.extend(self.data['train_posi2_hy'][num_hy[0]:num_hy[1] + 1])

            train_label_3tp.extend(self.data['train_label_kg'][num_kg[0]:num_kg[1] + 1])
            train_label_3tp.extend(self.data['train_label_tx'][num_tx[0]:num_tx[1] + 1])
            train_label_3tp.extend(self.data['train_label_hy'][num_hy[0]:num_hy[1] + 1])

            train_comp_fea_3tp.extend(self.data['train_comp_fea_kg'][num_kg[0]:num_kg[1] + 1])
            train_comp_fea_3tp.extend(self.data['train_comp_fea_tx'][num_tx[0]:num_tx[1] + 1])
            train_comp_fea_3tp.extend(self.data['train_comp_fea_hy'][num_hy[0]:num_hy[1] + 1])
            
        train_head_batch = self.data['train_head'][index]
        train_tail_batch = self.data['train_tail'][index]

        train_word_3tp = np.array(train_word_3tp).astype(np.int32)
        train_posi1_3tp = np.array(train_posi1_3tp).astype(np.int32)
        train_posi2_3tp = np.array(train_posi2_3tp).astype(np.int32)

        data_batch = [label, weights, scope, scope_kg, scope_tx, scope_hy,
                      train_head_batch, train_head_kg_batch, train_head_tx_batch, train_head_hy_batch,
                      train_tail_batch, train_tail_kg_batch, train_tail_tx_batch, train_tail_hy_batch,
                      index, index_kg, index_tx, index_hy,
                      scope_3tp, train_head_3tp_batch, train_tail_3tp_batch,
                      train_word_3tp, train_posi1_3tp, train_posi2_3tp,
                      train_label_3tp, train_comp_fea_3tp]
        
        return data_batch


    def __model_init(self, strategy=None):
        if not strategy:
            strategy = self.strategy
        else:
            strategy = strategy
        model = network.CNN(is_training=True, word_embeddings=self.data['word_vec'], max_length=self.max_length,
                            num_classes=self.num_classes, hidden_size=self.hidden_size, posi_size=self.posi_size,
                            margin=self.margin, rel_total=self.rel_total, seed=self.seed,
                            batch_size=self.batch_size, testing_batch_size=self.testing_batch_size, rank_topn=self.rank_topn, strategy=strategy)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        global_step_kgc = tf.Variable(0,name='global_step_kgc',trainable=False)

        print('building RE ...')
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

        print('building KGC ...')
        optimizer_kgc = tf.train.GradientDescentOptimizer(self.learning_rate_kgc)
        grads_and_vars_kgc = optimizer_kgc.compute_gradients(model.loss_kgc)
        train_op_kgc = optimizer_kgc.apply_gradients(grads_and_vars_kgc, global_step = global_step_kgc)

        return model, global_step, optimizer, grads_and_vars, train_op, global_step_kgc, optimizer_kgc, grads_and_vars_kgc, train_op_kgc
        

    def __cal_accuracy(self, s1, s2, tot1, tot2, label, correct_predictions):
        num = 0
        s = 0
        for num in correct_predictions:
            if label[s] == 0:
                tot1 += 1.0
                if num:
                    s1+= 1.0
            else:
                tot2 += 1.0
                if num:
                    s2 += 1.0
            s = s + 1
        return s1, s2, tot1, tot2

    def __preprocess_ug_paths(self, bags):

        def complexity_features(array):
            return np.array([[np.count_nonzero(ele), np.unique(ele).size] for ele in array]).astype(np.float32)

        def posi_embed(x, maxlen):
            return max(0, min(x + maxlen, maxlen + maxlen + 1))
        
        def indexing_paths(paths, posis, total):
            fixlen = self.max_length
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
                
                    paths_head[pathi] = headi
                    paths_tail[pathi] = taili
            
                    if instance_ep == [] or instance_ep[len(instance_ep) - 1] != ep:
	                instance_ep.append(ep)
	                instance_scope.append([pathi,pathi])
	            instance_scope[len(instance_ep) - 1][1] = pathi
                
                    for i in range(fixlen):
                        paths_word[pathi][i] = self.word2id['BLANK']
                        paths_posi1[pathi][i] = posi_embed(i - e1posi, fixlen)
                        paths_posi2[pathi][i] = posi_embed(i - e2posi, fixlen)
                
                    for i, word in enumerate(path):
                        if i >= fixlen:
                            break
                        elif not word in self.word2id:
                            paths_word[pathi][i] = self.word2id['UNK']
                        else:
                            paths_word[pathi][i] = self.word2id[word]

            return paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, np.array(instance_scope), complexity_features(paths_word), instance_ep

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

        paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, instance_scope, paths_comp_fea, lst_ep = indexing_paths(paths, posis, total)
        scope = [0]
        for num in instance_scope:
            scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)

        return paths_head, paths_tail, paths_word, paths_posi1, paths_posi2, np.array(scope), paths_comp_fea, lst_ep, paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help="please input 'train' or 'test'")
    
    parser.add_argument('--dir_train', type=str, default='./train_initialized', help='path to the preprocessed dataset for training')
    parser.add_argument('--dir_test', type=str, default='./test_initialized', help='path to the preprocessed dataset for testing')
    parser.add_argument('--model_dir', type=str, default='./model_saved', help='path to store the trained model')

    #parser.add_argument('--strategy', type=str, default='pretrain_ranking', help='training strategy: none, pretrain, ranking and pretrain_ranking')
    #parser.add_argument('--p_at_n', type=list, default=[500, 1000, 1500], help='a list of n for calculating precision at top n')
    #parser.add_argument('--result_dir', type=str, default='./results', help='dir to save the results')
    
    args = parser.parse_args()

    if args.mode == 'train':
        ugdsre_model = Model(dir_train=args.dir_train,
                             model_dir=args.model_dir,
                             load_graph=False)
        try:
            ugdsre_model.train()
        except KeyboardInterrupt:
            print('\nFinished training earlier.')

    if args.mode == 'test':
        ugdsre_model = Model(dir_test=args.dir_test,
                             model_dir=args.model_dir,
                             load_graph=False)
        ugdsre_model.test() 
