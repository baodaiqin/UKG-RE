import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network_ug as network
import json
from sklearn.metrics import average_precision_score
import sys
import threading
from tqdm import tqdm

from kg_dataset_transe import KnowledgeGraph

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_train', type=str, default='./train_initialized', help='path to the preprocessed dataset for training')
parser.add_argument('--model_dir', type=str, default='./model_saved', help='path to store the trained model')
parser.add_argument('--nb_batch_triple', type=int, default=200, help='the number of batch for kg triples')
parser.add_argument('--batch_size', type=int, default=50, help='batch size of target entity pairs')
parser.add_argument('--max_length', type=int, default=120, help='the maximum number of words in a path')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden feature size')
parser.add_argument('--posi_size', type=int, default=5, help='position embedding size')
parser.add_argument('--max_epoch', type=int, default=10, help='epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.02, help='learning rate for nn')
parser.add_argument('--learning_rate_kgc', type=float, default=0.05, help='learning rate for kgc')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
parser.add_argument('--keep_prob', type=float, default=0.5, help='dropout rate')
parser.add_argument('--strategy', type=str, default='none', help='training strategy: none, pretrain, ranking, pretrain+ranking')

args = parser.parse_args()

with open(args.dir_train + '/' + 'relation2id.json', 'r') as fle:
    relation2id = json.load(fle)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('rel_total', len(relation2id),'total of relations')
tf.app.flags.DEFINE_integer('max_length', args.max_length,'the maximum number of words in a path')
tf.app.flags.DEFINE_integer('posi_num', args.max_length * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', len(relation2id),'maximum of relations')
tf.app.flags.DEFINE_string('model_dir', args.model_dir,'path to store the trained model')

tf.app.flags.DEFINE_integer('hidden_size', args.hidden_size,'hidden feature size')
tf.app.flags.DEFINE_integer('posi_size', args.posi_size,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch', args.max_epoch,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size', args.batch_size,'entity pair numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('margin', 1.0, 'margin')
tf.app.flags.DEFINE_float('keep_prob',args.keep_prob,'dropout rate')
tf.app.flags.DEFINE_float('seed', 123,'random seed')

tf.app.flags.DEFINE_string('strategy', args.strategy,'training strategy, none, pretrain, ranking, pretrain+ranking')
tf.app.flags.DEFINE_integer('rank_topn', 50,'top n complex or simple paths')

KG = KnowledgeGraph(args.dir_train + '/')

hypa = {}
hypa['rel_total'] = len(relation2id)
hypa['max_length'] = args.max_length
hypa['num_classes'] = len(relation2id)
hypa['hidden_size'] = args.hidden_size
hypa['posi_size'] = args.posi_size
hypa['keep_prob'] = args.keep_prob
hypa['weight_decay'] = 0.00001

with open(args.dir_train + '/' + 'hyper_params.json', 'w') as fle_hypa:
    json.dump(hypa, fle_hypa)

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
        
def main(_):
    export_path = args.dir_train + '/'
    word_vec = np.load(export_path + 'vec.npy')
    print('reading training data ...')
    
    instance_scope = np.load(export_path + 'scope.npy')
    instance_scope_kg = np.load(export_path + 'scope_kg.npy')
    instance_scope_tx = np.load(export_path + 'scope_tx.npy')
    instance_scope_hy = np.load(export_path + 'scope_hy.npy')

    train_label = np.load(export_path + 'label.npy')
    reltot = cal_reltot(train_label)
    train_label_kg = np.load(export_path + 'label_kg.npy')
    train_label_tx = np.load(export_path + 'label_tx.npy')
    train_label_hy = np.load(export_path + 'label_hy.npy')
    
    train_word = np.load(export_path + 'word.npy')
    train_posi1 = np.load(export_path + 'posi1.npy')
    train_posi2 = np.load(export_path + 'posi2.npy')

    train_word_kg = np.load(export_path + 'word_kg.npy')
    train_posi1_kg = np.load(export_path + 'posi1_kg.npy')
    train_posi2_kg = np.load(export_path + 'posi2_kg.npy')

    train_word_tx = np.load(export_path + 'word_tx.npy')
    train_posi1_tx = np.load(export_path + 'posi1_tx.npy')
    train_posi2_tx = np.load(export_path + 'posi2_tx.npy')

    train_word_hy = np.load(export_path + 'word_hy.npy')
    train_posi1_hy = np.load(export_path + 'posi1_hy.npy')
    train_posi2_hy = np.load(export_path + 'posi2_hy.npy')
        
    train_head = np.load(export_path + 'head.npy')
    train_tail = np.load(export_path + 'tail.npy')

    train_comp_fea = complexity_features(train_word)
    train_comp_fea_kg = complexity_features(train_word_kg)
    train_comp_fea_tx = complexity_features(train_word_tx)
    train_comp_fea_hy = complexity_features(train_word_hy)
    
    print('reading finished')
    print('bags 		: %d' % (len(instance_scope)))
    print('paths		: %d' % (len(train_word)))
    print('relations            : %d' % (FLAGS.num_classes))
    print('word size            : %d' % (len(word_vec[0])))
    print('position size 	: %d' % (FLAGS.posi_size))
    print('hidden size		: %d' % (FLAGS.hidden_size))

    print('building network...')
	
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = network.CNN(is_training = True, word_embeddings = word_vec)
	
    global_step = tf.Variable(0,name='global_step',trainable=False)
    global_step_kgc = tf.Variable(0,name='global_step_kgc',trainable=False)

    print('building RE ...')
    optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

    print('building KGC ...')
    optimizer_kgc = tf.train.GradientDescentOptimizer(args.learning_rate_kgc)
    grads_and_vars_kgc = optimizer_kgc.compute_gradients(model.loss_kgc)
    train_op_kgc = optimizer_kgc.apply_gradients(grads_and_vars_kgc, global_step = global_step_kgc)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    print('building finished')
        
    def train_kgc(coord):
        def train_step_kgc(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
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

        batch_size = int(KG.n_triplet / args.nb_batch_triple)
        times_kg = 0
        while not coord.should_stop():
            times_kg += 1
            res = 0.0
            pos_batch_gen = KG.next_pos_batch(batch_size)
            neg_batch_gen = KG.next_neg_batch(batch_size)
            for batchi in range(int(args.nb_batch_triple)):
                pos_batch = next(pos_batch_gen)
                neg_batch = next(neg_batch_gen)
                ph = pos_batch[:, 0]
                pt = pos_batch[:, 1]
                pr = pos_batch[:, 2]

                nh = neg_batch[:, 0]
                nt = neg_batch[:, 1]
                nr = neg_batch[:, 2]

                res += train_step_kgc(ph, pt, pr, nh, nt, nr)
                time_str = datetime.datetime.now().isoformat()
		#print "batch %d time %s | loss : %f" % (times_kg, time_str, res)
        
    def train_nn(coord):
        def train_step(head, tail, word, posi1, posi2, label_index, label, scope, weights, comp_fea):
            feed_dict = {
		model.head_index: head,
		model.tail_index: tail,
		model.word: word,
		model.posi1: posi1,
		model.posi2: posi2,
		model.label_index: label_index,
		model.label: label,
		model.scope: scope,
		model.keep_prob: FLAGS.keep_prob,
		model.weights: weights,
                model.comp_fea: comp_fea
	    }
            _, step, loss, output, correct_predictions = sess.run([train_op, global_step, model.loss, model.output, model.correct_predictions], feed_dict)
            return output, loss, correct_predictions

        stack_output = []
        stack_label = []
        stack_ce_loss = []
        train_order = range(len(instance_scope))

        save_epoch = 2
        eval_step = 300

        if FLAGS.strategy in ['none', 'pretrain', 'prtrain+ranking']:
            if FLAGS.strategy == 'pretrain+ranking':
                print('pretrain phase in pretrain+ranking starts!')
            pbar = tqdm(range(FLAGS.max_epoch))
            for one_epoch in pbar:
                np.random.shuffle(train_order)
                s1 = 0.0
                s2 = 0.0
                tot1 = 0.0
                tot2 = 1.0
                losstot = 0.0
                for i in range(int(len(train_order)/float(FLAGS.batch_size))):
                    input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                    input_scope_kg = np.take(instance_scope_kg, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                    input_scope_tx = np.take(instance_scope_tx, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                    input_scope_hy = np.take(instance_scope_hy, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                    index = []
                    scope = [0]
                    index_kg = []
                    index_tx = []
                    index_hy = []
                    scope_kg = [0]
                    scope_tx = [0]
                    scope_hy = [0]

                    label = []
                    weights = []
                
                    train_head_kg_batch = []
                    train_tail_kg_batch = []
                
                    train_head_tx_batch = []
                    train_tail_tx_batch = []
                
                    train_head_hy_batch = []
                    train_tail_hy_batch = []

                    for num, num_kg, num_tx, num_hy in zip(input_scope, input_scope_kg, input_scope_tx, input_scope_hy):
                        index = index + range(num[0], num[1] + 1)
                        label.append(train_label[num[0]])
                        scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
                        weights.append(reltot[train_label[num[0]]])
                    
                        index_kg = index_kg + range(num_kg[0], num_kg[1] + 1)
                        scope_kg.append(scope_kg[len(scope_kg)-1] + num_kg[1] - num_kg[0] + 1)
                    
                        index_tx = index_tx + range(num_tx[0], num_tx[1] + 1)
                        scope_tx.append(scope_tx[len(scope_tx)-1] + num_tx[1] - num_tx[0] + 1)
                        
                        index_hy = index_hy + range(num_hy[0], num_hy[1] + 1)
                        scope_hy.append(scope_hy[len(scope_hy)-1] + num_hy[1] - num_hy[0] + 1)
                        
                        train_head_kg_batch += [train_head[num[0]]]*len(range(num_kg[0], num_kg[1] + 1))
                        train_tail_kg_batch += [train_tail[num[0]]]*len(range(num_kg[0], num_kg[1] + 1))
                    
                        train_head_tx_batch += [train_head[num[0]]]*len(range(num_tx[0], num_tx[1] + 1))
                        train_tail_tx_batch += [train_tail[num[0]]]*len(range(num_tx[0], num_tx[1] + 1))
                        
                        train_head_hy_batch += [train_head[num[0]]]*len(range(num_hy[0], num_hy[1] + 1))
                        train_tail_hy_batch += [train_tail[num[0]]]*len(range(num_hy[0], num_hy[1] + 1))

                    label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
                    label_[np.arange(FLAGS.batch_size), label] = 1

                    if FLAGS.strategy in ['none']:
                        output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:],
                                                                       train_posi1[index,:], train_posi2[index,:], train_label[index],
                                                                       label_, np.array(scope), weights, train_comp_fea[index,:])

                    elif FLAGS.strategy in ['pretrain', 'pretrain+ranking']:
                        output, loss, correct_predictions = train_step(train_head_tx_batch, train_tail_tx_batch, train_word_tx[index_tx,:],
                                                                       train_posi1_tx[index_tx,:], train_posi2_tx[index_tx,:], train_label_tx[index_tx],
                                                                       label_, np.array(scope_tx), weights, train_comp_fea_tx[index_tx,:])

                        output, loss, correct_predictions = train_step(train_head_hy_batch, train_tail_hy_batch, train_word_hy[index_hy,:],
                                                                       train_posi1_hy[index_hy,:], train_posi2_tx[index_hy,:], train_label_tx[index_hy],
                                                                       label_, np.array(scope_hy), weights, train_comp_fea_hy[index_hy,:])

                        output, loss, correct_predictions = train_step(train_head_kg_batch, train_tail_kg_batch, train_word_kg[index_kg,:],
                                                                       train_posi1_kg[index_kg,:], train_posi2_kg[index_kg,:], train_label_kg[index_kg],
                                                                       label_, np.array(scope_kg), weights, train_comp_fea_kg[index_kg,:])
                        
                        output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:],
                                                                       train_posi1[index,:], train_posi2[index,:], train_label[index],
                                                                       label_, np.array(scope), weights, train_comp_fea[index,:])

                    num = 0
                    s = 0
                    losstot += loss
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

                    time_str = datetime.datetime.now().isoformat()
                    description = "epoch %d step %d time %s | loss : %f, not NA accuracy: %f" % (one_epoch, i, time_str, loss, s2 / tot2)
                    pbar.set_description(description)
                    current_step = tf.train.global_step(sess, global_step)
                                
                if (one_epoch + 1) % save_epoch == 0 and (one_epoch + 1) >= (FLAGS.max_epoch - 4):
                    path = saver.save(sess,FLAGS.model_dir+'/'+FLAGS.strategy, global_step=current_step)
                    description = 'have saved model to '+path
                    pbar.set_description(description)

        if FLAGS.strategy == ['ranking', 'pretrain+ranking']:
            if FLAGS.strategy == 'pretrain+ranking':
                print('ranking phase in pretrain+ranking starts!')

            pbar = tqdm(FLAGS.max_epoch)
            for one_epoch in pbar:
                np.random.shuffle(train_order)
                s1 = 0.0
                s2 = 0.0
                tot1 = 0.0
                tot2 = 1.0
                losstot = 0.0
                for i in range(int(len(train_order)/float(FLAGS.batch_size))):
                    input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
                    index = []
                    scope = [0]

                    label = []
                    weights = []

                    for num in input_scope:
                        index = index + range(num[0], num[1] + 1)
                        label.append(train_label[num[0]])
                        scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
                        weights.append(reltot[train_label[num[0]]])

                    label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
                    label_[np.arange(FLAGS.batch_size), label] = 1

                    output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:],
                                                                   train_posi1[index,:], train_posi2[index,:], train_label[index],
                                                                   label_, np.array(scope), weights, train_comp_fea[index,:])
                    num = 0
                    s = 0
                    losstot += loss
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

                    time_str = datetime.datetime.now().isoformat()
                    description = "batch %d step %d time %s | loss : %f, not NA accuracy: %f" % (one_epoch, i, time_str, loss, s2 / tot2)
                    pbar.set_description(description)
                    current_step = tf.train.global_step(sess, global_step)

                if (one_epoch + 1) % save_epoch == 0 and (one_epoch + 1) >= (FLAGS.max_epoch - 4):
                    path = saver.save(sess,FLAGS.model_dir+'/'+FLAGS.strategy, global_step=current_step)
                    description = 'have saved model to '+path
                    pbar.set_description(description)

        coord.request_stop()


    coord = tf.train.Coordinator()
    threads = []
    threads.append(threading.Thread(target=train_kgc, args=(coord,)))
    threads.append(threading.Thread(target=train_nn, args=(coord,)))
    for t in threads: t.start()
    coord.join(threads)

if __name__ == "__main__":
    tf.app.run() 

