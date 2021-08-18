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

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from metrics import metrics

from kg_dataset_transe import KnowledgeGraph

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_test', type=str, default='./test_initialized', required=True, help='path to the preprocessed dataset for testing')
parser.add_argument('--model_addr', type=str, default=None, required=True, help='address to the trained model')
parser.add_argument('--result_dir', type=str, default=None, required=True, help='dir to save the results')
parser.add_argument('--nb_batch_triple', type=int, default=200, help='the number of batch for kg triples')
parser.add_argument('--batch_size', type=int, default=50, help='batch size of target entity pairs')
parser.add_argument('--max_length', type=int, default=120, help='the maximum number of words in a path')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden feature size')
parser.add_argument('--posi_size', type=int, default=5, help='position embedding size')
parser.add_argument('--max_epoch', type=int, default=50, help='epochs for training')
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
tf.app.flags.DEFINE_integer('model_add', args.model_add,'address to the trained model')

tf.app.flags.DEFINE_integer('hidden_size', args.hidden_size,'hidden feature size')
tf.app.flags.DEFINE_integer('posi_size', args.posi_size,'position embedding size')

tf.app.flags.DEFINE_integer('testing_batch_size', args.batch_size,'batch size for testing')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',args.keep_prob,'dropout rate')
tf.app.flags.DEFINE_float('seed', 123,'random seed')

tf.app.flags.DEFINE_float('strategy', args.strategy,'training strategy, none, pretrain, ranking, pretrain+ranking')
tf.app.flags.DEFINE_float('rank_topn', 50,'top n complex or simple paths')

def complexity_features(array):
    return np.array([[np.count_nonzero(ele), np.unique(ele).size] for ele in array]).astype(np.float32)
        
def main(_):
    export_path = args.dir_test + '/'
    word_vec = np.load(export_path + 'vec.npy')
    print('reading training data ...')
    
    instance_scope = np.load(export_path + 'scope.npy')
    instance_scope_kg = np.load(export_path + 'scope_kg.npy')
    instance_scope_tx = np.load(export_path + 'scope_tx.npy')
    instance_scope_hy = np.load(export_path + 'scope_hy.npy')

    test_label = np.load(export_path + 'label.npy')
    reltot = cal_reltot(test_label)
    test_label_kg = np.load(export_path + 'label_kg.npy')
    test_label_tx = np.load(export_path + 'label_tx.npy')
    test_label_hy = np.load(export_path + 'label_hy.npy')
    
    test_word = np.load(export_path + 'word.npy')
    test_posi1 = np.load(export_path + 'posi1.npy')
    test_posi2 = np.load(export_path + 'posi2.npy')

    test_word_kg = np.load(export_path + 'word_kg.npy')
    test_posi1_kg = np.load(export_path + 'posi1_kg.npy')
    test_posi2_kg = np.load(export_path + 'posi2_kg.npy')

    test_word_tx = np.load(export_path + 'word_tx.npy')
    test_posi1_tx = np.load(export_path + 'posi1_tx.npy')
    test_posi2_tx = np.load(export_path + 'posi2_tx.npy')

    test_word_hy = np.load(export_path + 'word_hy.npy')
    test_posi1_hy = np.load(export_path + 'posi1_hy.npy')
    test_posi2_hy = np.load(export_path + 'posi2_hy.npy')
        
    test_head = np.load(export_path + 'head.npy')
    test_tail = np.load(export_path + 'tail.npy')

    test_comp_fea = complexity_features(test_word)
    test_comp_fea_kg = complexity_features(test_word_kg)
    test_comp_fea_tx = complexity_features(test_word_tx)
    test_comp_fea_hy = complexity_features(test_word_hy)
    
    print('reading finished')
    print('bags 		: %d' % (len(instance_scope)))
    print('paths		: %d' % (len(test_word)))
    print('relations            : %d' % (FLAGS.num_classes))
    print('word size            : %d' % (len(word_vec[0])))
    print('position size 	: %d' % (FLAGS.posi_size))
    print('hidden size		: %d' % (FLAGS.hidden_size))

    print 'building network...'
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = network.CNN(is_training = True, word_embeddings = word_vec)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

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

    saver.restore(sess, args.model_addr)

    stack_output = []
    stack_label = []
    stack_att = []
    stack_pred = []
    stack_true = []
    stack_scope = []

    iteration = len(test_instance_scope)/FLAGS.batch_size
    for i in range(iteration):
	temp_str= 'running '+str(i)+'/'+str(iteration)+'...'
        sys.stdout.write(temp_str+'\r')
	sys.stdout.flush()
	input_scope = test_instance_scope[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size]
	index = []
	scope = [0]
	label = []

        for num in input_scope:
	    index = index + range(num[0], num[1] + 1)
	    label.append(test_label[num[0]])
	    scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
            
        label_ = np.zeros((FLAGS.test_batch_size, FLAGS.num_classes))
	label_[np.arange(FLAGS.test_batch_size), label] = 1

        output, test_att, test_pred = test_step(test_head[index], test_tail[index], test_word[index,:],
                                                test_posi1[index,:], test_posi2[index,:], test_label[index],
                                                label_, np.array(scope), test_comp_fea[index,:])
        stack_output.append(output)
	stack_label.append(label_)
        stack_att.append(test_att)
        stack_pred.append(test_pred)
        stack_true.extend(label)
        stack_scope.extend(input_scope)

    stack_output = np.concatenate(stack_output, axis=0)
    stack_label = np.concatenate(stack_label, axis = 0)
    stack_att = np.concatenate(stack_att, axis=0)
    stack_pred = np.concatenate(stack_pred, axis=0)
    stack_true = np.array(stack_true)
    stack_scope = np.array(stack_scope)

    exclude_na_flatten_output = stack_output[:,1:]
    exclude_na_flatten_label = stack_label[:,1:]
    score = metrics(stack_label, stack_output)
    score.precision_at_k([500, 1000, 1500, 2000])

    y_pred = np.argmax(stack_output, axis=1)
    y_true = np.argmax(stack_label, axis=1)
    print('Precision: %.3f' % precision_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro'))
    print('Recall: %.3f' % recall_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro'))
    print('F1: %.3f' % f1_score(y_true, y_pred, labels=range(1, stack_label.shape[1]), average='micro'))
                
    average_precision = average_precision_score(exclude_na_flatten_label,exclude_na_flatten_output, average = "micro")
    print('AUC: '+str(average_precision))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    np.save(args.result_dir+'/'+FLAGS.strategy+'_prob'+'.npy', exclude_na_flatten_output)
    np.save(args.result_dir+'/'+FLAGS.strategy+'_label'+'.npy',exclude_na_flatten_label)

if __name__ == "__main__":
    tf.app.run()
