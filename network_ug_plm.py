import numpy as np
import transformers
import settings_plm as conf
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class NN(object):
        def calc(self, e, t, r):
                return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

        def __init__(self, is_training, word_embeddings, max_length,
                     num_classes, hidden_size, posi_size, margin, rel_total, seed,
                     batch_size, testing_batch_size, rank_topn):

                self.word_total, self.word_size = word_embeddings.shape
                self.max_length = max_length

                self.num_classes = num_classes
                self.hidden_size = hidden_size
                self.output_size = conf.BERT_SIZE * 2
                self.margin = margin
                self.posi_num = max_length * 2 + 1
                self.posi_size = posi_size
                self.rel_total = rel_total
                self.seed = seed
		
                self.batch_size = batch_size
                self.testing_batch_size = testing_batch_size
                self.rank_topn = rank_topn
                
		# placeholders for text models
                # plm
                self.bert_model = transformers.TFBertModel.from_pretrained(conf.BERT_NAME)
                self.input_ids = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_ids')
                self.att_mask = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='att_mask')
                #self.type_ids = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='type_ids')
                self.ht_ids = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None, 2], name='ht_ids')

                self.label_index = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None], name='label_index')
                self.head_index = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None], name='head_index')
                self.tail_index = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None], name='tail_index')
                
                self.label = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None, self.num_classes], name='input_label')
                self.scope = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None], name='scope')
                self.keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name='keep_prob')
                self.weights = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None])
                
                self.comp_fea = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None, 2], name='input_comp_fea')
                self.comp_fea_weight = tf.constant([0.7, 0.3], dtype=tf.float32)
                
                # placeholders for kg models
                self.pos_h = tf.compat.v1.placeholder(tf.int32, [None])
                self.pos_t = tf.compat.v1.placeholder(tf.int32, [None])
                self.pos_r = tf.compat.v1.placeholder(tf.int32, [None])
                self.neg_h = tf.compat.v1.placeholder(tf.int32, [None])
                self.neg_t = tf.compat.v1.placeholder(tf.int32, [None])
                self.neg_r = tf.compat.v1.placeholder(tf.int32, [None])

                initializer = tf.initializers.GlorotUniform(seed=self.seed)
                with tf.name_scope("embedding-layers"):
			# word embeddings
                        self.word_embedding = tf.compat.v1.get_variable(initializer=word_embeddings,name = 'word_embedding',dtype=tf.float32)
                        
                        self.relation_matrix = tf.compat.v1.get_variable('relation_matrix',[self.num_classes, self.output_size*3],dtype=tf.float32,
                                                                         initializer=initializer)
                        self.bias = tf.compat.v1.get_variable('bias',[self.num_classes],dtype=tf.float32,
                                                              initializer=initializer)

                        self.relation_matrix_loc = tf.compat.v1.get_variable('relation_matrix_loc',[self.num_classes, self.output_size*1],dtype=tf.float32,
                                                                             initializer=initializer)
                        self.bias_loc = tf.compat.v1.get_variable('bias_loc',[self.num_classes],dtype=tf.float32,
                                                                  initializer=initializer)
                        
                        # relation embeddings and the transfer matrix between relations and textual relations
                        self.rel_embeddings = tf.compat.v1.get_variable(name = "rel_embedding", shape = [self.rel_total, self.word_size],
                                                                        initializer = initializer)
                        self.transfer_matrix = tf.compat.v1.get_variable("transfer_matrix", [self.output_size, self.word_size*1],
                                                                         initializer = initializer)
                        self.transfer_bias = tf.compat.v1.get_variable('transfer_bias', [self.word_size*1], dtype=tf.float32,
                                                                       initializer = initializer)
                        
                with tf.name_scope("embedding-lookup"):
			# knowledge embedding-lookup
                        pos_h = tf.nn.embedding_lookup(self.word_embedding, self.pos_h)
                        pos_t = tf.nn.embedding_lookup(self.word_embedding, self.pos_t)
                        pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

                        neg_h = tf.nn.embedding_lookup(self.word_embedding, self.neg_h)
                        neg_t = tf.nn.embedding_lookup(self.word_embedding, self.neg_t)
                        neg_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

                with tf.name_scope("knowledge_graph"):
                        pos = tf.reduce_sum(abs(pos_h + pos_r - pos_t), 1, keepdims = True)
                        neg = tf.reduce_sum(abs(neg_h + neg_r - neg_t), 1, keepdims = True)
                        self.loss_kgc = tf.reduce_sum(tf.norm(pos - neg + self.margin))
                        
        def transfer(self, x):
                res = tf.nn.bias_add(tf.matmul(x, self.transfer_matrix), self.transfer_bias)
                return res

        def katt(self, x, head_index, tail_index, scope, is_training = True, dropout = True):
                with tf.name_scope("knowledge-based-attention"):
                        head_e = tf.nn.embedding_lookup(self.word_embedding, head_index)
                        tail_e = tf.nn.embedding_lookup(self.word_embedding, tail_index)
                        kg_att = head_e - tail_e
                        attention_logit = tf.reduce_sum(self.transfer(x) * kg_att, 1)
                        tower_repre = []
                        for i in range(self.batch_size):
                                path_matrix = x[scope[i]:scope[i+1]]
                                attention_score = tf.nn.softmax(tf.reshape(attention_logit[scope[i]:scope[i+1]], [1, -1]))
                                final_repre = tf.reshape(tf.matmul(attention_score, path_matrix),[self.output_size])
                                tower_repre.append(final_repre)
                        if dropout:
                                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
                        else:
                                stack_repre = tf.stack(tower_repre)
                return stack_repre

        def katt_locloss(self, x, head_index, tail_index, scope, is_training = True, dropout = True):
                with tf.name_scope("knowledge-based-attention"):
                        head_e = tf.nn.embedding_lookup(self.word_embedding, head_index)
                        tail_e = tf.nn.embedding_lookup(self.word_embedding, tail_index)
                        kg_att = head_e - tail_e
                        attention_logit = tf.reduce_sum(self.transfer(x) * kg_att, 1)
                        tower_repre = []

                        tower_repre_kg = []
                        tower_repre_tx = []
                        tower_repre_hy = []

                        for i in range(self.batch_size):
                                i = i * 3
                                path_matrix = x[scope[i]:scope[i+3]]#e.g., scope: [0, 3, 7, 14, ...]
                                attention_score = tf.nn.softmax(tf.reshape(attention_logit[scope[i]:scope[i+3]], [1, -1]))
                                final_repre = tf.reshape(tf.matmul(attention_score, path_matrix),[self.output_size])
                                tower_repre.append(final_repre)

                                path_matrix_kg = x[scope[i]:scope[i+1]]
                                attention_score_kg = tf.nn.softmax(tf.reshape(attention_logit[scope[i]:scope[i+1]], [1, -1]))
                                final_repre_kg = tf.reshape(tf.matmul(attention_score_kg, path_matrix_kg),[self.output_size])
                                tower_repre_kg.append(final_repre_kg)

                                path_matrix_tx = x[scope[i+1]:scope[i+2]]
                                attention_score_tx = tf.nn.softmax(tf.reshape(attention_logit[scope[i+1]:scope[i+2]], [1, -1]))
                                final_repre_tx = tf.reshape(tf.matmul(attention_score_tx, path_matrix_tx),[self.output_size])
                                tower_repre_tx.append(final_repre_tx)

                                path_matrix_hy = x[scope[i+2]:scope[i+3]]
                                attention_score_hy = tf.nn.softmax(tf.reshape(attention_logit[scope[i+2]:scope[i+3]], [1, -1]))
                                final_repre_hy = tf.reshape(tf.matmul(attention_score_hy, path_matrix_hy),[self.output_size])
                                tower_repre_hy.append(final_repre_hy)

                        if dropout:
                                stack_repre = tf.compat.v1.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
                                stack_repre_kg = tf.compat.v1.layers.dropout(tf.stack(tower_repre_kg), rate = self.keep_prob, training = is_training)
                                stack_repre_tx = tf.compat.v1.layers.dropout(tf.stack(tower_repre_tx), rate = self.keep_prob, training = is_training)
                                stack_repre_hy = tf.compat.v1.layers.dropout(tf.stack(tower_repre_hy), rate = self.keep_prob, training = is_training)
                        else:
                                stack_repre = tf.stack(tower_repre)
                                stack_repre_kg = tf.stack(tower_repre_kg)
                                stack_repre_tx = tf.stack(tower_repre_tx)
                                stack_repre_hy = tf.stack(tower_repre_hy)
                
                return stack_repre, stack_repre_kg, stack_repre_tx, stack_repre_hy

        def katt_rank(self, x, head_index, tail_index, scope, is_training = True, dropout = True):
                with tf.name_scope("knowledge-based-attention"):
                        head_e = tf.nn.embedding_lookup(self.word_embedding, head_index)
                        tail_e = tf.nn.embedding_lookup(self.word_embedding, tail_index)
                        kg_att = head_e - tail_e
                        attention_logit = tf.reduce_sum(self.transfer(x) * kg_att, 1)
                        tower_repre = []
                        tower_repre_top = []
                        tower_repre_last = []

                        rank_scores = tf.sigmoid(tf.reduce_sum(self.comp_fea * self.comp_fea_weight, 1))#(B, 1)
                        for i in range(self.batch_size):
                                path_matrix = x[scope[i]:scope[i+1]]
                                attention_score = tf.nn.softmax(tf.reshape(attention_logit[scope[i]:scope[i+1]], [1, -1]))
                                final_repre = tf.reshape(tf.matmul(attention_score, path_matrix),[self.output_size])
                                tower_repre.append(final_repre)

                                rank_scores_batch = rank_scores[scope[i]:scope[i+1]]
                                rank_size = tf.size(rank_scores_batch)
                                rank_ind = tf.nn.top_k(rank_scores_batch, rank_size)[1]

                                top_n_rank_ind = rank_ind[:self.rank_topn]
                                last_n_rank_ind = rank_ind[-1 * self.rank_topn:]

                                path_matrix_top = tf.gather(path_matrix, top_n_rank_ind)
                                path_matrix_last = tf.gather(path_matrix, last_n_rank_ind)

                                attention_logit_top = tf.gather(attention_logit[scope[i]:scope[i+1]], top_n_rank_ind)
                                attention_logit_last = tf.gather(attention_logit[scope[i]:scope[i+1]], last_n_rank_ind)

                                attention_sc_top = tf.nn.softmax(tf.reshape(attention_logit_top, [1, -1]))
                                attention_sc_last = tf.nn.softmax(tf.reshape(attention_logit_last, [1, -1]))

                                final_repre_top = tf.reshape(tf.matmul(attention_sc_top, path_matrix_top),[self.output_size])
                                final_repre_last = tf.reshape(tf.matmul(attention_sc_last, path_matrix_last),[self.output_size])
                                
                                tower_repre_top.append(final_repre_top)
                                tower_repre_last.append(final_repre_last)

                        if dropout:
                                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
                                stack_repre_top = tf.layers.dropout(tf.stack(tower_repre_top), rate = self.keep_prob, training = is_training)
                                stack_repre_last = tf.layers.dropout(tf.stack(tower_repre_last), rate = self.keep_prob, training = is_training)
                        else:
                                stack_repre = tf.stack(tower_repre)
                                stack_repre_top = tf.stack(tower_repre_top)
                                stack_repre_last = tf.stack(tower_repre_last)
                return stack_repre, stack_repre_top, stack_repre_last

        def katt_test(self, x, head_index, tail_index, is_training = False):
                head_e = tf.nn.embedding_lookup(self.word_embedding, head_index)
                tail_e = tf.nn.embedding_lookup(self.word_embedding, tail_index)
                ht = head_e - tail_e
                each_att = tf.expand_dims(ht, -1)
                kg_att = tf.concat([each_att for i in range(self.num_classes)], 2)
                x = tf.reshape(self.transfer(x), [-1, 1, self.word_size*1])
                test_attention_logit = tf.matmul(x, kg_att)
                return tf.reshape(test_attention_logit, [-1, self.num_classes])

        def rank_test(self, comp_fea):
                rank_scores = tf.sigmoid(tf.reduce_sum(comp_fea * self.comp_fea_weight, 1))
                rank_size = tf.size(rank_scores)
                rank_ind = tf.nn.top_k(rank_scores, rank_size)[1]
                return rank_ind, rank_scores

class PLM(NN):

	def __init__(self, is_training, word_embeddings, max_length,
                     num_classes, hidden_size, posi_size, margin, rel_total, seed,
                     batch_size, testing_batch_size, rank_topn, strategy, is_inferring=False, nb_bags=None):

                self.strategy = strategy
                self.is_inferring = is_inferring
                self.nb_bags = nb_bags

                NN.__init__(self, is_training, word_embeddings, max_length,
                            num_classes, hidden_size, posi_size, margin, rel_total, seed,
                            batch_size, testing_batch_size, rank_topn)

                with tf.name_scope("bert-encoder"):
                        # plm
                        last_hidden_state, x_ = self.bert_model(
                                self.input_ids,
                                attention_mask = self.att_mask,
                                )
                        
                        ht_last_hidden_state = tf.compat.v1.batch_gather(last_hidden_state, self.ht_ids)
                        x = tf.reshape(ht_last_hidden_state, (-1, conf.BERT_SIZE * 2))
                        
                if self.strategy in ['none', 'pretrain', 'pretrain_ranking']:
                        stack_repre_part = self.katt(x, self.head_index, self.tail_index, self.scope, is_training)
                        stack_repre = tf.concat([stack_repre_part, stack_repre_part, stack_repre_part], axis=1)
                elif self.strategy == 'ranking':
                        stack_repre_all, stack_repre_top, stack_repre_last = self.katt_rank(x, self.head_index, self.tail_index, self.scope, is_training)
                        stack_repre = tf.concat([stack_repre_all, stack_repre_top, stack_repre_last], axis=1)
                elif self.strategy == 'locloss':
                        stack_repre_part, stack_repre_kg, stack_repre_tx, stack_repre_hy = self.katt_locloss(x, self.head_index, self.tail_index, self.scope, is_training)
                        stack_repre = tf.concat([stack_repre_part, stack_repre_part, stack_repre_part], axis=1)

                with tf.name_scope("loss"):
                        logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
                        self.loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
                        if self.strategy == 'locloss':
                                logits_kg = tf.matmul(stack_repre_kg, tf.transpose(self.relation_matrix_loc)) + self.bias_loc
                                logits_tx = tf.matmul(stack_repre_tx, tf.transpose(self.relation_matrix_loc)) + self.bias_loc
                                logits_hy = tf.matmul(stack_repre_hy, tf.transpose(self.relation_matrix_loc)) + self.bias_loc
                                loss_kg = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits_kg, weights = self.weights)
                                loss_tx = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits_tx, weights = self.weights)
                                loss_hy = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits_hy, weights = self.weights)
                                loss_loc = 1.0*loss_kg + 1.0*loss_tx + 1.0*loss_hy
                                self.loss = self.loss + loss_loc

                        self.output = tf.nn.softmax(logits)

                        self.predictions = tf.argmax(logits, 1, name="predictions")
                        self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
                        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

                if not is_training:
                        with tf.name_scope("test"):
                                test_attention_logit = self.katt_test(x, self.head_index, self.tail_index)

                                test_tower_output = []
                                test_att_output = []
                                test_pred_output = []
                                test_pred_score = []
                                
                                if not is_inferring:
                                        nb_data = self.testing_batch_size
                                else:
                                        nb_data = self.nb_bags

                                for i in range(nb_data):
                                        test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                                        final_repre_part = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])

                                        if self.strategy in ['none', 'pretrain', 'locloss']:
                                                final_repre = tf.concat([final_repre_part, final_repre_part, final_repre_part], axis=1)
                                        elif self.strategy in ['ranking', 'pretrain_ranking']:
                                                comp_fea = self.comp_fea[self.scope[i]:self.scope[i+1]]
                                                rank_ind, rank_scores = self.rank_test(comp_fea)
                                                rank_ind_top = rank_ind[:self.rank_topn]
                                                rank_ind_last = rank_ind[-1 * self.rank_topn:]

                                                x_batch = x[self.scope[i]:self.scope[i+1]]
                                                x_batch_top = tf.gather(x_batch, rank_ind_top)
                                                x_batch_last = tf.gather(x_batch, rank_ind_last)

                                                attention_logit_batch = test_attention_logit[self.scope[i]:self.scope[i+1],:]
                                                attention_logit_batch_top = tf.gather(attention_logit_batch, rank_ind_top)
                                                attention_logit_batch_last = tf.gather(attention_logit_batch, rank_ind_last)

                                                attention_score_batch_top = tf.nn.softmax(tf.transpose(attention_logit_batch_top))
                                                attention_score_batch_last = tf.nn.softmax(tf.transpose(attention_logit_batch_last))

                                                final_repre_top = tf.matmul(attention_score_batch_top, x_batch_top)
                                                final_repre_last = tf.matmul(attention_score_batch_last, x_batch_last)
                                                
                                                final_repre = tf.concat([final_repre_part, final_repre_top, final_repre_last], axis=1)

                                        logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                                        output = tf.compat.v1.diag_part(tf.nn.softmax(logits))
                                        test_att = tf.gather(test_attention_score, tf.argmax(output))
                                        test_att_output.append(test_att)
                                        test_pred_output.append(tf.argmax(output))
                                        test_pred_score.append(tf.reduce_max(output))

                                        test_tower_output.append(output)
                                test_stack_output = tf.reshape(tf.stack(test_tower_output),[nb_data, self.num_classes])
                                self.test_output = test_stack_output
                                self.test_att = tf.concat(test_att_output, 0, name="test_att")
                                self.test_pred = tf.stack(test_pred_output, name="test_pred")
                                self.test_sc = tf.stack(test_pred_score, name="test_sc")
                                
