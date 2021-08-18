import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

class NN(object):
        def calc(self, e, t, r):
		return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

	def __init__(self, is_training, word_embeddings):
		self.max_length = FLAGS.max_length
		self.num_classes = FLAGS.num_classes
		self.word_total, self.word_size = word_embeddings.shape
		self.hidden_size = FLAGS.hidden_size
		self.output_size = FLAGS.hidden_size

                self.margin = FLAGS.margin
		# placeholders for text models
		self.word = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_word')
		self.posi1 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_posi1')
		self.posi2 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_posi2')

		self.label_index = tf.placeholder(dtype=tf.int32,shape=[None], name='label_index')
		self.head_index = tf.placeholder(dtype=tf.int32,shape=[None], name='head_index')
		self.tail_index = tf.placeholder(dtype=tf.int32,shape=[None], name='tail_index')
                
                self.label = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size, self.num_classes], name='input_label')
		self.scope = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope')
		self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
		self.weights = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size])

                self.comp_fea = tf.placeholder(dtype=tf.float32,shape=[None, 2], name='input_comp_fea')
                self.comp_fea_weight = tf.constant([0.7, 0.3], dtype=tf.float32)
                
                # placeholders for kg models
		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])
		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

                with tf.name_scope("embedding-layers"):
			# word embeddings
			self.word_embedding = tf.get_variable(initializer=word_embeddings,name = 'word_embedding',dtype=tf.float32)
                        
                        self.relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, self.output_size*3],dtype=tf.float32,
                                                               initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.seed))
                        self.bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.seed))
			# position embeddings
			temp_posi1_embedding = tf.get_variable('temp_posi1_embedding',[FLAGS.posi_num,FLAGS.posi_size],dtype=tf.float32,
                                                               initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.seed))
			temp_posi2_embedding = tf.get_variable('temp_posi2_embedding',[FLAGS.posi_num,FLAGS.posi_size],dtype=tf.float32,
                                                               initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.seed))
			self.posi1_embedding = tf.concat([temp_posi1_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.posi_size,dtype=np.float32)),[1, FLAGS.posi_size])],0)
			self.posi2_embedding = tf.concat([temp_posi2_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.posi_size,dtype=np.float32)),[1, FLAGS.posi_size])],0)

                        # relation embeddings and the transfer matrix between relations and textual relations
			self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [FLAGS.rel_total, self.word_size],
                                                              initializer = tf.contrib.layers.xavier_initializer(seed=FLAGS.seed))
			self.transfer_matrix = tf.get_variable("transfer_matrix", [self.output_size, self.word_size*1],
                                                               initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.seed))
			self.transfer_bias = tf.get_variable('transfer_bias', [self.word_size*1], dtype=tf.float32,
                                                             initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.seed))
                        
                with tf.name_scope("embedding-lookup"):
			# textual embedding-lookup 
			input_word = tf.nn.embedding_lookup(self.word_embedding, self.word)
			input_posi1 = tf.nn.embedding_lookup(self.posi1_embedding, self.posi1)
			input_posi2 = tf.nn.embedding_lookup(self.posi2_embedding, self.posi2)
			self.input_embedding = tf.concat(values = [input_word, input_posi1, input_posi2], axis = 2)

			# knowledge embedding-lookup 
			pos_h = tf.nn.embedding_lookup(self.word_embedding, self.pos_h)
			pos_t = tf.nn.embedding_lookup(self.word_embedding, self.pos_t)
			pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

                        neg_h = tf.nn.embedding_lookup(self.word_embedding, self.neg_h)
			neg_t = tf.nn.embedding_lookup(self.word_embedding, self.neg_t)
			neg_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

		with tf.name_scope("knowledge_graph"):
			pos = tf.reduce_sum(abs(pos_h + pos_r - pos_t), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h + neg_r - neg_t), 1, keep_dims = True)
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
			for i in range(FLAGS.batch_size):
				path_matrix = x[scope[i]:scope[i+1]]
				attention_score = tf.nn.softmax(tf.reshape(attention_logit[scope[i]:scope[i+1]], [1, -1]))
				final_repre = tf.reshape(tf.matmul(attention_score, path_matrix),[self.output_size])
				tower_repre.append(final_repre)
			if dropout:
				stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
			else:
				stack_repre = tf.stack(tower_repre)
		return stack_repre

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
			for i in range(FLAGS.batch_size):
				path_matrix = x[scope[i]:scope[i+1]]
				attention_score = tf.nn.softmax(tf.reshape(attention_logit[scope[i]:scope[i+1]], [1, -1]))
				final_repre = tf.reshape(tf.matmul(attention_score, path_matrix),[self.output_size])
				tower_repre.append(final_repre)
                                
                                rank_scores_batch = rank_scores[scope[i]:scope[i+1]]
                                rank_size = tf.size(rank_scores_batch)
                                rank_ind = tf.nn.top_k(rank_scores_batch, rank_size)[1]

                                top_n_rank_ind = rank_ind[:FLAGS.rank_topn]
                                last_n_rank_ind = rank_ind[-1 * FLAGS.rank_topn:]

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

class CNN(NN):

	def __init__(self, is_training, word_embeddings):
		NN.__init__(self, is_training, word_embeddings)

		with tf.name_scope("conv-maxpool"):
			input_path = tf.expand_dims(self.input_embedding, axis=1)
			x = tf.layers.conv2d(inputs = input_path, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=FLAGS.seed)) 
			x = tf.reduce_max(x, axis=2)
			x = tf.nn.relu(tf.squeeze(x))
                        
                if FLAGS.strategy in ['none', 'pretrain', 'pretrain+ranking']:
                        stack_repre_part = self.katt(x, self.head_index, self.tail_index, self.scope, is_training)
                        stack_repre = tf.concat([stack_repre_part, stack_repre_part, stack_repre_part], axis=1)
                elif FLAGS.strategy == 'ranking':
                        stack_repre_all, stack_repre_top, stack_repre_last = self.katt_rank(x, self.head_index, self.tail_index, self.scope, is_training)
                        stack_repre = tf.concat([stack_repre_all, stack_repre_top, stack_repre_last], axis=1)
                
		with tf.name_scope("loss"):
			logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
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
				for i in range(FLAGS.testing_batch_size):
                                        test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
                                        final_repre_part = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])

                                        if FLAGS.strategy in ['none', 'pretrain']:
                                                final_repre = tf.concat([final_repre_part, final_repre_part, final_repre_part], axis=1)
                                        elif FLAGS.strategy in ['ranking', 'pretrain+ranking']:
                                                comp_fea = self.comp_fea[self.scope_path[i]:self.scope_path[i+1]]
                                                rank_ind, rank_scores = self.rank_test(comp_fea)
                                                rank_ind_top = rank_ind[:FLAGS.rank_topn]
                                                rank_ind_last = rank_ind[-1 * FLAGS.rank_topn:]

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
                                        output = tf.diag_part(tf.nn.softmax(logits))
                                        test_att = tf.gather(test_attention_score, tf.argmax(output))
                                        test_att_output.append(test_att)
                                        test_pred_output.append(tf.argmax(output))
                                        test_pred_score.append(tf.reduce_max(output))
                                                
					test_tower_output.append(output)
				test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
				self.test_output = test_stack_output
                                self.test_att = tf.concat(test_att_output, 0)
                                self.test_pred = tf.stack(test_pred_output)
                                self.test_sc = tf.stack(test_pred_score)
                                
                                
