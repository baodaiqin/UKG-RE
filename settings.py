"""
Settings for the mdoel.
"""

import os

# to set the gpu device
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#path to the preprocessed dataset for training
DIR_TRAIN = './train_initialized'

#path to the preprocessed dataset for testing
DIR_TEST = './test_initialized'

#path to store the trained model
MODEL_DIR = './model_saved'

#the number of batch for training KGC model
NB_BATCH_TRIPLE = 200

#batch size of training Distantly Supervised RE model
BATCH_SIZE = 50

#batch size for testing
TESTING_BATCH_SIZE = 50

#epochs for training
MAX_EPOCH = 100

#the maximum number of words in a path
MAX_LENGTH = 120

#hidden feature size
HIDDEN_SIZE = 100

#position embedding size
POSI_SIZE = 5

#learning rate for RE model
LR = 0.02

#learning rate for KGC model
LR_KGC = 0.02

#dropout rate
KEEP_PROB = 0.5

#margin for training KGC model
MARGIN = 1.0

#random seed for initializing weights
SEED = 123

#training strategy: none, pretrain, ranking, pretrain_ranking, locloss
STRATEGY = 'locloss'

#evaluate and save model every n-epoch
CHECKPOINT_EVERY = 2

#ranking attention over top or last n complex paths
RANK_TOPN = 5

#path to store the results
RESULT_DIR = './results'

#precision@top_n prediction
P_AT_N = [500, 1000, 1500]

#address of KG triplets for training
ADDR_KG_Train = 'data/kg_train.txt'

#address of KG trplets for testing
ADDR_KG_Test = 'data/kg_test.txt'

#address of textual triplets
ADDR_TX = 'data/tx.txt'

#address of pretrained word embeddings
ADDR_EMB = 'data/vec.txt'
