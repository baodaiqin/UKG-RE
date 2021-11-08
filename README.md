# UKG-RE
### Dependencies
- python = 2.x
- tensorflow = 1.9.0
- numpy
- sklearn
- tqdm
- tabulate
- NetworX

### Configuration:
Default settings are located in `settings.py` script. It contains the following configuration parameters:

| parameter | description |
|---|---|
| `CUDA_VISIBLE_DEVICES` | to set the gpu device |
| `TF_CPP_MIN_LOG_LEVEL` | disable tensorflow compilation warnings |
| `DIR_TRAIN` | path to the preprocessed dataset for training |
| `DIR_TEST` | path to the preprocessed dataset for testing |
| `MODEL_DIR` | path to store the trained model |
| `NB_BATCH_TRIPLE` | the number of batch for training KGC model |
| `BATCH_SIZE` | batch size of training Distantly Supervised RE model |
| `TESTING_BATCH_SIZE` | batch size for testing |
| `MAX_EPOCH` | epochs for training |
| `MAX_LENGTH` | the maximum number of words in a path |
| `HIDDEN_SIZE` | hidden feature size |
| `POSI_SIZE` | position embedding size |
| `LR` | learning rate for RE model |
| `LR_KGC` | learning rate for KGC model |
| `KEEP_PROB` | dropout rate |
| `MARGIN` | margin for training KGC model |
| `SEED` | random seed for initializing weights |
| `STRATEGY` | training strategy: none, pretrain, ranking and pretrain_ranking|
| `CHECKPOINT_EVERY` | evaluate and save model every n-epoch |
| `RANK_TOPN` | ranking attention over top or last n complex paths |
| `RESULT_DIR` | path to store the results |
| `P_AT_N`| precision@top_n prediction |
| `ADDR_KG_Train` | address of KG triplets for training, e.g., "e1 tab 'location contain' tab e2 \n" |
| `ADDR_KG_Test` | address of KG trplets for testing |
| `ADDR_TX` | address of textual triplets, e.g., "e1 tab 'lived and studied in' tab e2 \n", where the textual relation can be tokenized by space. |
| `ADDR_EMB` | address of pretrained word embeddings from the Word2Vec, e.g., "cases 4.946734 15.195805 6.550739 2.514410 ..." |

### Usage
1. Prepare Knowledge Grpah triplets (i.e., `ADDR_KG_Train` and `ADDR_KG_Test`), Textual triplets (i.e., `ADDR_TX`) and a file of pretrained word embeddings (i.e., `ADDR_EMB`).
2. Preprocess the dataset (i.e., `ADDR_KG_Train`, `ADDR_KG_Test` and `ADDR_TX`) and store the processed data in specified folders (i.e., `DIR_TRAIN` and `DIR_TEST`).

    ~~~~
    python preprocess_ug.py \
      --nb_path 10 \
      --cutoff 3
    ~~~~
    - `nb_path` is the maximum number of paths given an entity pair and a graph (e.g., Textual Graph).
    - `cutoff` is th depth to stop the search of multi-hop path.
    
3. Train your own model on the preprocessed dataset. Necessary static configuration (e.g., `HIDDEN_SIZE`) is located in `settings.py` script as mentioned above.
    ~~~~
    CUDA_VISIBLE_DEVICES=1 python2 ugdsre.py --mode train
    ~~~~

4. Test the trained model.
    ~~~~
    CUDA_VISIBLE_DEVICES=1 python2 ugdsre.py --mode test
    ~~~~
    

### Easy Start
- You can import our package and load the recently trained models.
  ~~~~
  >>> import ugdsre
  >>> model = ugdsre.UGDSRE()
  ~~~~
  
- Then use `infer` to predict the relation given a list of entity pairs `[(e1, e2), (e2, e3), ...]`.
  ~~~~
  >>> list_ep = [(e1, e2), (e2, e3), ...]
  >>> results = model.infer(list_ep, nb_path=10, cutoff=3)
  ~~~~
  - `nb_path` is the maximum number of paths given an entity pair and a graph (e.g., Textual Graph).
  - `cutoff` is th depth to stop the search of multi-hop path.
  - `infer` outputs a list of predicted results.
  -  Notice that some entity pairs lack multi-hop path for predicting their relation, thus `len(results) <= len(list_ep)`.
  
- Check the predicted relation and corresponding confidence score:
  ~~~~
  >>> results[i]['triple_sc']
  (e1, rel1, e2, score)
  ~~~~
  
- Check the supporting multi-hop path evidences and corresponding attention score:
  ~~~~
  >>> results[i]['path_att']
  [(path1, att1), (path2, att2), ...]
  ~~~~
