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
Static settings are located in `settings.py` script. It contains the following configuration parameters:

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
| `ADDR_KG_Train` | address of KG triplets for training, e.g., "e1 \t 'location contain' \t e2" |
| `ADDR_KG_Test` | address of KG trplets for testing |
| `ADDR_TX` | address of textual triplets, e.g., "e1 \t 'lived and studied in' \t e2" |
| `ADDR_EMB` | address of pretrained word embeddings from the Word2Vec, e.g., "cases 4.946734 15.195805 6.550739 2.514410 ..." |

### Dataset Format:
~~~~
{
  "word2id": {"w1": 0, "w2": 1, ...},
  "word2vec": {"w1": [ele1, ele2, ...], "w2": [ele1, ele2, ...], ...},
  "relation2id": {"rel1": 0, "rel2": 1, ...},
  "triples": [["h1", "t1", "rel1"], ["h2", "t2", "rel2"], ...],
  "train":[
    {
      "e1_id": "h1",
      "e2_id": "t1",
      "e1_word": "w1",
      "e2_word": "w2",
      "relation": "rel1",
      "kg_paths": [[kp1_w1, kp1_w2, ...], [kp2_w1, kp2_w2, ...], ...],
      "textual_paths": [[tp1_w1, tp1_w2, ...], [tp2_w1, tp2_w2, ...], ...],
      "hybrid_paths": [[hp1_w1, hp1_w2, ...], [hp2_w1, hp2_w2, ...], ...],
      "kg_path_e1_e2_positions": [[kp1_e1_posi, kp1_e2_posi], [kp2_e1_posi, kp2_e2_posi], ...],
      "textual_path_e1_e2_positions": [[tp1_e1_posi, tp1_e2_posi], [tp2_e1_posi, tp2_e2_posi], ...],
      "hybrid_path_e1_e2_positions": [[kp1_e1_posi, kp1_e2_posi], [kp2_e1_posi, kp2_e2_posi], ...]
    },
    {}, ...
  ],
  "test": [{same as "train"}, ...]
}
~~~~
   - `word2id` is the mapping of word to its id.
   - `word2vec` is the mapping of word to its word vector.
   - `relation2id` is the mapping of relation to its id.
   - `triples` is the list of KG triples for training a KGC model.
   - Each entry fo `train` and `test` is a bag of paths, where
      - `e1_id` and `e2_id` are the KG id of target entity `e1` and entity `e2`.
      - `relation` is the relation between target entity `e1` and entity `e2`.
      - `e1_word` and `e2_word` are the textual expression of target entity `e1` and entity `e2`.
      - `kg_paths` is the bag of paths only consist of knowledge graph edges.
      - `textual_paths` is the bag of paths only consist of textual edges.
      - `hybrid_paths` is the bag of paths consist of both knowledge and textual edges.
      - `kg_path_e1_e2_positions` contains the position of the target entity `e1` and entity `e2` in each path of `kg_paths`.
      - `textual_path_e1_e2_positions` contains the position of the target entity `e1` and entity `e2` in each path of `textual_paths`.
      - `hybrid_path_e1_e2_positions` contains the position of the target entity `e1` and entity `e2` in each path of `hybrid_paths`.

### Usage
1. Prepare the dataset (e.g., `dataset.json`) in the format as introduced above.
2. Preprocess the dataset, and store the processed data in specified folders (e.g., `./train_initialized` and `./test_initialized`).

    ~~~~
    python preprocess.py 
        --data dataset.json \
        --fixlen 120 \
        --dir_out_train ./train_initialized \
        --dir_out_test ./test_initialized \
    ~~~~
3. Train your own model on the preprocessed dataset. Static configuration (e.g., hidden size) is located in `settings.py` script.
    ~~~~
    CUDA_VISIBLE_DEVICES=0 python ugdsre.py \
        --mode train \
        --dir_train ./train_initialized \
        --model_dir ./model_saved \
        --learning_rate 0.02 \
        --max_epoch 50 \
        --strategy pretrain+ranking 
    ~~~~
5. Test the trained model.
    ~~~~
    CUDA_VISIBLE_DEVICES=0 python ugdsre.py \
        --mode test \
        --dir_test ./test_initialized \
        --model_addr ./model_saved/... 
    ~~~~
    

### Easy Start
- You can import our package and load pre-trained models.
~~~~
>>> import ugdsre
>>> model = ugdsre.UGDSRE(dir_train='folder_of_the_preprocessed_trainig_data', model_dir='address_to_the_trained_model')
~~~~
- Then use `infer` to do bag-level relation extraction from multi-hop `paths`.
~~~~
>>> model.infer([
                  {"e1_id": "h1", "e2_id": "t1", "e1_word": "w1", "e2_word": "w2", 
                  "paths": [["p1_w1", "p1_w2", ...], ...], 
                  "path_e1_e2_positions": [[p1_e1_posi, p1_e2_posi], ...]}, ...
                  ])

[("rel1", score), ...]
~~~~
