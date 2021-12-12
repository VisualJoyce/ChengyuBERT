![ChengyuBERT](chengyubert1.png)

# ChengyuBERT

A repository for Chinese Idiom/Chengyu Recommendation.

In this repo, we release code for the two papers on Chengyu Recommendation and one paper on Chengyu Embedding Evaluation.

Table of Contents
=================

   * [ChengyuBERT](#chengyubert)
      * [Data](#data)
      * [Pretrained Models](#pretrained-models)
      * [Dual Embeddings](#dual-embeddings)
         * [Preprocessing](#preprocessing)
         * [Training](#training)
         * [Evaluation](#evaluation)
      * [Two Stage](#two-stage)
         * [Preprocessing](#preprocessing-1)
         * [Stage One](#stage-one)
         * [Stage Two for Official](#stage-two-for-official)
         * [Stage Two for Competition](#stage-two-for-competition)
      * [Learning and Evaluating Chinese Idiom Embeddings](#learning-and-evaluating-chinese-idiom-embeddings)
         * [Train Chengyu Embeddings](#train-chengyu-embeddings)
         * [Evaluate Chengyu Embeddings](#evaluate-chengyu-embeddings)
        
      * [Acknowledgement](#acknowledgement)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

## Data
We used the newly released dataset [ChID](https://github.com/zhengcj1/ChID-Dataset).
Users of this repo are encouraged to read their paper to get detailed descriptions of each splits.

The data directory has the following structure
```shell script
(base) mhtan@chase ➜  ChengyuBERT git:(master) ✗ tree data/annotations 
data/annotations
├── competition
│   ├── dev_answer.csv
│   ├── dev.txt
│   ├── idiomDict.json
│   ├── sample_submission.csv
│   ├── test_answer.csv
│   ├── test.txt
│   ├── train_answer.csv
│   └── train.txt
├── idiomList.txt
├── idioms_pretrain.json
├── official
│   ├── dev_data.txt
│   ├── test_data_ord.txt
│   ├── test_data_sim.txt
│   ├── test_data.txt
│   ├── test_out_data.txt
│   └── train_data.txt
└── pretrain
    └── train_data.txt

```

## Pretrained Models
We choose to use pretrained models hosted on [🤗 models](https://huggingface.co/models). 
For example, one can configure `pretrained_model_name_or_path=hfl/chinese-bert-wwm-ext` in the configuration file to use
[chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext).
For other pretrained models, one can get from their online repos and put it into `data/pretrained` as following:
```shell script
(base) mhtan@chase ➜  ChengyuBERT git:(master) ✗ tree data/pretrained 
data/pretrained
├── albert_xlarge_zh
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── roberta_wwm_large_ext
│   ├── bert_config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── wwm_ext
│   ├── bert_config.json
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
└── wwm_ext_pretrain,xinhua-4-21796
    ├── config.json
    ├── pytorch_model.bin
    ├── training_args.bin
    └── vocab.txt

```

## Dual Embeddings

``` bibtex
@inproceedings{tan-jiang-2020-bert,
    title = "A {BERT}-based Dual Embedding Model for {C}hinese Idiom Prediction",
    author = "Tan, Minghuan  and Jiang, Jing",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.113",
    pages = "1312--1322",
    abstract = "Chinese idioms are special fixed phrases usually derived from ancient stories, whose meanings are oftentimes highly idiomatic and non-compositional. The Chinese idiom prediction task is to select the correct idiom from a set of candidate idioms given a context with a blank. We propose a BERT-based dual embedding model to encode the contextual words as well as to learn dual embeddings of the idioms. Specifically, we first match the embedding of each candidate idiom with the hidden representation corresponding to the blank in the context. We then match the embedding of each candidate idiom with the hidden representations of all the tokens in the context thorough context pooling. We further propose to use two separate idiom embeddings for the two kinds of matching. Experiments on a recently released Chinese idiom cloze test dataset show that our proposed method performs better than the existing state of the art. Ablation experiments also show that both context pooling and dual embedding contribute to the improvement of performance.",
}
```

### Preprocessing

On the ChID official released dataset
```shell script
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_train
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_dev
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_test
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_sim
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_ran
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_out
```

On the ChID competition dataset, 
```shell script
CONFIG_FILE="dual_embedding/roberta-wwm-ext-large_competition.json" bash docker_preprocess.sh $PWD/data/annotations competition_train
CONFIG_FILE="dual_embedding/roberta-wwm-ext-large_competition.json" bash docker_preprocess.sh $PWD/data/annotations competition_dev
CONFIG_FILE="dual_embedding/roberta-wwm-ext-large_competition.json" bash docker_preprocess.sh $PWD/data/annotations competition_test
CONFIG_FILE="dual_embedding/roberta-wwm-ext-large_competition.json" bash docker_preprocess.sh $PWD/data/annotations competition_out
```
For more information about the competition, please refer to [Chinese Idiom Understanding Contest](https://www.biendata.xyz/competition/idiom/).

Since our `txt_db` may be preprocessed via different tokenizers, we use the model path or name as part of the db's path.
If the user is sure that the models sharing the same tokenizer and vocabulary, one can use relative soft link to avoid repeated preprocessing.
```shell script
└── txt_db
    ├── hfl
    │   └── chinese-bert-wwm-ext
    │       ├── external_pretrain.db
    │       ├── official_dev.db
    │       ├── official_out.db
    │       ├── official_ran.db
    │       ├── official_sim.db
    │       ├── official_test.db
    │       └── official_train.db
    └── visualjoyce
        └── chengyubert_2stage_stage1_wwm_ext -> ../hfl/chinese-bert-wwm-ext
```

### Training

To run the baseline BL-IdmEmb (w/o EC)
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" \
bash docker_train.sh official \
"MODEL=chengyubert-cloze CANDIDATES=original LEARNING_RATE=0.00005 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

To run the dual model CP+DE
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" \
bash docker_train.sh official \
"MODEL=chengyubert-dual CANDIDATES=combined LEARNING_RATE=0.00005 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

To run the dual model for the competition
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="dual_embedding/roberta-wwm-ext-large_competition.json" \
bash docker_train.sh competition \
"MODEL=chengyubert-dual CANDIDATES=combined LEARNING_RATE=0.00005 NUM_TRAIN_STEPS=5003 GRADIENT_ACCUMULATION_STEPS=5 VALID_STEPS=100 GRAD_NORM=1"
```

### Evaluation  

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" \
bash docker_infer.sh official \
"MODEL=chengyubert-dual CANDIDATES=combined LEARNING_RATE=0.00005 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

## Two Stage

We collect a large corpus as described in the TALLIP paper, and do a two-stage training using this corpus and further fine-tuning over ChID.

```bibtex
@article{10.1145/3453185,
    author = {Tan, Minghuan and Jiang, Jing and Dai, Bing Tian},
    title = {A BERT-Based Two-Stage Model for Chinese Chengyu Recommendation},
    year = {2021},
    issue_date = {November 2021},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {20},
    number = {6},
    issn = {2375-4699},
    url = {https://doi.org/10.1145/3453185},
    doi = {10.1145/3453185},
    abstract = {In Chinese, Chengyu are fixed phrases consisting of four characters. As a type of
    idioms, their meanings usually cannot be derived from their component characters.
    In this article, we study the task of recommending a Chengyu given a textual context.
    Observing some of the limitations with existing work, we propose a two-stage model,
    where during the first stage we re-train a Chinese BERT model by masking out Chengyu
    from a large Chinese corpus with a wide coverage of Chengyu. During the second stage,
    we fine-tune the re-trained, Chengyu-oriented BERT on a specific Chengyu recommendation
    dataset. We evaluate this method on ChID and CCT datasets and find that it can achieve
    the state of the art on both datasets. Ablation studies show that both stages of training
    are critical for the performance gain.},
    journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
    month = aug,
    articleno = {92},
    numpages = {18},
    keywords = {Question answering, Chengyu recommendation, idiom understanding}
}
```

### Preprocessing

For official data, we can reuse the preprocessing above. 
For the collected pretraining corpus, we can process as following:
```shell script
CONFIG_FILE="two_stage/stage1-wwm-ext.json" bash docker_preprocess.sh $PWD/data/annotations external_pretrain
```

### Stage One 
To run the pretraining
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="two_stage/stage1-wwm-ext.json" \
bash docker_train.sh pretrain \
"MODEL=chengyubert-2stage-stage1 CANDIDATES=combined LEARNING_RATE=0.00005 NUM_TRAIN_STEPS=250000 GRADIENT_ACCUMULATION_STEPS=12 VALID_STEPS=100 GRAD_NORM=1"
```

### Stage Two for Official
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="two_stage/stage2-wwm-ext_official.json" \
bash docker_train.sh official \
"MODEL=chengyubert-2stage-stage2 CANDIDATES=combined LEARNING_RATE=0.00005 NUM_TRAIN_STEPS=25000 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

### Stage Two for Competition
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="two_stage/stage2-wwm-ext_competition.json" \
bash docker_train.sh competition \
"MODEL=chengyubert-2stage-stage2 CANDIDATES=combined LEARNING_RATE=0.00005 NUM_TRAIN_STEPS=5000 GRADIENT_ACCUMULATION_STEPS=5 VALID_STEPS=100 GRAD_NORM=1"
```

## Learning and Evaluating Chinese Idiom Embeddings
We study the task of learning and evaluating Chinese idiom embeddings.
We first construct a new evaluation dataset that contains idiom synonyms and antonyms.
Observing that existing Chinese word embedding methods may not be suitable for learning idiom embeddings, we further present a BERT-based method that directly learns embedding vectors for individual idioms.
We empirically compare representative existing methods and our method.
We find that our method substantially outperforms existing methods on the evaluation dataset we have constructed.
```bibtex
@inproceedings{tan-jiang-2021-learning,
    title = "Learning and Evaluating {C}hinese Idiom Embeddings",
    author = "Tan, Minghuan  and
      Jiang, Jing",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)",
    month = sep,
    year = "2021",
    address = "Held Online",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/2021.ranlp-main.155",
    pages = "1387--1396",
    abstract = "We study the task of learning and evaluating Chinese idiom embeddings. We first construct a new evaluation dataset that contains idiom synonyms and antonyms. Observing that existing Chinese word embedding methods may not be suitable for learning idiom embeddings, we further present a BERT-based method that directly learns embedding vectors for individual idioms. We empirically compare representative existing methods and our method. We find that our method substantially outperforms existing methods on the evaluation dataset we have constructed.",
}
```

### Train Chengyu Embeddings
```shell
CUDA_VISIBLE_DEVICES=0,1 CONFIG_FILE="train-embeddings-base-1gpu.json" \
  bash docker_train.sh embeddings "MODEL=chengyubert-ns-cls-mask-300 TRAIN_BATCH_SIZE=11000 NUM_TRAIN_STEPS=500000 MAX_TXT_LEN=16"
```
```shell
CUDA_VISIBLE_DEVICES=0,1 CONFIG_FILE="train-embeddings-base-1gpu.json" \
  bash docker_train.sh embeddings "MODEL=chengyubert-ns-cls-mask-300 TRAIN_BATCH_SIZE=11000 NUM_TRAIN_STEPS=500000 MAX_TXT_LEN=32"
```
### Evaluate Chengyu Embeddings
```shell
CUDA_VISIBLE_DEVICES=7 python eval_embedding.py --model_path data/output/chengyubert-cls-ns-300/wwm_ext/pretrain_4_500003_5e-05/ckpt/model_step_490000.pt
```

## Acknowledgement
The author of this repo learned a lot from the code of the following repos:
* [🤗 Transformers](https://github.com/huggingface/transformers)
* [UNITER](https://github.com/ChenRocks/UNITER)
* [中文BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
