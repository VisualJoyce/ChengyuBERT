# ChengyuBERT

A repository for Chinese Idiom/Chengyu Recommendation.

![ZhugeBERT](chengyubert1.png)

In this repo, we release code for the two papers on Chengyu Recommendation.

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
    author = "Tan, Minghuan  and
      Jiang, Jing",
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

```shell script
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_train
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_dev
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_test
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_sim
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_ran
CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" bash docker_preprocess.sh $PWD/data/annotations official_out
```

### Training

To run the baseline BL-IdmEmb (w/o EC)
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" \
bash docker_train.sh official \
"MODEL=chengyubert-cloze CANDIDATES=original LEARNING_RATE=0.0001 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

To run the dual model CP+DE
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" \
bash docker_train.sh official \
"MODEL=chengyubert-dual CANDIDATES=combined LEARNING_RATE=0.0001 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

### Evaluation  

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json" \
bash docker_infer.sh official \
"MODEL=chengyubert-dual CANDIDATES=combined LEARNING_RATE=0.0001 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

## Two Stage

We collect a large corpus as described in the TALLIP paper, and do a two-stage training using this corpus and further fine-tuning over ChID.

```bibtex
@misc{chengyu-2stage,
      title={A BERT-based Two-Stage Model for Chinese Chengyu Recommendation}, 
      author={Minghuan, Tan and Jing, Jiang and Bing Tian Dai},
      year={2020},
      primaryClass={cs.CL}
}
```

### Preprocessing

For official data, we can reuse the preprocessing above. 
For the collected pretraining corpus, we can process as following:
```shell script
CONFIG_FILE="two_stage/stage1-wwm-ext.json" bash docker_preprocess.sh $PWD/data/annotations external_pretrain
```

To run the pretraining
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="two_stage/stage1-wwm-ext.json" \
bash docker_train.sh pretrain \
"MODEL=chengyubert-2stage-stage1 CANDIDATES=combined LEARNING_RATE=0.0001 NUM_TRAIN_STEPS=250000 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

## Acknowledgement
The author of this repo learned a lot from the code of the following repos:
* [🤗 Transformers](https://github.com/huggingface/transformers)
* [UNITER](https://github.com/ChenRocks/UNITER)
* [中文BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
