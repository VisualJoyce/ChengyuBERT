# ChengyuBERT

A repository for Chinese Idiom/Chengyu Recommendation.

In this repo, we release code for the two papers 
``` bibtex
@inproceedings{chengyu-dual,
    title = "A BERT-based Dual Embedding Model for Chinese Idiom Prediction",
    author = "Minghuan, Tan  and
      Jing, Jiang",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}

@misc{chengyu-2stage,
      title={A BERT-based Two-Stage Model for Chinese Chengyu Recommendation}, 
      author={Minghuan, Tan and Jing, Jiang and Bing Tian Dai},
      year={2020},
      primaryClass={cs.CL}
}
```

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
For pretrained models, one can get from online repos and put it into `data/pretrained` as following:
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

### Preprocessing
```shell script
CONFIG_FILE="train-official-bert-base-1gpu.json" bash docker_preprocess.sh $PWD/data/annotations official_train
CONFIG_FILE="train-official-bert-base-1gpu.json" bash docker_preprocess.sh $PWD/data/annotations official_dev
CONFIG_FILE="train-official-bert-base-1gpu.json" bash docker_preprocess.sh $PWD/data/annotations official_test
CONFIG_FILE="train-official-bert-base-1gpu.json" bash docker_preprocess.sh $PWD/data/annotations official_sim
CONFIG_FILE="train-official-bert-base-1gpu.json" bash docker_preprocess.sh $PWD/data/annotations official_ran
CONFIG_FILE="train-official-bert-base-1gpu.json" bash docker_preprocess.sh $PWD/data/annotations official_out
```
### Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="train-official-bert-base-1gpu.json" \
bash docker_train.sh official \
"MODEL=chengyubert-dual ENLARGED_CANDIDATES=1 LEARNING_RATE=0.0001 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```
### Evaluation  
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="train-official-bert-base-1gpu.json" \
bash docker_infer.sh official \
"MODEL=chengyubert-dual ENLARGED_CANDIDATES=1 LEARNING_RATE=0.0001 NUM_TRAIN_STEPS=15003 GRADIENT_ACCUMULATION_STEPS=1 VALID_STEPS=100 GRAD_NORM=1"
```

## Acknowledgement
The author of this repo learned a lot from the code of the following repos:
* [🤗 Transformers](https://github.com/huggingface/transformers)
* [UNITER](https://github.com/ChenRocks/UNITER)
* [中文BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)