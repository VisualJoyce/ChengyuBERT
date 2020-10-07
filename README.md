# ChengyuBERT

A repository for Chinese Idiom/Chengyu Recommendation.

In this repo, we release code for the two papers 

## Data
We used the newly released dataset ChID[https://github.com/zhengcj1/ChID-Dataset].
Users of this repo are encouraged to read their paper to get detailed descriptions of each splits.

## Data Preprocessing
```shell script
CONFIG_FILE="train-balanced-contrastivepair-1gpu.json" bash preprocess.sh chengyu $PWD/data/annotations balanced_train
```

## Dual Embeddings
### Training and Evaluation  
```shell script
CUDA_VISIBLE_DEVICES=7 CONFIG_FILE="train-official-bert-base-1gpu.json" \
bash docker_train.sh official "MODEL=bert-dual-vocab"
```

## Idiom-oriented Pretraning
### Training and Evaluation  
```shell script
CUDA_VISIBLE_DEVICES=7 CONFIG_FILE="train-official-bert-base-1gpu.json" \
bash docker_train.sh official "MODEL=bert-dual-vocab"
```

## Acknowledgement
The author of this repo learned a lot from the code of the following repos:
