#set -x
PROJECT=$1
BERT_TYPE=$2
DEVICE=$3
K=$4
DROPOUT=$5
WEIGHT_DECAY=$6
MORE_OPTS=$7
USE_CONTEXT=False
STEPS=$(expr $K \* 5)
TRAIN_STEPS=$(expr $K \* 500)

declare -a models=(
  affection-max-pooling
  affection-max-pooling-masked
  affection-max-pooling-masked-latent-idiom
  affection-max-pooling-masked-latent-idiom-with-gate
  affection-compose
  affection-compose-masked
  affection-compose-masked-latent-idiom
  affection-compose-masked-latent-idiom-with-gate
  affection-coattention-masked
  affection-coattention-masked-latent-idiom
  affection-coattention-masked-latent-idiom-with-gate
  affection-coattention-masked-full
  affection-coattention-masked-full-latent-idiom
)

declare -a configs=(
  ${BERT_TYPE}_limit${K}.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
)

declare -a opts=(
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS}"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS}"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} WEIGHT_DECAY=${WEIGHT_DECAY} ${MORE_OPTS} USE_UNLABELED=1"
)

echo "$K $STEPS"
for ((i = 0; i < ${#models[*]}; ++i)); do
  model="${models[$i]}"
  opt="${opts[$i]}"
  config="${configs[$i]}"
  CUDA_VISIBLE_DEVICES=${DEVICE} CONFIG_FILE=affection/${PROJECT}/${config} \
    bash docker_train.sh affection \
    "NUM_TRAIN_STEPS=${TRAIN_STEPS} VALID_STEPS=${STEPS} GRADIENT_ACCUMULATION_STEPS=1 \
       MODEL=${model} ${opt}"
done

py_script="
import sys
import pandas as pda

data = {}
for l in sys.stdin:
    if f'validation on test split' in l:
        data.setdefault('test', {})
        split = 'test'
    elif f'validation on val split' in l:
        data.setdefault('val', {})
        split = 'val'
    else:
        if 'score:' in l:
            name, value = l.split(':', 1)
            name = name.replace(' ', '_')
            value, _ = value.rsplit(',', 1)
            value = eval(value)
            if isinstance(value, dict):
                for k in value:
                    data[split][f'idiom-wise_{k}'] = value[k]
            else:
                data[split][name] = value
print(data)
print(pda.DataFrame.from_records(data).transpose())
"

for ((i = 0; i < ${#models[*]}; ++i)); do
  echo "-------------------------------------------------"
  model="${models[$i]}"
  opt="${opts[$i]}"
  config="${configs[$i]}"
  echo "$model $opt"
  log_txt=data/output/${model}_context-${USE_CONTEXT}/${BERT_TYPE}/${config}/${PROJECT}_1_${TRAIN_STEPS}_5e-05_${DROPOUT}_${WEIGHT_DECAY}/log/log.txt
  cat ${log_txt} | grep "on test split" -A12 | python -c "$py_script"
done
