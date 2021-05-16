#set -x
PROJECT=$1
BERT_TYPE=$2
DEVICE=$3
K=$4
DROPOUT=$5
MORE_OPTS=$6
USE_CONTEXT=False
STEPS=$(expr $K \* 5)
TRAIN_STEPS=$(expr $K \* 500)

declare -a models=(
  chengyubert-${PROJECT}-max-pooling
  chengyubert-${PROJECT}-compose-only
  chengyubert-${PROJECT}-compose-only-masked
  chengyubert-${PROJECT}-latent-idiom-masked
  chengyubert-${PROJECT}-compose-latent-idiom-masked
  chengyubert-${PROJECT}-latent-idiom-masked-coattention
  chengyubert-${PROJECT}-latent-idiom-masked-coattention-full
)

declare -a configs=(
  ${BERT_TYPE}_limit${K}.json
  ${BERT_TYPE}_limit${K}.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
  ${BERT_TYPE}_limit${K}_masked.json
)

declare -a opts=(
  "DROPOUT=${DROPOUT} ${MORE_OPTS}"
  "DROPOUT=${DROPOUT} ${MORE_OPTS}"
  "DROPOUT=${DROPOUT} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} ${MORE_OPTS} USE_UNLABELED=1"
  "DROPOUT=${DROPOUT} ${MORE_OPTS} USE_UNLABELED=1"
)

echo "$K $STEPS"
for ((i = 0; i < ${#models[*]}; ++i)); do
  model="${models[$i]}"
  opt="${opts[$i]}"
  config="${configs[$i]}"
  CUDA_VISIBLE_DEVICES=${DEVICE} CONFIG_FILE=${PROJECT}/${config} \
    bash docker_train.sh ${PROJECT} \
    "NUM_TRAIN_STEPS=${TRAIN_STEPS} VALID_STEPS=${STEPS} GRADIENT_ACCUMULATION_STEPS=8 \
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
            value, _ = value.rsplit(',', 1)
            value = eval(value)
            if isinstance(value, dict):
                for k in value:
                    data[split][f'idiom-wise_{k}'] = value[k]
            else:
                data[split][name] = value
print(data)
print(pda.DataFrame.from_records(data))
"

for ((i = 0; i < ${#models[*]}; ++i)); do
  echo "-------------------------------------------------"
  model="${models[$i]}"
  opt="${opts[$i]}"
  config="${configs[$i]}"
  echo "$model"
  log_txt=data/output/${model}_context-${USE_CONTEXT}/${BERT_TYPE}/${config}/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
  cat ${log_txt} | grep "on test split" -A12 | python -c "$py_script"
done
