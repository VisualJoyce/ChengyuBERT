set -x
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
  CUDA_VISIBLE_DEVICES=${DEVICE} CONFIG_FILE=${PROJECT}/${BERT_TYPE}_limit${K}_masked.json \
    bash docker_train.sh ${PROJECT} \
    "NUM_TRAIN_STEPS=${TRAIN_STEPS} VALID_STEPS=${STEPS} GRADIENT_ACCUMULATION_STEPS=8 \
       MODEL=${model} ${opt}"
done

echo "-------------------------------------------------"
tail -n 20 data/output/chengyubert-${PROJECT}-max-pooling_context-${USE_CONTEXT}/${BERT_TYPE}/${BERT_TYPE}_limit${K}.json/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
echo "-------------------------------------------------"
tail -n 20 data/output/chengyubert-${PROJECT}-compose-only_context-${USE_CONTEXT}/${BERT_TYPE}/${BERT_TYPE}_limit${K}.json/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
echo "-------------------------------------------------"
tail -n 20 data/output/chengyubert-${PROJECT}-compose-only-masked_context-${USE_CONTEXT}/${BERT_TYPE}/${BERT_TYPE}_limit${K}_masked.json/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
echo "-------------------------------------------------"
tail -n 20 data/output/chengyubert-${PROJECT}-latent-idiom-masked_context-${USE_CONTEXT}/${BERT_TYPE}/${BERT_TYPE}_limit${K}_masked.json/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
echo "-------------------------------------------------"
tail -n 20 data/output/chengyubert-${PROJECT}-compose-latent-idiom-masked_context-${USE_CONTEXT}/${BERT_TYPE}/${BERT_TYPE}_limit${K}_masked.json/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
echo "-------------------------------------------------"
tail -n 20 data/output/chengyubert-${PROJECT}-latent-idiom-masked-coattention_context-${USE_CONTEXT}/${BERT_TYPE}/${BERT_TYPE}_limit${K}_masked.json/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
echo "-------------------------------------------------"
tail -n 20 data/output/chengyubert-${PROJECT}-latent-idiom-masked-coattention-full_context-${USE_CONTEXT}/${BERT_TYPE}/${BERT_TYPE}_limit${K}_masked.json/${PROJECT}_8_${TRAIN_STEPS}_5e-05_${DROPOUT}/log/log.txt
