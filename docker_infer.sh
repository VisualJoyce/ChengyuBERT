# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
WORK_DIR=$(readlink -f .)
DATA_DIR=${WORK_DIR}/data
SUB_PROJECT="$1"
MODEL_PARA="$2"
CONFIG_DIR=/src/config
TXT_DB=${DATA_DIR}/txt_db
OUTPUT=${DATA_DIR}/output
PRETRAIN_DIR=${DATA_DIR}/pretrained
ANNOTATION_DIR=${DATA_DIR}/annotations

if [ -z "$MODEL_PARA" ]; then
  MODEL_PARA=""
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  CUDA_VISIBLE_DEVICES='all'
fi

if [ -z "$CONFIG_FILE" ]; then
  CONFIG_FILE="train-${SUB_PROJECT}-base-1gpu.json"
fi

if [ ! -d "${OUTPUT}" ]; then
  mkdir -p "${OUTPUT}"
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
  --mount src="${WORK_DIR}",dst=/src,type=bind \
  --mount src="$OUTPUT",dst=/storage,type=bind \
  --mount src="$PRETRAIN_DIR",dst=/pretrain,type=bind,readonly \
  --mount src=$ANNOTATION_DIR,dst=/annotations,type=bind,readonly \
  --mount src="$TXT_DB",dst=/txt,type=bind \
  -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -w /src vimos/uniter_ve:latest \
  bash -c " PYTHONPATH=/src ${MODEL_PARA} \\
    python train_${SUB_PROJECT}.py --config=$CONFIG_DIR/$CONFIG_FILE --mode infer"
