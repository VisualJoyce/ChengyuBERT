# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
ANNOTATION_DIR=$1
SOURCE_SPLIT=$2

WORK_DIR=$(readlink -f .)
DATA_DIR=${WORK_DIR}/data
CONFIG_DIR=/src/config
PRETRAIN_DIR=${DATA_DIR}/pretrained

echo "extracting text features..."
if [ ! -d $OUT_DIR ]; then
  mkdir -p $OUT_DIR
fi

if [ -z "$CONFIG_FILE" ]; then
  CONFIG_FILE="train-chengyubert-base-1gpu.json"
fi

docker run --ipc=host --rm \
  --mount src="${WORK_DIR}",dst=/src,type=bind \
  --mount src="$PRETRAIN_DIR",dst=/pretrain,type=bind,readonly \
  --mount src=$ANNOTATION_DIR,dst=/annotations,type=bind,readonly \
  -w /src vimos/zhuge:latest \
  bash -c "python preprocess.py --annotation $SOURCE_SPLIT --config=$CONFIG_DIR/$CONFIG_FILE"

echo "done"
