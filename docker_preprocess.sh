# Licensed under the MIT license.
cat smusg.logo
ANNOTATION_DIR=$1
SOURCE_SPLIT=$2

WORK_DIR=$(readlink -f .)
DATA_DIR=${WORK_DIR}/data
PRETRAIN_DIR=${DATA_DIR}/pretrained
TXT_DB=${DATA_DIR}/txt_db
CACHE_DIR=${DATA_DIR}/.cache

if [ ! -d "${TXT_DB}" ]; then
  mkdir -p "${TXT_DB}"
fi

if [ -z "$CONFIG_FILE" ]; then
  CONFIG_FILE="dual_embedding/bert-wwm-ext_official.json"
fi

CONFIG_DIR=/src/config
docker run --ipc=host --rm \
  --mount src="${WORK_DIR}",dst=/src,type=bind \
  --mount src="$TXT_DB",dst=/txt,type=bind \
  --mount src="$PRETRAIN_DIR",dst=/pretrain,type=bind,readonly \
  --mount src="$ANNOTATION_DIR",dst=/annotations,type=bind,readonly \
  --mount src="$CACHE_DIR",dst=/root/.cache,type=bind \
  -w /src visualjoyce/chengyubert:latest \
  bash -c "python preprocess.py --annotation $SOURCE_SPLIT --config=$CONFIG_DIR/$CONFIG_FILE"

echo "done"
