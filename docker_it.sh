# Licensed under the MIT license.
cat smusg.logo
WORK_DIR=$(readlink -f .)
DATA_DIR=${WORK_DIR}/data
SUB_PROJECT="$1"
MODEL_PARA="$2"
CONFIG_DIR=/src/config
TXT_DB=${DATA_DIR}/txt_db
OUTPUT=${DATA_DIR}/output
PRETRAIN_DIR=${DATA_DIR}/pretrained
ANNOTATION_DIR=${DATA_DIR}/annotations
CACHE_DIR=${DATA_DIR}/.cache

if [ -z "$MODEL_PARA" ]; then
  MODEL_PARA=""
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  CUDA_VISIBLE_DEVICES='all'
  N_GPU=1
else
  N_GPU=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c)
  N_GPU=$((N_GPU+1))
fi

HOROVOD_PARA=""
if [ $N_GPU -gt 1 ];
then
  HOROVOD_PARA="horovodrun -np $N_GPU  -H localhost:$N_GPU"
fi

echo "Training using ${N_GPU} GPUs: ${CUDA_VISIBLE_DEVICES}!"

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
  --mount src="$ANNOTATION_DIR",dst=/annotations,type=bind,readonly \
  --mount src="$TXT_DB",dst=/txt,type=bind \
  --mount src="$CACHE_DIR",dst=/root/.cache,type=bind \
  -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -w /src visualjoyce/chengyubert:latest \
  bash
