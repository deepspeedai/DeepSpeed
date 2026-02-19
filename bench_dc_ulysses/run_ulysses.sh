#!/bin/bash

SEQ_LEN=${1:-1024}
COMPILE=${2:-eager}
SP_SIZE=${3:-2}
DP_SIZE=${4:-1}
LAYER_COUNT=${5:-""}
EXP_NAME=${6:-""}

if [[ "$COMPILE" != "eager" && "$COMPILE" != "compile" && "$COMPILE" != "deepcompile" && "$COMPILE" != "ringattn" ]]; then
    echo "Invalid mode: $COMPILE. Choose from eager, compile, deepcompile, ringattn."
    exit 1
fi

HOST_IP=$(hostname -i | awk '{print $1}')
PORT=$(python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
NUM_NODES=1
NUM_PROCESSES=$((SP_SIZE * DP_SIZE))
MODEL="meta-llama/Llama-2-7b-chat-hf"
# MODEL="meta-llama/Llama-3.1-8B"
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="meta-llama/Llama-3.2-3B"
PROFILE_DIR=${PROFILE_DIR:-profiles}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile_dir ${PROFILE_DIR}"

COMPILE_OPTS="--compile ${COMPILE}"
CONFIG_FILE="configs/torchcompile_config.yaml"
if [ "${COMPILE}" == "deepcompile" ]; then
    CONFIG_FILE="configs/deepcompile_config.yaml"
fi


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/log_${COMPILE}_seq${SEQ_LEN}_${TIMESTAMP}.log

echo "HOST_IP: ${HOST_IP}"
echo "PORT: ${PORT}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "MODEL: ${MODEL}"
echo "COMPILE: ${COMPILE}"
echo "SEQ_LEN: ${SEQ_LEN}"
echo "LOG_FILE: ${LOG_FILE}"

EXTRA_OPTS="--seq_length=${SEQ_LEN} --experiment_folder=${EXP_NAME} --sp_size=${SP_SIZE} --dp_size=${DP_SIZE}"

# Only pass --num_layers if provided
NUM_LAYER_OPTS=""
if [[ -n "${LAYER_COUNT}" ]]; then
    NUM_LAYER_OPTS="--num_layers ${LAYER_COUNT}"
fi

(
accelerate launch --main_process_ip ${HOST_IP} --main_process_port ${PORT} \
--num_machines ${NUM_NODES} --num_processes ${NUM_PROCESSES} --machine_rank 0 \
--config_file ${CONFIG_FILE} \
run_acc_lm.py \
--model_name "${MODEL}" ${NUM_LAYER_OPTS} \
${PROFILE_OPTS} \
${EXTRA_OPTS} \
${COMPILE_OPTS}
) 2>&1 | tee ${LOG_FILE}


