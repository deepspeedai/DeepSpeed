#!/bin/bash

NUM_NODES=${NUM_NODES:-$(wc -l < hostfile_n4)}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
NUM_PROCESSES=$((${NUM_NODES} * ${NGPUS_PER_NODE}))

BACKEND="deepspeed" # ignore
MODEL="meta-llama/Meta-Llama-3-8B"
COMPILE=0
PASSES="ALL"
EXTRA_OPTS=""

EAGER=0
DEEPCOMPILE=0
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION_CHECKPOINTING=1
BATCH_SIZE=1
SEQ_LENGTH=512

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --batch_size $2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            EXTRA_OPTS="${EXTRA_OPTS} --seq_length $2"
            shift 2
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            # EXTRA_OPTS="${EXTRA_OPTS} --gradient_accumulation_steps $2"
            shift 2
            ;;
        --activation-checkpointing)
            ACTIVATION_CHECKPOINTING=1
            EXTRA_OPTS="${EXTRA_OPTS} --activation_checkpointing"
            shift
            ;;   
        --compile)
            COMPILE=1
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --eager)
            EAGER=1
            EXTRA_OPTS="${EXTRA_OPTS} --backend eager"
            shift
            ;;
        --deepcompile)
            DEEPCOMPILE=1
            shift
            ;;
        --passes)
            PASSES="$2"
            EXTRA_OPTS="${EXTRA_OPTS} $1 $2"
            shift 2
            ;;
        --profile)
            EXTRA_OPTS="${EXTRA_OPTS} $1"
            shift
            ;;
        --profile-dir)
            EXTRA_OPTS="${EXTRA_OPTS} --profile_dir $2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-layers)
            EXTRA_OPTS="${EXTRA_OPTS} --num_layers $2"
            shift 2
            ;;
         --num-gpus)
            NGPUS_PER_NODE="$2"
            NUM_PROCESSES=$((${NUM_NODES} * ${NGPUS_PER_NODE}))
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


HOST_IP=$(hostname -i)

mkdir -p logs

SCRIPT_DIR=$(dirname $(realpath $0))

#replace , with _ in PASSES
PASSES=$(echo $PASSES | tr ',' '_')

LOG_FILE=debug_b${BACKEND}np${NUM_PROCESSES}c${COMPILE}dc${DEEPCOMPILE}bs${BATCH_SIZE}seq${SEQ_LENGTH}.log

if [ "${NUM_NODES}" == "1" ]; then
    # avoid dependency on pdsh when possible
    cd ${SCRIPT_DIR}; bash ./run.sh ${HOST_IP} ${NUM_NODES} ${NUM_PROCESSES} ${BACKEND} ${MODEL} ${GRADIENT_ACCUMULATION_STEPS} ${DEEPCOMPILE} ${EXTRA_OPTS} \
        2>&1 | tee logs/${LOG_FILE}
else
    ds_ssh -f hostfile_n${NUM_NODES} "cd ${SCRIPT_DIR}; bash ./run.sh ${HOST_IP} ${NUM_NODES} ${NUM_PROCESSES} ${BACKEND} ${MODEL} ${GRADIENT_ACCUMULATION_STEPS} ${SCHEDULE} ${OFFLOAD_OPT_STATES} ${EXTRA_OPTS}" \
        2>&1 | tee logs/${LOG_FILE}
fi
