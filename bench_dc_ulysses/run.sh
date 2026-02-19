HOST_IP=$1
NUM_NODES=$2
NUM_PROCESSES=$3
BACKEND=$4
MODEL=$5
GRADIENT_ACCUMULATION_STEPS=$6
DEEPCOMPILE=$7
shift 7
EXTRA_OPTS="$@"

export NCCL_DEBUG=WARN

CONFIG_TEMPLATE=configs/ds_config.yaml.template

echo "HOST_IP: ${HOST_IP}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "BACKEND: ${BACKEND}"
echo "MODEL: ${MODEL}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "EXTRA_OPTS: ${EXTRA_OPTS}"

MACHINE_RANK=$(hostname | sed 's/[^0-9]*//g')

python generate_conf.py \
    --machine_rank ${MACHINE_RANK} \
    --num_machines ${NUM_NODES} \
    --num_processes ${NUM_PROCESSES} \
    --template_file ${CONFIG_TEMPLATE} \
    --output_file configs/config.yaml

GAS_OPTS="--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"
DEEPCOMPILE_OPTS=""
if [ "${DEEPCOMPILE}" == "1" ]; then
    DEEPCOMPILE_OPTS="--deepcompile"
fi

if [ "${BACKEND}" == "deepspeed" ]; then
    python generate_conf.py \
        --machine_rank ${MACHINE_RANK} \
        --num_machines ${NUM_NODES} \
        --num_processes ${NUM_PROCESSES} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        ${DEEPCOMPILE_OPTS} \
        --template_file configs/ds_config.json.template \
        --output_file configs/ds_config.json
fi

accelerate launch --main_process_ip ${HOST_IP} --main_process_port 12345 \
--num_machines ${NUM_NODES} --num_processes ${NUM_PROCESSES} --machine_rank ${MACHINE_RANK} \
--config_file configs/config.yaml \
run_acc_lm.py \
--model_name "${MODEL}" \
${GAS_OPTS} \
${EXTRA_OPTS} \
2>&1 | tee ${LOG_FILE}