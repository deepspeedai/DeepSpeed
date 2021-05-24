#!/bin/bash

usage() {
  echo """
Usage: $0 [defined arguments...] [other arguments...]

[defined]
	-g, --num_gpus          num gpus per node to use
	-h, --help              this help text
	-n, --num_nodes         num nodes to use
	-e, --epochs		        num of training epochs
	-b, --batch_size 	      training batch size
  -p, --port              master port for nccl

[other arguments]
	all undefined arguments will be passed to the user's application
  """
}

validate_folder() {
	dir=$1
	dir_name=$2

	if [[ -d ${dir} ]]; then
		echo "Using ${dir_name}: ${dir}"
	else
		echo "${dir} folder not found"
		exit 1
	fi
}

remove_folder() {
	dir=$1
	dir_name=$2

	if [[ -d ${dir} ]]; then
		echo "The variable ${dir_name} is set to ${dir} which already exists, so removing and creating a fresh one"
    rm -rvf ${dir}
	fi
}

num_nodes=1
num_gpus=8
epochs=2
batch_size=24
enable_deepspeed=false
master_port=$((20000+RANDOM%5000))
LR=3e-5

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -g|--num_gpus)
    num_gpus="$2"
    shift
    shift
    ;;
    -n|--num_nodes)
    num_nodes="$2"
    shift
    shift
    ;;
    -e|--epochs)
    epochs="$2"
    shift
    shift
    ;;
    -b|--batch_size)
    batch_size="$2"
    shift
    shift
    ;;
    -p|--master_port)
    master_port="$2"
    shift
    shift
    ;;
    -d|--deepspeed)
    enable_deepspeed=true
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *) # other arguments
    other_args="${other_args} $1"
    shift
    ;;
esac
done

# Validate path to BingBertSquad script
if [ -z "${BingBertSquad_DIR+x}" ]; then
  export BingBertSquad_DIR=../../../../DeepSpeedExamples/BingBertSquad
  echo "BingBertSquad_DIR environment variable not set; trying default: ${BingBertSquad_DIR}"
fi
validate_folder ${BingBertSquad_DIR} "BingBertSquad_DIR"

# Validate path to processed Squad data
if [ -z "${SQUAD_DIR+x}" ]; then
  export SQUAD_DIR=/data/BingBertSquad
  echo "SQUAD_DIR environment variable not set; trying default: ${SQUAD_DIR}"
fi
validate_folder ${SQUAD_DIR} "SQUAD_DIR"

# Set output path
if [ -z "${OUTPUT_DIR+x}" ]; then
  export OUTPUT_DIR=/tmp/BingBertSquad-Output
  echo "OUTPUT_DIR environment variable not set; trying default: ${OUTPUT_DIR}"
fi
remove_folder ${OUTPUT_DIR} "OUTPUT_DIR"

echo "num_nodes: ${num_nodes}"
echo "num_gpus: ${num_gpus}"
echo "epochs: ${epochs}"
echo "batch_size: ${batch_size}"
echo "master_port: ${master_port}"
echo "deepspeed: ${enable_deepspeed}"
echo "other_args: ${other_args}"

EFFECTIVE_BATCH_SIZE=${batch_size}
MAX_GPU_BATCH_SIZE=3
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/num_gpus))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi

if [[ ${enable_deepspeed} == true ]]; then
  BingBertSquad_script=${BingBertSquad_DIR}/nvidia_run_squad_deepspeed.py
else
  BingBertSquad_script=${BingBertSquad_DIR}/nvidia_run_squad_baseline.py
fi

JOB_NAME="BingBertSquad_ds-${enable_deepspeed}_${num_gpus}-gpu"

squad_args="--bert_model bert-large-uncased \
            --do_train \
            --do_lower_case \
            --train_file ${SQUAD_DIR}/train-v1.1.json \
            --predict_file ${SQUAD_DIR}/dev-v1.1.json \
            --train_batch_size ${PER_GPU_BATCH_SIZE} \
            --learning_rate ${LR} \
            --num_train_epochs ${epochs} \
            --max_seq_length 384 \
            --doc_stride 128 \
            --do_predict \
            --output_dir ${OUTPUT_DIR} \
            --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
		      	--job_name ${JOB_NAME} \
            --model_file ${SQUAD_DIR}/training_state_checkpoint_162.tar
            "

run_cmd="deepspeed.pt \
      --num_nodes ${num_nodes} \
      --num_gpus ${num_gpus} \
      --master_port ${master_port}
      ${BingBertSquad_script} ${other_args} ${squad_args}"

echo ${run_cmd}
eval ${run_cmd}

set +x

#python ${BingBertSquad_DIR}/evaluate-v1.1.py ${SQUAD_DIR}/dev-v1.1.json ${OUTPUT_DIR}/predictions.json > ${OUTPUT_DIR}/CorrectnessScores.txt
