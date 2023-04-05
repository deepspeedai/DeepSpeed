#! /bin/bash

helpFunction()
{
    echo ""
    echo "Usage: $0 -m model-parallelism -g gpu-per-node -n node# -b batch-size -s steps -l layers -h hidden_size -q seq_length -e heads -c ckpt_num_layers -p [-d]"
    echo -e "\t-m model parallelism"
    echo -e "\t-g gpus per node"
    echo -e "\t-n node count"
    echo -e "\t-b batch size"
    echo -e "\t-s training steps"
    echo -e "\t-l layers"
    echo -e "\t-h hidden size"
    echo -e "\t-q sequence length"
    echo -e "\t-e attention heads"
    echo -e "\t-c checkpoint num_layers"
    echo -e "\t-o other args"
    echo -e "\t-d DeepSpeed config json file"
    echo -e "\t-z Enable Zero optimization"
    echo -e "\t-p DeepSpeed master port"
    exit 1
}

layers=2
hidden_size=128
seq_length=1024
ckpt_num_layers=1
other_args=""
ds_opt=""
zero_opt=""
master_port=29600

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

while getopts "m:g:n:b:s:l:h:q:e:c:o:d:z" opt
do
    case "$opt" in
        m ) mp="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        n ) nodes="$OPTARG" ;;
        b ) bs="$OPTARG" ;;
        s ) steps="$OPTARG" ;;
        l ) layers="$OPTARG" ;;
        h ) hidden_size="$OPTARG" ;;
        q ) seq_length="$OPTARG" ;;
        e ) heads="$OPTARG" ;;
        c ) ckpt_num_layers="$OPTARG" ;;
        p ) master_port="$OPTARG" ;;
        o ) other_args="$OPTARG" ;;
        d ) ds_opt="--deepspeed --deepspeed_config $script_dir/$OPTARG" ;;
        z ) zero_opt="--zero_optimization" ;;
        ? ) helpFunction ;;
    esac
done

# Print helpFunction in case parameters are empty
if [ -z "$mp" ] || [ -z "$gpus" ] || [ -z "$nodes" ] || [ -z "$bs" ] || [ -z "$steps" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000

gpt_options=" \
       --model-parallel-size ${mp} \
       --num-layers ${layers} \
       --hidden-size ${hidden_size} \
       --num-attention-heads ${heads} \
       --batch-size ${bs} \
       --seq-length ${seq_length} \
       --max-position-embeddings ${seq_length} \
       --train-iters ${steps} \
       --train-data webtext \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --checkpoint-num-layers ${ckpt_num_layers} \
       --fp16 \
       --cache-dir /tmp/cache_dir \
       --log-interval 1 \
       ${other_args} \
       ${ds_opt} \
       ${zero_opt} \
"

work_dir="../../../DeepSpeedExamples/Megatron-LM/"
run_cmd="(cd ${work_dir} && deepspeed --master_port ${master_port} --num_nodes $nodes --num_gpus $gpus pretrain_gpt2.py ${gpt_options})"
echo ${run_cmd}
eval ${run_cmd}

set +x
