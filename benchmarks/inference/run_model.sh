set -x

model=$1
branch1=$2
branch2=$3
dtype=$4
graphs=$5
kernel=$6
gpus=$7

version=0
log_path=results/${model}_${dtype}_${graphs}_${kernel}_${gpus}gpus_v${version}
mkdir -p ${log_path}

params="--dtype $dtype "
if [[ "$graphs" == "true" ]]; then
    params+="--graphs "
fi
if [[ "$kernel" == "true" ]]; then
    params+="--kernel "
fi

echo "baseline $log_path"
deepspeed --num_gpus 1 gpt-bench.py -m "${model}" $params &> ${log_path}/baseline.log

cd ../../
git checkout ${branch1}
cd -
echo "ds ${branch1} $log_path"
deepspeed --num_gpus $gpus gpt-bench.py --deepspeed -m "${model}" $params &> ${log_path}/ds-${branch1}.log

cd ../../
git checkout ${branch2}
cd -
echo "ds ${branch2} $log_path"
deepspeed --num_gpus $gpus gpt-bench.py --deepspeed -m "${model}" $params&> ${log_path}/ds-${branch2}.log
