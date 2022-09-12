set -ex

model=$1
branch1=$2
branch2=$3

version=0
log_path=results/${model}_v${version}
mkdir -p ${log_path}

echo "baseline $log_path"
deepspeed --num_gpus 1 gpt-bench.py -m "${model}" &> ${log_path}/baseline.log

cd ../../
git checkout ${branch1}
cd -
echo "ds ${branch1} $log_path"
deepspeed --num_gpus 1 gpt-bench.py --deepspeed -m "${model}" &> ${log_path}/ds-${branch1}.log

cd ../../
git checkout ${branch2}
cd -
echo "ds ${branch2} $log_path"
deepspeed --num_gpus 1 gpt-bench.py --deepspeed -m "${model}" &> ${log_path}/ds-${branch2}.log
