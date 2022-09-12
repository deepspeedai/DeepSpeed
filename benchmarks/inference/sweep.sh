set -ex

export TRANSFORMERS_CACHE=/tmp/hf-cache

branch1=$1
branch2=$2

for m in `echo "EleutherAI/gpt-neo-2.7B EleutherAI/gpt-neo-1.3B EleutherAI/gpt-neo-125M"`; do
  bash run_model.sh $m $branch1 $branch2
done

for m in `echo "gpt2 gpt2-large gpt2-xl"`; do
  bash run_model.sh $m $branch1 $branch2
done

for m in `echo "EleutherAI/gpt-j-6B"`; do
  bash run_model.sh $m $branch1 $branch2
done

for m in `echo "facebook/opt-125m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b"`; do
  bash run_model.sh $m $branch1 $branch2
done

for m in `echo "bigscience/bloom-560m bigscience/bloom-1b7 bigscience/bloom-3b bigscience/bloom-7b1"`; do
  bash run_model.sh $m $branch1 $branch2
done
