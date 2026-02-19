PROFILE_DIR=${PROFILE_DIR:-profiles}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
DC_OPTS="--compile --deepcompile"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"

MODEL="meta-llama/Llama-2-7b-chat-hf"
BATCH_SIZE_OPTS=(1)
SEQ_LENGTH=$1

for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    ARGS="--model ${MODEL} --batch-size ${BATCH_SIZE} ${ACC_OPTS} ${PROFILE_OPTS}"
    
    # compiled ulysses
    bash ./run_multinode.sh --backend inductor ${ARGS} ${DC_OPTS} --num-layers 1 --num-gpus 2 --seq-length ${SEQ_LENGTH}
    cp -r logs ${PROFILE_DIR}/
done
