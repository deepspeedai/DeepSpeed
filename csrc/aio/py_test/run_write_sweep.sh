#!/bin/bash
function prep_folder()
{
    folder=$1
    if [[ -d ${folder} ]]; then
        rm -f ${folder}/*
    else
        mkdir -p ${folder}
    fi
}

function validate_environment()
{
    validate_cmd="python ./validate_async_io.py"
    eval ${validate_cmd}
    res=$?
    if [[ $res != 0 ]]; then
        echo "Failing because environment is not properly configured"
        echo "Possible fix: sudo apt-get install libaio-dev"
        exit 1
    fi
}



validate_environment

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <write size in [K,M,G]> <write dir ><output log dir>"
    exit 1
fi

SIZE=$1
WRITE_DIR=$2
LOG_DIR=$3/aio_perf_sweep

WRITE_OPT="--folder ${WRITE_DIR} --io_size ${SIZE} --loops 3"
IO_ENGINE="torch_fastio"
ENGINE_OPTS=""
if [[ $IO_ENGINE == "aio_handle" ]]; then
    IO_PARALLEL="1" # "1 2 4 8"
    QUEUE_DEPTH="8 16 32 64 128"
    BLOCK_SIZE="128K 256K 512K 1M 2M 4M 8M 16M"
    SUBMIT="block"
    OVERLAP="overlap"
elif [[ $IO_ENGINE == "torch_fastio" ]]; then
    IO_PARALLEL="1" # "1 2 4 8"
    QUEUE_DEPTH="8 16 32 64 128"
    BLOCK_SIZE="128K 256K 512K 1M 2M 4M 8M 16M"
    SUBMIT="block"
    OVERLAP="overlap"
    ENGINE_OPTS="--torch_legacy --fast_io_size ${SIZE}"
else
    IO_PARALLEL="1"
    QUEUE_DEPTH="8"
    BLOCK_SIZE="128K"
    SUBMIT="single"
    OVERLAP="sequential"
fi

prep_folder ${WRITE_DIR}
prep_folder ${LOG_DIR}

RUN_SCRIPT=./test_ds_aio.py

OUTPUT_FILE=${MAP_DIR}/ds_aio_write_${SIZE}B.pt
WRITE_OPT=""


prep_folder ${MAP_DIR}
prep_folder ${LOG_DIR}


if [[ ${GPU_MEM} == "gpu" ]]; then
    gpu_opt="--gpu"
else
    gpu_opt=""
fi
if [[ ${USE_GDS} == "gds" ]]; then
    gds_opt="--use_gds"
else
    gds_opt=""
fi

DISABLE_CACHE="sync; bash -c 'echo 1 > /proc/sys/vm/drop_caches' "
SYNC="sync"

for sub in  ${SUBMIT}; do
    if [[ $sub == "single" ]]; then
        sub_opt="--single_submit"
    else
        sub_opt=""
    fi
    for ov in ${OVERLAP}; do
        if [[ $ov == "sequential" ]]; then
            ov_opt="--sequential_requests"
        else
            ov_opt=""
        fi
        for p in 1;  do
            for t in ${IO_PARALLEL}; do
                for d in ${QUEUE_DEPTH}; do
                    for bs in ${BLOCK_SIZE}; do
                        SCHED_OPTS="${sub_opt} ${ov_opt} --engine ${IO_ENGINE} --io_parallel ${t} ${ENGINE_OPTS}"
                        OPTS="--multi_process ${p} --queue_depth ${d} --block_size ${bs}"
                        LOG="${LOG_DIR}/write_${sub}_${ov}_t${t}_p${p}_d${d}_bs${bs}.txt"
                        cmd="python ${RUN_SCRIPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"
                        echo ${DISABLE_CACHE}
                        echo ${cmd}
                        echo ${SYNC}

                        eval ${DISABLE_CACHE}
                        eval ${cmd}
                        eval ${SYNC}
                        sleep 2
                    done
                done
        done
        done
    done
done
