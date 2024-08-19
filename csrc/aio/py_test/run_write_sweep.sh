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

IO_SIZE=$1
LOG_DIR=$2/aio_perf_sweep
MAP_DIR=$2/aio
GPU_MEM=$3
USE_GDS=$4
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

for sub in single block; do
    if [[ $sub == "single" ]]; then
        sub_opt="--single_submit"
    else
        sub_opt=""
    fi
    for ov in overlap sequential; do
        if [[ $ov == "sequential" ]]; then
            ov_opt="--sequential_requests"
        else
            ov_opt=""
        fi
        for p in 1 2 4 8; do
            for t in 1 2 4 8; do
                for d in 32 64 128; do
                    for bs in 256K 512K 1M; do
                        SCHED_OPTS="${sub_opt} ${ov_opt} --handle ${gpu_opt} ${gds_opt} --folder ${MAP_DIR}"
                        OPTS="--queue_depth ${d} --block_size ${bs} --io_size ${IO_SIZE} --multi_process ${p} --io_parallel ${t}"
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
