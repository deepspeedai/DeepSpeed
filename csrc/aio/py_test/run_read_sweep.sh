#!/bin/bash
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <io_size> <output log dir> <target_gpu>"
    exit 1
fi

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
    validate_cmd="TORCH_EXTENSIONS_DIR=./torch_extentions python3 ./validate_async_io.py"
    eval ${validate_cmd}
    res=$?
    if [[ $res != 0 ]]; then
        echo "Failing because environment is not properly configured"
        echo "Possible fix: sudo apt-get install libaio-dev"
        exit 1
    fi
}

function fileExists() {
    local file="$1"
    if [[ -f "$file" ]]; then
        return 0
    else
        return 1
    fi
}

validate_environment

IO_SIZE=$1
LOG_DIR=./aio_perf_sweep
MAP_DIR=$2/aio
GPU_MEM=$3
USE_GDS=$4
RUN_SCRIPT=./test_ds_aio.py
READ_OPT="--read"

prep_folder ${MAP_DIR}
prep_folder ${LOG_DIR}

if [[ -d ${LOG_DIR} ]]; then
    rm -f ${LOG_DIR}/*
else
    mkdir -p ${LOG_DIR}
fi

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

DISABLE_CACHE="sudo sync; sudo bash -c 'echo 1 > /proc/sys/vm/drop_caches' "
SYNC="sudo sync"

for xtype in cpu gpu gds; do
    if [[ $xtype == "cpu" ]]; then
        gpu_opt=""
        gds_opt=""
    elif [[ $xtype == "gpu" ]]; then
        gpu_opt="--gpu"
        gds_opt=""
    else
        gpu_opt="--gpu"
        gds_opt="--use_gds"
    fi
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
                    for d in 8 16 32 64 128; do
                        for bs in 128K 256K 512K 1M 2M 4M 8M 16M; do
                            SCHED_OPTS="${sub_opt} ${ov_opt} --handle ${gpu_opt} ${gds_opt} --folder_to_device_mapping /mnt/nvme01:0"
                            OPTS="--queue_depth ${d} --block_size ${bs} --io_size ${IO_SIZE} --io_parallel ${t}"
                            LOG="${LOG_DIR}/read_${xtype}_${sub}_${ov}_t${t}_p${p}_d${d}_bs${bs}.txt"
                            cmd="/usr/bin/time python ${RUN_SCRIPT} ${READ_OPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"

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
done
