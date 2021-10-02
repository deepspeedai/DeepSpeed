#!/bin/bash
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input file> <output log dir>"
    exit 1
fi


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

INPUT_FILE=$1
if [[ ! -f ${INPUT_FILE} ]]; then
    echo "Input file not found: ${INPUT_FILE}"
    exit 1
fi

LOG_DIR=$2/aio_perf_sweep
RUN_SCRIPT=./test_ds_aio.py
READ_OPT="--read_file ${INPUT_FILE}"

if [[ -d ${LOG_DIR} ]]; then
    rm -f ${LOG_DIR}/*
else
    mkdir -p ${LOG_DIR}
fi

DISABLE_CACHE="sync; sudo bash -c 'echo 1 > /proc/sys/vm/drop_caches' "
SYNC="sync"

for sub in single block; do
    if [[ $sub == "single" ]]; then
        sub_opt="--single_submit"
    else
        sub_opt=""
    fi
    for ov in overlap sequential; do
        if [[ $ov == "overlap" ]]; then
            ov_opt="--overlap_events"
        else
            ov_opt=""
        fi
        for t in 1 2 4 8; do
            for p in 1 ; do
                for d in 1 2 4 8 16 32; do
                    for bs in 128K 256K 512K 1M; do
                        SCHED_OPTS="${sub_opt} ${ov_opt} --handle --threads ${t}"
                        OPTS="--io_parallel ${p} --queue_depth ${d} --block_size ${bs}"
                        LOG="${LOG_DIR}/read_${sub}_${ov}_t${t}_p${p}_d${d}_bs${bs}.txt"
                        cmd="python ${RUN_SCRIPT} ${READ_OPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"
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
