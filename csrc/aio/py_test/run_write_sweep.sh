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

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <write size in MB> <write dir ><output log dir>"
    exit 1
fi

SIZE="$1M"
WRITE_DIR=$2
LOG_DIR=$3

OUTPUT_FILE=${WRITE_DIR}/ds_aio_write_${SIZE}B.pt
WRITE_OPT="--write_file ${OUTPUT_FILE} --write_size ${SIZE}"


prep_folder ${WRITE_DIR}
prep_folder ${LOG_DIR}

RUN_SCRIPT=./test_ds_aio.py

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
            for p in 1; do
                for d in 1 2 4 8 16 32; do
                    for bs in 128K 256K 512K 1M; do
                        SCHED_OPTS="${sub_opt} ${ov_opt} --handle --threads 1"
                        OPTS="--io_parallel ${p} --queue_depth ${d} --block_size ${bs}"
                        LOG="${LOG_DIR}/write_${SIZE}B_${sub}_${ov}_t${t}_p${p}_d${d}_bs${bs}.txt"
                        cmd="python ${RUN_SCRIPT} ${WRITE_OPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"
                        echo ${cmd}
                        eval ${cmd}
                        sleep 2
                    done
                done
        done
        done
    done
done
