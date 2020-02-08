#!/bin/bash

name=${1-deepspeed}
image=deepspeed/deepspeed:latest
echo "starting docker image named $name"
docker run -d -t --name $name \
        --network host \
        -v ${HOME}/workdir:/home/deepspeed/workdir \
        -v ${HOME}/.ssh:/home/deepspeed/.ssh \
        -v /job/hostfile:/job/hostfile \
        --gpus all $image bash -c 'sudo service ssh start && sleep infinity'
