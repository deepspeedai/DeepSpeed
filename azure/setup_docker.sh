#!/bin/bash

azure_config=azure_config.json
if [ ! -f ${azure_config} ]; then
    echo "Cannot find $azure_config"
    exit 1
fi

ssh_key=`cat ${azure_config} | jq .ssh_private_key | sed 's/"//g'`
if [ $ssh_key == "null" ]; then echo 'missing ssh_private_key in config'; exit 1; fi
num_vms=`cat ${azure_config} | jq .num_vms`
if [ $num_vms == "null" ]; then echo 'missing num_vms in config'; exit 1; fi

args="-i ${ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
username=deepspeed
for node_id in `seq 0 $((num_vms - 1))`; do
    ip_addr=`az vm list-ip-addresses | jq .[${node_id}].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
    addr=${username}@${ip_addr}
    ssh ${args} $addr "docker pull deepspeed/deepspeed:latest"
    ssh ${args} $addr "cd workdir/DeepSpeed; git pull; git submodule update --init --recursive; bash azure/start_container.sh"
done
