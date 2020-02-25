#!/bin/bash

azure_config=azure_config.json
if [ ! -f ${azure_config} ]; then
    echo "Cannot find $azure_config"
    exit 1
fi
location=`cat ${azure_config} | jq .location | sed 's/"//g'`
rg=deepspeed_rg_$location

ssh_key=`cat ${azure_config} | jq .ssh_private_key | sed 's/"//g'`
if [ $ssh_key == "null" ]; then echo 'missing ssh_private_key in config'; exit 1; fi
docker_ssh_port=`cat ${azure_config} | jq .docker_ssh_port`
if [ $docker_ssh_port == "null" ]; then echo 'missing docker_ssh_port in config'; exit 1; fi

username=deepspeed
args="-i ${ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

num_vms=`az vm list  -g $rg | jq '. | length'`
first_ip_addr=`az vm list-ip-addresses  -g $rg | jq .[0].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
num_slots=`ssh $args ${username}@${first_ip_addr} 'nvidia-smi -L | wc -l'`
echo "number of slots per vm: $num_slots"

hostfile=hostfile
ssh_config=config
echo -n "" > $hostfile
echo -n "" > $ssh_config
for node_id in `seq 0 $((num_vms - 1))`; do
    private_ip_addr=`az vm list-ip-addresses  -g $rg | jq .[${node_id}].virtualMachine.network.privateIpAddresses[0] | sed 's/"//g'`
    echo "worker-${node_id} slots=${num_slots}" >> hostfile
    echo "Host worker-${node_id}
    HostName ${private_ip_addr}
    Port ${docker_ssh_port}
    StrictHostKeyChecking no
    " >> ${ssh_config}
done

update_script="
sudo mkdir -p /job;
sudo chmod -R 777 /job;
mkdir -p workdir;
git clone https://github.com/microsoft/DeepSpeed.git workdir/DeepSpeed;
"

for node_id in `seq 0 $((num_vms - 1))`; do
    ip_addr=`az vm list-ip-addresses  -g $rg | jq .[${node_id}].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
    addr=${username}@${ip_addr}
    echo "copying ssh keys, ssh config, hostfile to worker-${node_id}"
    ssh $args ${addr} $update_script
    scp $args ${ssh_key}* ${addr}:.ssh/
    scp $args ${ssh_config} ${addr}:.ssh/
    scp $args ${hostfile} ${addr}:/job/
done
rm $hostfile $ssh_config
