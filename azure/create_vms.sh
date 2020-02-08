#!/bin/bash

azure_config=azure_config.json

# Make sure jq is installed
command -v jq
if [ $? != 0 ]; then
    echo "Missing dependency of jq, please 'apt-get install jq'"
    exit 1
fi

if [ ! -f ${azure_config} ]; then
    echo "Cannot find $azure_config"
    exit 1
fi
cat $azure_config

num_vms=`cat ${azure_config} | jq .num_vms`
if [ $num_vms == "null" ]; then echo 'missing num_vms in config'; exit 1; fi
location=`cat ${azure_config} | jq .location | sed 's/"//g'`
if [ $location == "null" ]; then echo 'missing location in config'; exit 1; fi
azure_sku=`cat ${azure_config} | jq .azure_sku | sed 's/"//g'`
if [ $azure_sku == "null" ]; then echo 'missing azure_sku in config'; exit 1; fi
ssh_private_key=`cat ${azure_config} | jq .ssh_private_key | sed 's/"//g'`
if [ $ssh_private_key == "null" ]; then echo 'missing ssh_private_key in config'; exit 1; fi
ssh_key=${ssh_private_key}.pub

if [ ! -f ${ssh_private_key} ]; then
    echo "Cannot find $ssh_private_key"
    exit 1
fi
if [ ! -f ${ssh_key} ]; then
    echo "Cannot find $ssh_key"
    exit 1
fi

resource_group=deepspeed_rg_$location
az group create --name ${resource_group} --location $location

base_vm_name=deepspeed
vm_image="nvidia:ngc_azure_17_11:ngc_gpu_cloud_19_11_3:19.11.3"

az vm image terms accept --urn ${vm_image}

for i in `seq 0 $(( num_vms - 1))`; do
    vm_name=${base_vm_name}_$i
    echo "creating $vm_name"
    az vm create \
      --resource-group ${resource_group} \
      --name ${vm_name} \
      --image ${vm_image} \
      --admin-username deepspeed \
      --size ${azure_sku} \
      --ssh-key-values ${ssh_key}
done
