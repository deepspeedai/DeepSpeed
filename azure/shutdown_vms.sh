#!/bin/bash

azure_config=azure_config.json
if [ ! -f ${azure_config} ]; then
    echo "Cannot find $azure_config"
    exit 1
fi

delete=0
while getopts 'd' flag; do
  case "${flag}" in
    d) delete=1 ;;
    *)
        echo "Unexpected option ${flag}"
        exit 1
        ;;
  esac
done

num_vms=`cat ${azure_config} | jq .num_vms`
if [ $num_vms == "null" ]; then echo 'missing num_vms in config'; exit 1; fi
location=`cat ${azure_config} | jq .location | sed 's/"//g'`
if [ $location == "null" ]; then echo 'missing location in config'; exit 1; fi

base_vm_name=deepspeed
resource_group=deepspeed_rg_$location

for i in `seq 0 $(( num_vms - 1))`; do
    vm_name=${base_vm_name}_$i
    if [ $delete == 0 ]; then
        echo "deallocating $vm_name"
        az vm deallocate --resource-group $resource_group --name $vm_name --no-wait
    else
        echo "deleting $vm_name"
        az vm delete -y --resource-group $resource_group --name $vm_name --no-wait
    fi
done
