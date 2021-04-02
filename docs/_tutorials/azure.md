---
title: "Getting Started with DeepSpeed on Azure"
---

This tutorial will help you get started running DeepSpeed on [Azure virtual
machines](https://azure.microsoft.com/en-us/services/virtual-machines/).
Looking forward, we will be integrating these techniques and additional enhancements
into the [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/) platform to
benefit all your large model training jobs.

If you don't already have an Azure account please see more details here: [https://azure.microsoft.com/](https://azure.microsoft.com/).

To use DeepSpeed on [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/), please take a look at easy-to-use examples for Transformers and CIFAR training from [AzureML Examples GitHub](https://github.com/Azure/azureml-examples/tree/main/workflows/train/deepspeed).

To help with launching Azure instances we suggest using the [Azure
CLI](https://docs.microsoft.com/en-us/cli/azure/?view=azure-cli-latest). We have created
several helper scripts to get you quickly started using DeepSpeed with Azure.
 * Install Azure CLI on your local box: [https://docs.microsoft.com/en-us/cli/azure/install-azure-cli](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).
 * Alternatively, you can use the Azure in-browser shell: [https://shell.azure.com/](https://shell.azure.com/).

## Create an SSH key
Generate an SSH key that will be used across this tutorial to SSH into your VMs and
between Docker containers. `ssh-keygen` is the recommended way of doing this. Our scripts
assume your key is located inside the same directory as the Azure scripts.

## Azure Config JSON
Our helper scripts depend on the following a configuration JSON for deployment
and setup.  We have provided a simple example JSON in `azure_config.json` that
sets up a basic environment with two VMs. This config uses the NV6_Promo
instance type which has one NVIDIA Tesla M60 GPU per VM. You can read more
details about the VM on the [Linux Virtual Machines
Pricing](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)
page.

See the example below:
 ```json
{
  "num_vms": 2,
  "location": "southcentralus",
  "azure_sku": "Standard_NV6_Promo",
  "ssh_private_key": "id_rsa",
  "docker_ssh_port": 2222
}
```

## Dependencies
The scripts in this tutorial require [jq](https://stedolan.github.io/jq/) to help with
parsing JSON from the command line. Also it is recommended to install
[pdsh](https://linux.die.net/man/1/pdsh) to help launch ssh connections in parallel.

## Create Azure VMs
We first need to allocate the VMs. We provide a script
```bash
./create_vms.sh
```
to create VMs with the Azure SKU in the region specified in `azure_config.json`. Feel
free to customize your JSON to your desired region/SKU. This step will take a few minutes
to complete while it sets up all of your VMs on Azure.

## Setup VM environment to use DeepSpeed
Next, we need to configure the VM environment for DeepSpeed. We provide a script
```bash
./setup_vms.sh
```
to generate a [hostfile](/getting-started/#resource-configuration-multi-node) and SSH
configuration on all of the VMs. This configuration will be used by the DeepSpeed
Docker containers in the next step.

## Start the DeepSpeed docker container
We now setup the DeepSpeed Docker containers on the VMs. We provide a script
```bash
./setup_docker.sh
```
to pull the DeepSpeed image onto all VMs and start a container instance in the
background. This will take several minutes since it needs to pull the entire Docker
image.

## Access VMs
The tool `azure_ssh.sh` will let you SSH into any of the VMs with this
syntax:
```bash
./azure_ssh.sh <node-id> [command]
```
where the `node-id` is a number between `0` and `num_vms-1`.  This script will find the
public IP address of your VM and use the SSH key provided in the Azure configuration
JSON.

## Access DeepSpeed container
Everything should be up and running at this point. Let's access the running DeepSpeed
container on the first VM and make sure we can talk to the other containers in our deployment.

 * SSH into the first VM via: `./azure_ssh.sh 0`
 * Change directories into the azure folder of this repo via: `cd ~/workdir/DeepSpeed/azure`
 * Attach the running docker container via: `./attach.sh`
 * You should now be able to `ssh` into any other docker container, the containers can be
   accessed via their SSH alias of `worker-N`, where `N` is the VM number between `0`
   and `num_vms-1`. In this example we should be able to successfully run `ssh worker-1
   hostname` which will return the hostname of worker-1.

## Parallel SSH across containers
 DeepSpeed comes installed with a helper script `ds_ssh` which is a wrapper around
 the [pdsh](https://linux.die.net/man/1/pdsh) command that lets you issue commands
 to groups of hosts (via SSH) in parallel. This wrapper simply connects with the
 hostfile that defines all the containers in your deployment. For example if you run
 `ds_ssh hostname` you should see a list of all the hostnames in your deployment.

## Run CIFAR-10 example model
We will now run the DeepSpeed CIFAR-10 model example to test the VM setup. From inside
the first DeepSpeed container:

  1) Install the python dependencies necessary to run the CIFAR-10 example model. You can
  do this across your cluster via:
  ```bash
  ds_ssh pip install -r ~/workdir/DeepSpeed/DeepSpeedExamples/cifar/requirements.txt
  ```

  2) Now change directories to the CIFAR example:
  ```bash
  cd ~/workdir/DeepSpeed/DeepSpeedExamples/cifar
  ```

  3) Finally, launch training across all VMs:
  ```bash
  deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json
  ```

## Megatron-LM GPT2
DeepSpeed includes an example model using Megatron-LM's GPT2. Please refer to the full
[Megatron tutorial](/tutorials/megatron/) for more details.
 * In order to fully train GPT2 with DeepSpeed and ZeRO we recommend using 8 instances of
   Azure's Standard_ND40rs_v2 SKU for a total of 64 NVIDIA V100 GPUs. With this setup and
   a batch size of 1536 you should be able to complete 100k training steps (153.6 million
   samples) in less than 2 weeks of training.
