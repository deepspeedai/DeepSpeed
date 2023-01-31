---
title: "Getting Started with DeepSpeed on Azure"
tags: getting-started
---

This tutorial will help you get started with DeepSpeed on Azure.

If you don't already have an Azure account please see more details here: [https://azure.microsoft.com/](https://azure.microsoft.com/).

# DeepSpeed on Azure via AzureML

The recommended and simplest method to try DeepSpeed on Azure is through [AzureML](https://azure.microsoft.com/en-us/services/machine-learning/). A training example and a DeepSpeed autotuning example using AzureML v2 can be found [here](https://github.com/Azure/azureml-examples/tree/main/cli/jobs/deepspeed).

For AzureML v1 examples, please take a look at easy-to-use examples for Megatron-DeepSpeed, Transformers and CIFAR training [here](https://github.com/Azure/azureml-examples/tree/main/v1/python-sdk/workflows/train/deepspeed).

> Our [Megatron-DeepSpeed](https://github.com/microsoft/megatron-deepspeed) contains the most up to date [recipe](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples/azureml) for end-to-end training on AzureML.

# DeepSpeed on Azure VMs

If you don't have access to AzureML or if want to build a custom environments using [Azure virtual machines](https://azure.microsoft.com/en-us/services/virtual-machines/) or Azure VM Scale-Sets ([VMSS](https://docs.microsoft.com/en-us/azure/virtual-machine-scale-sets/overview)), we are working on easy-to-use cluster setup scripts that will be published in the next few weeks.

If you already have a cluster setup, you can use the [azure recipes](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples/azure) that can easily be modified to train various model configurations.
