#!/bin/bash

# Copyright 2020 The Microsoft DeepSpeed Team

command -v pdsh
if [ $? != 0 ]; then
    echo "Cannot find pdsh, please install via 'apt-get install -y pdsh'"
    exit 1
fi

hostfile=/job/hostfile

while getopts "h?f:" opt; do
  case "$opt" in
    h|\?)
      echo "-f <hostfile>: specify a hostfile, defaults to /job/hostfile"
      exit 0
      ;;
    f)
      hostfile=$OPTARG
      shift $((OPTIND-1))
      ;;
  esac
done

echo "hostfile=$hostfile"

if [ -f $hostfile ]; then
    hosts=`cat $hostfile | awk '{print $1}' | paste -sd "," -`
    export PDSH_RCMD_TYPE=ssh
    pdsh -w ${hosts} $@
else
    echo "Missing hostfile at ${hostfile}, unable to proceed"
fi
