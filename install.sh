#!/bin/bash

set -e
err_report() {
    echo "Error on line $1"
    echo "Fail to install deepspeed"
}
trap 'err_report $LINENO' ERR

usage() {
  echo """
Usage: install.sh [options...]

By default will install deepspeed and all third party dependecies accross all machines listed in
hostfile (hostfile: /job/hostfile). If no hostfile exists, will only install locally

[optional]
    -d, --deepspeed_only    Install only deepspeed and no third party dependencies
    -t, --third_party_only  Install only third party dependencies and not deepspeed
    -l, --local_only        Installs only on local machine
    -H, --hostfile          Path to MPI-style hostfile (default: /job/hostfile)
    -h, --help              This help text
  """
}

ds_only=0
tp_only=0
deepspeed_install=1
third_party_install=1
local_only=0
entire_dlts_job=1
hostfile=/job/hostfile

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -d|--deepspeed_only)
    deepspeed_install=1;
    third_party_install=0;
    ds_only=1;
    shift
    ;;
    -t|--third_party_only)
    deepspeed_install=0;
    third_party_install=1;
    tp_only=1;
    shift
    ;;
    -l|--local_only)
    local_only=1;
    shift
    ;;
    -H|--hostfile)
    hostfile=$2
    if [ ! -f $2 ]; then
        echo "User provided hostfile does not exist at $hostfile, exiting"
        exit 1
    fi
    shift
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)
    echo "Unkown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done

if [ "$ds_only" == "1" ] && [ "$tp_only" == "1" ]; then
    echo "-d and -t are mutually exclusive, only choose one or none"
    usage
    exit 1
fi

echo "Updating git hash/branch info"
echo "git_hash = '$(git rev-parse --short HEAD)'" > deepspeed/version_info.py
echo "git_branch = '$(git rev-parse --abbrev-ref HEAD)'" >> deepspeed/version_info.py
cat deepspeed/version_info.py

install_apex='sudo -H pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" third_party/apex'

if [ ! -f $hostfile ]; then
        echo "No hostfile exists at $hostfile, installing locally"
        local_only=1
fi

# Ensure dependencies are installed locally
sudo -H pip install -r requirements.txt

# Build wheels
if [ "$third_party_install" == "1" ]; then
    echo "Checking out sub-module(s)"
    git submodule update --init --recursive

    echo "Building apex wheel"
    cd third_party/apex
    python setup.py --cpp_ext --cuda_ext bdist_wheel
    cd -
fi
if [ "$deepspeed_install" == "1" ]; then
    echo "Installing deepspeed"
    python setup.py bdist_wheel
fi


if [ "$local_only" == "1" ]; then
    if [ "$third_party_install" == "1" ]; then
        echo "Installing apex"
        sudo -H pip uninstall -y apex
        sudo -H pip install third_party/apex/dist/apex*.whl
    fi
    if [ "$deepspeed_install" == "1" ]; then
        echo "Installing deepspeed"
        sudo -H pip uninstall -y deepspeed
        sudo -H pip install dist/deepspeed*.whl
        python -c 'import deepspeed; print("deepspeed info:", deepspeed.__version__, deepspeed.__git_branch__, deepspeed.__git_hash__)'
        echo "Installation is successful"
    fi
else
    local_path=`pwd`
    if [ -f $hostfile ]; then
        hosts=`cat $hostfile | awk '{print $1}' | paste -sd "," -`;
    else
        echo "hostfile not found, cannot proceed"
        exit 1
    fi
    export PDSH_RCMD_TYPE=ssh;
    tmp_wheel_path="/tmp/deepspeed_wheels"

    pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm $tmp_wheel_path/*.whl; else mkdir -pv $tmp_wheel_path; fi"
    pdcp -w $hosts requirements.txt ${tmp_wheel_path}/
    pdsh -w $hosts "sudo -H pip install -r ${tmp_wheel_path}/requirements.txt"
    if [ "$third_party_install" == "1" ]; then
        pdsh -w $hosts "sudo -H pip uninstall -y apex"
        pdcp -w $hosts third_party/apex/dist/apex*.whl $tmp_wheel_path/
        pdsh -w $hosts "sudo -H pip install $tmp_wheel_path/apex*.whl"
        pdsh -w $hosts 'python -c "import apex"'
    fi
    if [ "$deepspeed_install" == "1" ]; then
        echo "Installing deepspeed"
        pdsh -w $hosts "sudo -H pip uninstall -y deepspeed"
        pdcp -w $hosts dist/deepspeed*.whl $tmp_wheel_path/
        pdsh -w $hosts "sudo -H pip install $tmp_wheel_path/deepspeed*.whl"
        pdsh -w $hosts "python -c 'import deepspeed; print(\"deepspeed info:\", deepspeed.__version__, deepspeed.__git_branch__, deepspeed.__git_hash__)'"
        echo "Installation is successful"
    fi
    pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm $tmp_wheel_path/*.whl $tmp_wheel_path/requirements.txt; rmdir $tmp_wheel_path; fi"
fi
