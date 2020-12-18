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
    -l, --local_only        Install only on local machine
    -s, --pip_sudo          Run pip install with sudo (default: no sudo)
    -r, --allow_sudo        Allow script to be run by root (probably don't want this, instead use --pip_sudo)
    -n, --no_clean          Do not clean prior build state, by default prior build files are removed before building wheels
    -m, --pip_mirror        Use the specified pip mirror (default: the default pip mirror)
    -H, --hostfile          Path to MPI-style hostfile (default: /job/hostfile)
    -v, --verbose           Verbose logging
    -h, --help              This help text
  """
}

ds_only=0
tp_only=0
deepspeed_install=1
third_party_install=1
local_only=0
pip_sudo=0
entire_dlts_job=1
hostfile=/job/hostfile
pip_mirror=""
apex_commit=""
skip_requirements=0
allow_sudo=0
no_clean=0
verbose=0

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -s|--pip_sudo)
    pip_sudo=1;
    shift
    ;;
    -m|--pip_mirror)
    pip_mirror=$2;
    shift
    shift
    ;;
    -v|--verbose)
    verbose=1;
    shift
    ;;
    -r|--allow_sudo)
    allow_sudo=1;
    shift
    ;;
    -n|--no_clean)
    no_clean=1;
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

user=`whoami`
if [ "$allow_sudo" == "0" ]; then
    if [ "$user" == "root" ]; then
        echo "WARNING: running as root, if you want to install DeepSpeed with sudo please use -s/--pip_sudo instead"
        usage
        exit 1
    fi
fi

if [ "$ds_only" == "1" ] && [ "$tp_only" == "1" ]; then
    echo "-d and -t are mutually exclusive, only choose one or none"
    usage
    exit 1
fi

if [ "$verbose" == "1" ]; then
    VERBOSE="-v"
else
    VERBOSE=""
fi

rm_if_exist() {
    echo "Attempting to remove $1"
    if [ -f $1 ]; then
        rm $VERBOSE $1
    elif [ -d $1 ]; then
        rm -r $VERBOSE $1
    fi
}

if [ "$no_clean" == "0" ]; then
    # remove deepspeed build files
    rm_if_exist deepspeed/git_version_info_installed.py
    rm_if_exist dist
    rm_if_exist build
    rm_if_exist deepspeed.egg-info
fi

if [ "$pip_sudo" == "1" ]; then
    PIP_SUDO="sudo -H"
else
    PIP_SUDO=""
fi

if [ "$pip_mirror" != "" ]; then
    PIP_INSTALL="pip install $VERBOSE -i $pip_mirror"
else
    PIP_INSTALL="pip install $VERBOSE"
fi


if [ ! -f $hostfile ]; then
    echo "No hostfile exists at $hostfile, installing locally"
    local_only=1
fi

echo "Building deepspeed wheel"
python setup.py $VERBOSE bdist_wheel

if [ "$local_only" == "1" ]; then
    echo "Installing deepspeed"
    $PIP_SUDO pip uninstall -y deepspeed
    $PIP_SUDO $PIP_INSTALL dist/deepspeed*.whl
    ds_report
else
    local_path=`pwd`
    if [ -f $hostfile ]; then
        hosts=`cat $hostfile | awk '{print $1}' | paste -sd "," -`;
    else
        echo "hostfile not found, cannot proceed"
        exit 1
    fi
    export PDSH_RCMD_TYPE=ssh
    tmp_wheel_path="/tmp/deepspeed_wheels"

    pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm $tmp_wheel_path/*.whl; else mkdir -pv $tmp_wheel_path; fi"
    pdcp -w $hosts requirements/requirements.txt ${tmp_wheel_path}/

    echo "Installing deepspeed"
    pdsh -w $hosts "$PIP_SUDO pip uninstall -y deepspeed"
    pdcp -w $hosts dist/deepspeed*.whl $tmp_wheel_path/
    pdsh -w $hosts "$PIP_SUDO $PIP_INSTALL $tmp_wheel_path/deepspeed*.whl"
    pdsh -w $hosts "ds_report"
    pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm $tmp_wheel_path/*.whl; rm $tmp_wheel_path/*.txt; rmdir $tmp_wheel_path; fi"
fi
