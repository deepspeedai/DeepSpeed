#!/usr/bin/env bash

set -e
err_report() {
    echo "Error on line $1"
    echo "Fail to install deepspeed"
}
trap 'err_report $LINENO' ERR

usage() {
  echo """
Usage: install.sh [options...]

By default will install deepspeed and all third party dependencies across all machines listed in
hostfile (hostfile: /job/hostfile). If no hostfile exists, will only install locally

[optional]
    -l, --local_only        Install only on local machine
    -s, --pip_sudo          Run pip install with sudo (default: no sudo)
    -r, --allow_sudo        Allow script to be run by root (probably don't want this, instead use --pip_sudo)
    -n, --no_clean          Do not clean prior build state, by default prior build files are removed before building wheels
    -m, --pip_mirror        Use the specified pip mirror (default: the default pip mirror)
    -H, --hostfile          Path to MPI-style hostfile (default: /job/hostfile)
    -e, --examples          Checkout deepspeed example submodule (no install)
    -v, --verbose           Verbose logging
    -h, --help              This help text
  """
}

ds_only=0
local_only=0
pip_sudo=0
entire_dlts_job=1
hostfile=/job/hostfile
pip_mirror=""
skip_requirements=0
allow_sudo=0
no_clean=0
verbose=0
examples=0

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -l|--local_only)
    local_only=1;
    shift
    ;;
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
        echo "User-provided hostfile does not exist at $hostfile, exiting"
        exit 1
    fi
    shift
    shift
    ;;
    -e|--examples)
    examples=1
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)
    echo "Unknown argument(s)"
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

if [ "$examples" == "1" ]; then
    git submodule update --init --recursive
    exit 0
fi

if [ "$verbose" == "1" ]; then
    VERBOSE="-v"
    PIP_VERBOSE=""
else
    VERBOSE=""
    PIP_VERBOSE="--disable-pip-version-check"
fi

rm_if_exist() {
    echo "Attempting to remove $1"
    if [ -f $1 ]; then
        rm $VERBOSE $1
    elif [ -d $1 ]; then
        rm -rf $VERBOSE $1
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
    PIP_INSTALL="pip install $VERBOSE $PIP_VERBOSE -i $pip_mirror"
else
    PIP_INSTALL="pip install $VERBOSE $PIP_VERBOSE"
fi


if [ ! -f $hostfile ]; then
    echo "No hostfile exists at $hostfile, installing locally"
    local_only=1
fi

echo "Building deepspeed wheel"
python setup.py $VERBOSE bdist_wheel

if [ "$local_only" == "1" ]; then
    echo "Installing deepspeed"
#    $PIP_SUDO pip uninstall -y deepspeed
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

    pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm $tmp_wheel_path/*; else mkdir -pv $tmp_wheel_path; fi"
    pdcp -w $hosts requirements/requirements.txt ${tmp_wheel_path}/

    echo "Installing deepspeed"
    pdsh -w $hosts "$PIP_SUDO pip uninstall -y deepspeed"
    pdcp -w $hosts dist/deepspeed*.whl $tmp_wheel_path/
    pdsh -w $hosts "$PIP_SUDO $PIP_INSTALL $tmp_wheel_path/deepspeed*.whl"
    pdsh -w $hosts "ds_report"
    pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm $tmp_wheel_path/*.whl; rm $tmp_wheel_path/*.txt; rmdir $tmp_wheel_path; fi"
fi
