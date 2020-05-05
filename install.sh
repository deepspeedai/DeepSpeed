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
    -l, --local_only        Install only on local machine
    -s, --pip_sudo          Run pip with sudo (default: no sudo)
    -m, --pip_mirror        Use the specified pip mirror (default: the default pip mirror)
    -H, --hostfile          Path to MPI-style hostfile (default: /job/hostfile)
    -a, --apex_commit       Install a specific commit hash of apex, instead of the one deepspeed points to
    -k, --skip_requirements Skip installing DeepSpeed requirements
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
    -s|--pip_sudo)
    pip_sudo=1;
    shift
    ;;
    -m|--pip_mirror)
    pip_mirror=$2;
    shift
    shift
    ;;
    -a|--apex_commit)
    apex_commit=$2;
    shift
    shift
    ;;
    -k|--skip_requirements)
    skip_requirements=1;
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
echo "git_hash = '$(git rev-parse --short HEAD)'" > deepspeed/git_version_info.py
echo "git_branch = '$(git rev-parse --abbrev-ref HEAD)'" >> deepspeed/git_version_info.py
cat deepspeed/git_version_info.py

if [ "$pip_sudo" == "1" ]; then
  PIP_SUDO="sudo -H"
else
  PIP_SUDO=""
fi

if [ "$pip_mirror" != "" ]; then
  PIP_INSTALL="pip install -i $pip_mirror"
else
  PIP_INSTALL="pip install"
fi


if [ ! -f $hostfile ]; then
        echo "No hostfile exists at $hostfile, installing locally"
        local_only=1
fi

if [ "$skip_requirements" == "0" ]; then
    # Ensure dependencies are installed locally
    $PIP_SUDO $PIP_INSTALL -r requirements.txt
fi

# Build wheels
if [ "$third_party_install" == "1" ]; then
    echo "Checking out sub-module(s)"
    git submodule update --init --recursive

    echo "Building apex wheel"
    cd third_party/apex

    if [ "$apex_commit" != "" ]; then
        echo "Installing a non-standard version of apex at commit: $apex_commit"
        git fetch
        git checkout $apex_commit
    fi

    python setup.py --cpp_ext --cuda_ext bdist_wheel
    cd -

    echo "Installing apex locally so that deepspeed will build"
    $PIP_SUDO pip uninstall -y apex
    $PIP_SUDO $PIP_INSTALL third_party/apex/dist/apex*.whl
fi
if [ "$deepspeed_install" == "1" ]; then
    echo "Building deepspeed wheel"
    python setup.py bdist_wheel
fi

if [ "$local_only" == "1" ]; then
    if [ "$deepspeed_install" == "1" ]; then
        echo "Installing deepspeed"
        $PIP_SUDO pip uninstall -y deepspeed
        $PIP_SUDO $PIP_INSTALL dist/deepspeed*.whl
        python basic_install_test.py
        if [ $? == 0 ]; then
            echo "Installation is successful"
        else
            echo "Installation failed"
        fi
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
    if [ "$skip_requirements" == "0" ]; then
        pdsh -w $hosts "$PIP_SUDO $PIP_INSTALL -r ${tmp_wheel_path}/requirements.txt"
    fi
    if [ "$third_party_install" == "1" ]; then
        pdsh -w $hosts "$PIP_SUDO pip uninstall -y apex"
        pdcp -w $hosts third_party/apex/dist/apex*.whl $tmp_wheel_path/
        pdsh -w $hosts "$PIP_SUDO $PIP_INSTALL $tmp_wheel_path/apex*.whl"
        pdsh -w $hosts 'python -c "import apex"'
    fi
    if [ "$deepspeed_install" == "1" ]; then
        echo "Installing deepspeed"
        pdsh -w $hosts "$PIP_SUDO pip uninstall -y deepspeed"
        pdcp -w $hosts dist/deepspeed*.whl $tmp_wheel_path/
        pdcp -w $hosts basic_install_test.py $tmp_wheel_path/
        pdsh -w $hosts "$PIP_SUDO $PIP_INSTALL $tmp_wheel_path/deepspeed*.whl"
        pdsh -w $hosts "python $tmp_wheel_path/basic_install_test.py"
        echo "Installation is successful"
    fi
    pdsh -w $hosts "if [ -d $tmp_wheel_path ]; then rm $tmp_wheel_path/*.whl $tmp_wheel_path/basic_install_test.py $tmp_wheel_path/requirements.txt; rmdir $tmp_wheel_path; fi"
fi
