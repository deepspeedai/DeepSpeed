#!/bin/bash

MOUNT_CMD="sudo mount -v -o data=ordered"

for dir in nvme23 nvme45 nvme67 nvme89; do
    mnt_point=/mnt/${dir}
    sudo mkdir -p ${mnt_point}
    sudo chmod -R a+rw ${mnt_point}
done
${MOUNT_CMD} /dev/md127 /mnt/nvme23
${MOUNT_CMD} /dev/md126 /mnt/nvme45
${MOUNT_CMD} /dev/md125 /mnt/nvme67
${MOUNT_CMD} /dev/md124 /mnt/nvme89

lsblk -f
