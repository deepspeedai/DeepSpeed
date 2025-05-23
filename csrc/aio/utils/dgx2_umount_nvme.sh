#!/bin/bash

UMOUNT_CMD="sudo umount -v"

for md in md127 md126 md125 md124; do
    mnt_device=/dev/${md}
    ${UMOUNT_CMD} ${mnt_device}
done

lsblk -f
