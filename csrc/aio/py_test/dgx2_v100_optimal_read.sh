python test_ds_aio.py \
        --read \
        --handle --io_size 400M \
        --loops 3  \
        --folder_to_device_mapping \
        /mnt/nvme23/aio:0 \
        /mnt/nvme23/aio:1 \
        /mnt/nvme23/aio:2 \
        /mnt/nvme23/aio:3 \
        /mnt/nvme45/aio:4 \
        /mnt/nvme45/aio:5 \
        /mnt/nvme45/aio:6 \
        /mnt/nvme45/aio:7 \
        /mnt/nvme67/aio:8 \
        /mnt/nvme67/aio:9 \
        /mnt/nvme67/aio:10 \
        /mnt/nvme67/aio:11 \
        /mnt/nvme89/aio:12  \
        /mnt/nvme89/aio:13  \
        /mnt/nvme89/aio:14  \
        /mnt/nvme89/aio:15  \
