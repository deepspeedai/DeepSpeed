#!/bin/bash

name=${1-deepspeed}
docker exec -i -w /home/deepspeed -t $name /bin/bash
