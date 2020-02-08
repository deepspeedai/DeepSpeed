#!/bin/bash

name=${1-deepspeed}
docker exec -i -t $name /bin/bash
