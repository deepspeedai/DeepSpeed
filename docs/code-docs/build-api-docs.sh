#!/bin/bash

sphinx-apidoc -f -o source ../../deepspeed
make html
