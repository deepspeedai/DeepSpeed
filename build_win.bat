@echo off

set DS_BUILD_AIO=0
set DS_BUILD_CUTLASS_OPS=0
set DS_BUILD_EVOFORMER_ATTN=0
set DS_BUILD_FP_QUANTIZER=0
set DS_BUILD_RAGGED_DEVICE_OPS=0
set DS_BUILD_SPARSE_ATTN=0

echo Administrative permissions required. Detecting permissions...

net session >nul 2>&1
if %errorLevel% == 0 (
    echo Success: Administrative permissions confirmed.
) else (
    echo Failure: Current permissions inadequate.
    goto end
)


python setup.py bdist_wheel

:end
