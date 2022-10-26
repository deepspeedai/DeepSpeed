@echo off

set DS_BUILD_AIO=0
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
