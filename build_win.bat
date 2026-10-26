@echo off

set DISTUTILS_USE_SDK=1
set DS_BUILD_OPS=1
set DS_BUILD_AIO=1
set DS_BUILD_PIN_MEMORY=1
set DS_ENABLE_NINJA=1

python -m build --wheel --no-isolation

:end
