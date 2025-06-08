@echo off

REM Set CUDA environment
set CUDA_HOME=%CUDA_PATH%
set DISTUTILS_USE_SDK=1

REM Disable Windows-incompatible operations
REM AIO (Async I/O) is not supported on Windows
set DS_BUILD_AIO=0
REM GDS (GPU Direct Storage) is not supported on Windows
set DS_BUILD_GDS=0
REM Disable other potentially problematic operations
set DS_BUILD_CUTLASS_OPS=0
set DS_BUILD_EVOFORMER_ATTN=0
set DS_BUILD_FP_QUANTIZER=0
set DS_BUILD_RAGGED_DEVICE_OPS=0
set DS_BUILD_SPARSE_ATTN=0
set DS_BUILD_DEEP_COMPILE=0

REM Ensure DS_BUILD_OPS is set to enable Windows pre-compilation
set DS_BUILD_OPS=1

echo Building DeepSpeed for Windows...
echo CUDA_HOME=%CUDA_HOME%
echo DS_BUILD_AIO=%DS_BUILD_AIO%
echo DS_BUILD_GDS=%DS_BUILD_GDS%
echo DS_BUILD_OPS=%DS_BUILD_OPS%

python -m build --wheel --no-isolation

if %ERRORLEVEL% neq 0 (
    echo Build failed with error code %ERRORLEVEL%
    echo.
    echo Common issues and solutions:
    echo 1. Ensure you are running in "Developer Command Prompt for VS 2022"
    echo 2. Verify CUDA toolkit is properly installed and CUDA_PATH is set
    echo 3. Check that PyTorch with CUDA support is installed
    echo 4. Run as Administrator if you encounter permission issues
    pause
    exit /b %ERRORLEVEL%
)

echo Build completed successfully!
:end
