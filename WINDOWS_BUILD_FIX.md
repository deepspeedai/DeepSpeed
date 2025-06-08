# Windows Build Fix for LNK1181 Errors

This document describes the fix for the Windows build issue where users encounter `LNK1181` errors about missing `aio.lib` and `cufile.lib` files.

## Problem Description

When running `build_win.bat` on Windows, users encountered these errors:
```
LINK : fatal error LNK1181: не удается открыть входной файл "aio.lib"
LINK : fatal error LNK1181: не удается открыть входной файл "cufile.lib"
```

## Root Cause

The issue was caused by the build system attempting to link against libraries for operations that are not supported on Windows:

1. **AIO (Async I/O)** - Uses Linux-specific `libaio` library
2. **GDS (GPU Direct Storage)** - Uses NVIDIA's `cufile` library which is not available on Windows

Even though these operations were disabled in `build_win.bat` with environment variables, the compatibility checks in the op builders were not properly detecting Windows and returning early.

## Solution

### 1. Enhanced build_win.bat

- Added better error handling and informative messages
- Ensured `DS_BUILD_OPS=1` is set for Windows pre-compilation
- Added diagnostic output to help troubleshoot issues
- Improved error messages with common solutions

### 2. Fixed Op Builder Compatibility Checks

Updated all async_io builders to properly detect Windows:
- `op_builder/async_io.py`
- `op_builder/cpu/async_io.py` 
- `op_builder/npu/async_io.py`
- `op_builder/xpu/async_io.py`

Updated GDS builder:
- `op_builder/gds.py`

Each now includes an early Windows detection check:
```python
def is_compatible(self, verbose=False):
    # AIO/GDS is not supported on Windows
    if sys.platform == "win32":
        if verbose:
            self.warning(f"{self.NAME} is not supported on Windows")
        return False
    # ... rest of compatibility checks
```

### 3. Updated Documentation

Enhanced `CONTRIBUTING.md` with:
- Windows-specific development section
- Prerequisites for Windows builds
- Common troubleshooting steps
- Known limitations on Windows

## Files Modified

1. `build_win.bat` - Enhanced build script with better error handling
2. `op_builder/async_io.py` - Added Windows compatibility check
3. `op_builder/cpu/async_io.py` - Added Windows compatibility check  
4. `op_builder/npu/async_io.py` - Added Windows compatibility check
5. `op_builder/xpu/async_io.py` - Added Windows compatibility check
6. `op_builder/gds.py` - Added Windows compatibility check
7. `CONTRIBUTING.md` - Added Windows development section

## Testing

A test script `test_windows_build.py` is provided to verify the fixes work correctly.

Run it with:
```bash
python test_windows_build.py
```

## Usage

### For Users Experiencing the Issue

1. Pull the latest changes with these fixes
2. Ensure you're using "Developer Command Prompt for VS 2022" with Administrator privileges
3. Run `build_win.bat`
4. The build should now complete without LNK1181 errors

### For Developers

The fixes ensure that:
- Windows-incompatible operations are properly detected and skipped
- Build process provides clear error messages and solutions
- Documentation includes Windows-specific guidance

## Verification

After applying these fixes:
1. AIO operations will be properly disabled on Windows
2. GDS operations will be properly disabled on Windows  
3. Build process will complete successfully
4. No more LNK1181 errors for missing aio.lib or cufile.lib

## Future Considerations

- Monitor for other Windows-incompatible operations that may need similar fixes
- Consider adding automated Windows build testing to CI/CD pipeline
- Keep Windows limitations documented as new features are added
