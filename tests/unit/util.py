# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator, is_current_accelerator_supported
from deepspeed.git_version_info import torch_info

from io import StringIO
import deepspeed
import logging
import pytest
import re
import sys
import torch


def skip_on_arch(min_arch=7):
    if get_accelerator().device_name() == 'cuda':
        if torch.cuda.get_device_capability()[0] < min_arch:  #ignore-cuda
            pytest.skip(f"needs higher compute capability than {min_arch}")
    else:
        assert is_current_accelerator_supported()
        return


def skip_on_cuda(valid_cuda):
    split_version = lambda x: map(int, x.split('.')[:2])
    if get_accelerator().device_name() == 'cuda':
        CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])
        CUDA_VERSION = (CUDA_MAJOR * 10) + CUDA_MINOR
        if valid_cuda.count(CUDA_VERSION) == 0:
            pytest.skip(f"requires cuda versions {valid_cuda}")
    else:
        assert is_current_accelerator_supported()
        return


def bf16_required_version_check(accelerator_check=True):
    split_version = lambda x: map(int, x.split('.')[:2])

    # torch_info may have stale/zero values if installed without --no-build-isolation
    # In that case, fall back to runtime detection
    if torch_info['version'] == '0.0':
        # Use runtime torch version
        TORCH_MAJOR, TORCH_MINOR = split_version(torch.__version__)
    else:
        TORCH_MAJOR, TORCH_MINOR = split_version(torch_info['version'])

    if torch_info['nccl_version'] == '0.0':
        # Use runtime NCCL version if available
        if torch.cuda.is_available():  #ignore-cuda
            try:
                nccl_ver = torch.cuda.nccl.version()  #ignore-cuda
                NCCL_MAJOR, NCCL_MINOR = nccl_ver[0], nccl_ver[1]
            except (AttributeError, RuntimeError):
                NCCL_MAJOR, NCCL_MINOR = 0, 0
        else:
            # No CUDA means no NCCL; NPU/HPU/XPU have separate checks below
            NCCL_MAJOR, NCCL_MINOR = 0, 0
    else:
        NCCL_MAJOR, NCCL_MINOR = split_version(torch_info['nccl_version'])

    if torch_info['cuda_version'] == '0.0':
        # Use runtime CUDA version
        if torch.cuda.is_available():  #ignore-cuda
            cuda_ver = torch.version.cuda
            if cuda_ver:
                CUDA_MAJOR, CUDA_MINOR = split_version(cuda_ver)
            else:
                CUDA_MAJOR, CUDA_MINOR = 0, 0
        else:
            CUDA_MAJOR, CUDA_MINOR = 0, 0
    else:
        CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])

    # Sometimes bf16 tests are runnable even if not natively supported by accelerator
    if accelerator_check:
        accelerator_pass = get_accelerator().is_bf16_supported()
    else:
        accelerator_pass = True

    torch_version_available = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
    cuda_version_available = CUDA_MAJOR >= 11
    nccl_version_available = NCCL_MAJOR > 2 or (NCCL_MAJOR == 2 and NCCL_MINOR >= 10)
    npu_available = get_accelerator().device_name() == 'npu'
    hpu_available = get_accelerator().device_name() == 'hpu'
    xpu_available = get_accelerator().device_name() == 'xpu'

    if torch_version_available and cuda_version_available and nccl_version_available and accelerator_pass:
        return True
    elif npu_available:
        return True
    elif hpu_available:
        return True
    elif xpu_available:
        return True
    else:
        return False


def required_amp_check():
    from importlib.util import find_spec
    if find_spec('apex') is None:
        return False
    else:
        return True


class no_child_process_in_deepspeed_io:

    def __enter__(self):
        # deepspeed_io defaults to creating a dataloader that uses a
        # multiprocessing pool. Our tests use pools and we cannot nest pools in
        # python. Therefore we're injecting this kwarg to ensure that no pools
        # are used in the dataloader.
        self.old_method = deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io

        def new_method(*args, **kwargs):
            kwargs["num_local_io_workers"] = 0
            return self.old_method(*args, **kwargs)

        deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io = new_method

    def __exit__(self, *_):
        deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io = self.old_method


def torch_assert_equal(actual, expected, **kwargs) -> None:
    """
    Compare two tensors or non-tensor numbers for their equality.
    Add msg=blah to add an additional comment to when assert fails.
    """
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, **kwargs)


def torch_assert_close(actual, expected, **kwargs) -> None:
    """
    Compare two tensors or non-tensor numbers for their closeness.

    Add msg=blah to add an additional comment to when assert fails.

    For default values of `rtol` and `atol` which are dtype dependent, see the table at https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close
    For example for bf16 it is `rtol=1.6e-2` and `atol=1e-5`.

    The check doesn't assert when `|a - b| <= (atol + rtol * |b|)`
    """
    torch.testing.assert_close(actual, expected, **kwargs)


def torch_assert_dicts_of_tensors_equal(actual, expected, **kwargs):
    """
    Compare two dicts of tensors or non-tensor numbers for their equality.
    Add msg=blah to add an additional comment to when assert fails.
    """
    for k in actual.keys():
        torch.testing.assert_close(actual[k], expected[k], rtol=0.0, atol=0.0, **kwargs)


# CaptureStd, CaptureLogger context managers from https://github.com/stas00/ml-engineering/blob/master/testing/testing_utils.py


# When any function contains print() calls that get overwritten, like progress bars,
# a special care needs to be applied, since under pytest -s captured output (capsys
# or contextlib.redirect_stdout) contains any temporary printed strings, followed by
# \r's. This helper function ensures that the buffer will contain the same output
# with and without -s in pytest, by turning:
# foo bar\r tar mar\r final message
# into:
# final message
# it can handle a single string or a multiline buffer
def apply_print_resets(buf):
    return re.sub(r"^.*\r", "", buf, 0, re.M)


class CaptureStd:
    """
    Context manager to capture:

    - stdout: replay it, clean it up and make it available via ``obj.out``
    - stderr: replay it and make it available via ``obj.err``
    - combined: combined the chosen streams and make it available via ``obj.combined``

    init arguments:

    - out - capture stdout:`` True``/``False``, default ``True``
    - err - capture stdout: ``True``/``False``, default ``True``
    - replay - whether to replay or not: ``True``/``False``, default ``True``. By default each
    captured stream gets replayed back on context's exit, so that one can see what the test was
    doing. If this is a not wanted behavior and the captured data shouldn't be replayed, pass
    ``replay=False`` to disable this feature.

    Examples::

        # to capture stdout only with auto-replay
        with CaptureStdout() as cs:
            print("Secret message")
        assert "message" in cs.out

        # to capture stderr only with auto-replay
        import sys
        with CaptureStderr() as cs:
            print("Warning: ", file=sys.stderr)
        assert "Warning" in cs.err

        # to capture both streams with auto-replay
        with CaptureStd() as cs:
            print("Secret message")
            print("Warning: ", file=sys.stderr)
        assert "message" in cs.out
        assert "Warning" in cs.err

        # to capture just one of the streams, and not the other, with auto-replay
        with CaptureStd(err=False) as cs:
            print("Secret message")
        assert "message" in cs.out
        # but best use the stream-specific subclasses

        # to capture without auto-replay
        with CaptureStd(replay=False) as cs:
            print("Secret message")
        assert "message" in cs.out

        # sometimes it's easier to not try to figure out if it's stdout or stderr, and yet at
        # other times the software may send the same output to stderr or stdout depending on
        # environment, so to make the test robust a combined entry of both streams is available

    """

    def __init__(self, out=True, err=True, replay=True):
        self.replay = replay

        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

            self.combined = "error: CaptureStd context is unfinished yet, called too early"

    def __enter__(self):
        if self.out_buf is not None:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf is not None:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        self.combined = ""

        return self

    def __exit__(self, *exc):
        if self.out_buf is not None:
            sys.stdout = self.out_old
            captured = self.out_buf.getvalue()
            if self.replay:
                sys.stdout.write(captured)
            self.out = apply_print_resets(captured)
            self.combined += self.out

        if self.err_buf is not None:
            sys.stderr = self.err_old
            captured = self.err_buf.getvalue()
            if self.replay:
                sys.stderr.write(captured)
            self.err = captured
            self.combined += self.err

    def __repr__(self):
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


# in tests it's the best to capture only the stream that's wanted, otherwise
# it's easy to miss things, so unless you need to capture both streams, use the
# subclasses below (less typing). Or alternatively, configure `CaptureStd` to
# disable the stream you don't need to test.


class CaptureStdout(CaptureStd):
    """Same as CaptureStd but captures only stdout"""

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """Same as CaptureStd but captures only stderr"""

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
    Context manager to capture `logging` streams

    Args:

    - logger: 'logging` logger object

    Results:
        The captured output is available via `self.out`

    Example::

        >>> from transformers import logging
        >>> from transformers.testing_utils import CaptureLogger

        >>> msg = "Testing 1, 2, 3"
        >>> logging.set_verbosity_info()
        >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart")
        >>> with CaptureLogger(logger) as cl:
        ...     logger.info(msg)
        >>> assert cl.out, msg+"\n"
    """

    def __init__(self, logger):
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        return f"captured: {self.out}\n"
