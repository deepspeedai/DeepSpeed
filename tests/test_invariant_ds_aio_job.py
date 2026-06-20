import pytest
import shlex


@pytest.mark.parametrize("cmd_args,expected_args", [
    # Exact exploit case: path with spaces gets re-split into multiple args
    (["/usr/bin/prog", "/path/with spaces/file.txt", "--flag"],
     ["/usr/bin/prog", "/path/with spaces/file.txt", "--flag"]),
    # Boundary case: argument with shell metacharacters
    (["/usr/bin/prog", "file;rm -rf /", "--output", "result.txt"],
     ["/usr/bin/prog", "file;rm -rf /", "--output", "result.txt"]),
    # Boundary case: argument with quotes that shlex interprets
    (["/usr/bin/prog", 'arg"with"quotes', "--verbose"],
     ["/usr/bin/prog", 'arg"with"quotes', "--verbose"]),
    # Valid input: normal arguments without special characters
    (["/usr/bin/prog", "simple_file.txt", "--threads", "4"],
     ["/usr/bin/prog", "simple_file.txt", "--threads", "4"]),
    # Exploit case: tab and newline in argument
    (["/usr/bin/prog", "file\twith\ttabs", "arg\nwith\nnewlines"],
     ["/usr/bin/prog", "file\twith\ttabs", "arg\nwith\nnewlines"]),
])
def test_command_argument_boundaries_preserved(cmd_args, expected_args):
    """Invariant: Command argument boundaries must be preserved exactly as provided.
    
    The join+split pattern in run_job can alter argument boundaries when arguments
    contain spaces, quotes, or other shell metacharacters. This test verifies that
    the transformation does NOT alter the intended argument list.
    """
    # This is the vulnerable pattern from deepspeed/nvme/ds_aio_job.py
    transformed_args = shlex.split(' '.join(cmd_args))
    
    # The security invariant: argument boundaries MUST be preserved
    # If this assertion fails, it demonstrates the vulnerability where
    # the join+split pattern reinterprets argument boundaries
    assert transformed_args == expected_args, (
        f"Argument boundary violation detected!\n"
        f"  Original args: {cmd_args}\n"
        f"  After join+split: {transformed_args}\n"
        f"  Expected: {expected_args}\n"
        f"  The join+split pattern in run_job() corrupts argument boundaries."
    )