# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""
import subprocess


class Job(object):

    def __init__(self, cmd_line, output_file=None, work_dir=None):
        self.cmd_line = cmd_line
        self.output_file = output_file
        self.work_dir = work_dir
        self.output_fd = None

    def cmd(self):
        return self.cmd_line

    def get_stdout(self):
        return self.output_fd

    def get_stderr(self):
        return self.output_fd

    def get_cwd(self):
        return self.work_dir

    def open_output_file(self):
        if self.output_file is not None:
            self.output_fd = open(self.output_file, 'w')

    def close_output_file(self):
        if self.output_fd is not None:
            self.output_fd.close()
            self.output_fd = None


def run_job(job):
    args = ' '.join(job.cmd())
    print(f'args = {args}')
    job.open_output_file()
    proc = subprocess.run(args=args, shell=True, stdout=job.get_stdout(), stderr=job.get_stderr(), cwd=job.get_cwd())
    job.close_output_file()
    assert proc.returncode == 0, \
    f"This command failed: {job.cmd()}"
