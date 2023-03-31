# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .monitor import Monitor
import os

import deepspeed.comm as dist


class csvMonitor(Monitor):

    def __init__(self, csv_config):
        super().__init__(csv_config)
        self.filenames = []
        self.enabled = csv_config.enabled
        self.output_path = csv_config.output_path
        self.job_name = csv_config.job_name
        self.log_dir = self.setup_log_dir()

    def setup_log_dir(self, base=os.path.join(os.path.expanduser("~"), "csv_monitor")):
        if self.enabled and dist.get_rank() == 0:
            if self.output_path is not None:
                log_dir = os.path.join(self.output_path, self.job_name)
            # NOTE: This code path currently is never used since the default tensorboard_output_path is an empty string and not None. Saving it in case we want this functionality in the future.
            else:
                if "DLWS_JOB_ID" in os.environ:
                    infra_job_id = os.environ["DLWS_JOB_ID"]
                elif "DLTS_JOB_ID" in os.environ:
                    infra_job_id = os.environ["DLTS_JOB_ID"]
                else:
                    infra_job_id = "unknown-job-id"

                csv_monitor_dir_name = os.path.join(infra_job_id, "logs")
                log_dir = os.path.join(base, csv_monitor_dir_name, self.job_name)
            os.makedirs(log_dir, exist_ok=True)
            return log_dir

    def write_events(self, event_list):
        if self.enabled and dist.get_rank() == 0:
            import csv
            # We assume each event_list element is a tensorboard-style tuple in the format: (log_name: String, value, step: Int)
            for event in event_list:
                log_name = event[0]
                value = event[1]
                step = event[2]

                # Set the header to the log_name
                # Need this check because the deepspeed engine currently formats log strings to separate with '/'
                if '/' in log_name:
                    record_splits = log_name.split('/')
                    header = record_splits[len(record_splits) - 1]
                else:
                    header = log_name

                # sanitize common naming conventions into filename
                filename = log_name.replace('/', '_').replace(' ', '_')
                fname = self.log_dir + '/' + filename + '.csv'

                # Open file and record event. Insert header if this is the first time writing
                with open(fname, 'a+') as csv_monitor_file:
                    csv_monitor_writer = csv.writer(csv_monitor_file)
                    if filename not in self.filenames:
                        self.filenames.append(filename)
                        csv_monitor_writer.writerow(['step', header])
                    csv_monitor_writer.writerow([step, value])
