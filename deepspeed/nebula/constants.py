# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#########################################
# nebula
#########################################
# Nebula. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
NEBULA_FORMAT = '''
nebula should be enabled as:
"session_params": {
  "nebula": {
        "enabled": true,
        "persistent_storage_path": "/foo/bar",
        "persistent_time_interval": 100,
        "num_of_version_in_retention": 2,
        "enable_nebula_load": true
    }
}
'''

NEBULA = "nebula"

NEBULA_ENABLED = "enabled"
NEBULA_ENABLED_DEFAULT = False

# There is a case where customer want to load the checkpoint saved
# by raw torch. Because nebula cannot load torch checkpoint directly
# as they have different folder structures to bring the gap for
# loading(the data are totally same in bytes for torch and nebula
# saving).
# In this case, we must disable nebula load to use raw torch load.
# Customer can just set NEBULA_ENABLE_NEBULA_LOAD to False. Then use
# original way of deepspeed to load, i.e. set the value of "--load".
NEBULA_ENABLE_NEBULA_LOAD = "enable_nebula_load"
NEBULA_ENABLE_NEBULA_LOAD_DEFAULT = True

# When you want to resume the previous checkpoint saved by nebula,
# you can set NEBULA_LOAD_PATH as the parent folder of checkpoint.
# If NEBULA_LOAD_PATH is None, the NEBULA_PERSISTENT_STORAGE_PATH
# will be the default path to load.
NEBULA_LOAD_PATH = "nebula_load_path"
NEBULA_LOAD_PATH_DEFAULT = None

# Nebula will save the checkpoint under NEBULA_LOAD_PATH in the
# asynchronous way.
NEBULA_PERSISTENT_STORAGE_PATH = "persistent_storage_path"
NEBULA_PERSISTENT_STORAGE_PATH_DEFAULT = None

# Time interval to trigger the nebula persistence.
NEBULA_PERSISTENT_TIME_INTERVAL = "persistent_time_interval"
NEBULA_PERSISTENT_TIME_INTERVAL_DEFAULT = 100

# Checkpoint number which will be kept in memory. Let us say,
# if the value is 2. Then we have checkpoints 1 and 2 are ready
# now. When it comes to checkpoint 3, the 1 will be removed if
# 1 has been persisted to disk.
NEBULA_NUM_OF_VERSION_IN_RETENTION = "num_of_version_in_retention"
NEBULA_NUM_OF_VERSION_IN_RETENTION_DEFAULT = 2

# Nebula envs
NEBULA_EXPORT_ENVS = [
    'DLTS_JOB_ID', 'DLTS_NUM_WORKER', 'NEBULA_PERSISTENT_STORAGE_PATH', 'NEBULA_PERSISTENT_TIME_INTERVAL',
    'AML_RUN_ID', 'AZUREML_RUN_TOKEN', 'AZUREML_WORKSPACE_SCOPE', 'AZUREML_EXPERIMENT_SCOPE',
    'AZUREML_RUN_HISTORY_SERVICE_ENDPOINT', 'AZUREML_RUN_ID', 'NEBULA_MEMORY_BUFFER_SIZE',
    'AZUREML_PARAMETER_ITPJOB_NAME', 'FC_TASKROLE_NAME', 'FC_TASK_INDEX', 'MASTER_HOST', 'LOCAL_HOST',
    'AZUREML_BLOB_ACCOUNT_NAME', 'AZUREML_BLOB_ACCOUNT_KEY'
]

# ITP env files
DLTS_POD_ENV_PATH = '/dlts-runtime/env/pod.env'
