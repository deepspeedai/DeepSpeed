#########################################
# Tensorboard
#########################################
# Tensorboard. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
TENSORBOARD_FORMAT = '''
Tensorboard can be specified as:
"tensorboard": {
  "enabled": true,
  "output_path": "/home/myname/foo",
  "job_name": "model_lr2e-5_epoch3_seed2_seq64"
}
'''
TENSORBOARD = "tensorboard"

# Tensorboard enable signal
TENSORBOARD_ENABLED = "enabled"
TENSORBOARD_ENABLED_DEFAULT = False

# Tensorboard output path
TENSORBOARD_OUTPUT_PATH = "output_path"
TENSORBOARD_OUTPUT_PATH_DEFAULT = ""

# Tensorboard job name
TENSORBOARD_JOB_NAME = "job_name"
TENSORBOARD_JOB_NAME_DEFAULT = "DeepSpeedJobName"

#########################################
# Wandb
#########################################
# Wandb. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
WANDB_FORMAT = '''
Wandb can be specified as:
"wandb": {
  "enabled": true,
  "team_name": "deepspeed"
  "project_name": "zero"
  "group_name": "zero: stage 3",
}
'''
WANDB = "wandb"

# Wandb enable signal
WANDB_ENABLED = "enabled"
WANDB_ENABLED_DEFAULT = False

# Wandb team
WANDB_TEAM_NAME = "team"
WANDB_TEAM_NAME_DEFAULT = None

# Wandb project
WANDB_PROJECT_NAME = "project"
WANDB_PROJECT_NAME_DEFAULT = "deepspeed"

# Wandb group
WANDB_GROUP_NAME = "group"
WANDB_GROUP_NAME_DEFAULT = None

#########################################
# csv monitor
#########################################
# Basic CSV monitor. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
CSV_FORMAT = '''
The basic csv monitor can be specified as:
"csv_monitor": {
  "enabled": true,
  "output_path": "/home/myname/foo",
  "job_name": "model_lr2e-5_epoch3_seed2_seq64"
}
'''
CSV_MONITOR = "csv_monitor"

# csv monitor enable signal
CSV_MONITOR_ENABLED = "enabled"
CSV_MONITOR_ENABLED_DEFAULT = False

# csv monitor output path
CSV_MONITOR_OUTPUT_PATH = "output_path"
CSV_MONITOR_OUTPUT_PATH_DEFAULT = ""

# csv_monitor job name
CSV_MONITOR_JOB_NAME = "job_name"
CSV_MONITOR_JOB_NAME_DEFAULT = "DeepSpeedJobName"
