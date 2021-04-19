"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""
#########################################
# AIO
#########################################
AIO_FORMAT = '''
"aio": {
  "block_size": 1048576,
  "queue_depth": 8,
  "thread_count": 1,
  "single_submit": false,
  "overlap_events": true
}
'''
AIO = "aio"
AIO_BLOCK_SIZE = "block_size"
AIO_BLOCK_SIZE_DEFAULT = 1048576
AIO_QUEUE_DEPTH = "queue_depth"
AIO_QUEUE_DEPTH_DEFAULT = 8
AIO_THREAD_COUNT = "thread_count"
AIO_THREAD_COUNT_DEFAULT = 1
AIO_SINGLE_SUBMIT = "single_submit"
AIO_SINGLE_SUBMIT_DEFAULT = False
AIO_OVERLAP_EVENTS = "overlap_events"
AIO_OVERLAP_EVENTS_DEFAULT = True
