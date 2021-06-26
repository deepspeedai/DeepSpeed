""" debug utils """

import fcntl

# for debug purposes map module and param objects to their fully qualified names
module_names = {}
param_names = {}


def debug_extract_module_and_param_names(model):
    # extract the fully qualified names as soon as the model is acquired
    global module_names
    global param_names
    # XXX: can probably make a map of param2module and vice-versa
    module_names = {module: name for name, module in model.named_modules()}
    param_names = {param: name for name, param in model.named_parameters()}


def debug_module2name(module):
    if module in module_names:
        return module_names[module]
    else:
        return "unknown"


def debug_module2name_id(module):
    return f"name={debug_module2name(module)} id={module.id}"


def debug_module2name_class(module):
    return f"name={debug_module2name(module)} {module.__class__.__name__}"


def debug_param2name(param):
    if param in param_names:
        return param_names[param]
    else:
        return "unknown"


def debug_param2name_id(param):
    return f"name={debug_param2name(param)} id={param.ds_id}"


def debug_param2name_id_shape(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape}"


def debug_param2name_id_shape_device(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} device={param.device}"


def debug_param2name_id_numel(param):
    return f"name={debug_param2name(param)} id={param.ds_id} numel={param.numel()}"


def debug_param2name_id_shape_status(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} status={param.ds_status}"


def printflock(*msgs):
    """

    For printing messages for all concurrent gpus w/o getting interleaved text.

    This is useful when debugging issues where multi-gpus don't sync.

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions with ::

        def print_rank_0(message, debug=False, force=False):
            rank = torch.distributed.get_rank()
            printflock(f"[{rank}] {message}")
    3. run the program and you get both logs non-interleaved

    But this makes it very difficult to make sense of the output, so the ``log_rank_file`` helper
    function might be more useful, as it's easier to send each log stream into a separate file and
    then compare those.

    """

    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


fh = None


def log_rank_file(rank, *msgs):
    """
    Print to a log file of the given rank

    This is useful for debugging hanging in sync processes. Here is a possible workflow:

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions of print_rank_0 in those files with ::

        def print_rank_0(message, debug=False, force=False):
            rank = torch.distributed.get_rank()
            log_rank_file(rank, message)

    3. run the program
    4. fix up the expected differences, e.g. different cuda numbers ::

        perl -pi -e 's|cuda:1|cuda:0|' log_rank_*

    5. now diff and see where names and ids diverge - you will find where the gpus don't do the same
    work (e.g. when some layers get conditionally skipped on one gpu but not all)

        diff -u log_rank_0.txt log_rank_1.txt | less

    """
    global fh
    if fh is None:
        fh = open(f"log_rank_{rank}.txt", "w")
    for m in msgs:
        fh.write(f"{m}\n")
    fh.flush()
