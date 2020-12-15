try:
    #  This is populated by setup.py
    from .git_version_info_installed import *
except ModuleNotFoundError:
    import os
    if os.path.isfile('version.txt'):
        # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
        version = open('version.txt', 'r').read().strip()
    else:
        version = "0.0.0"
    git_hash = '[none]'
    git_branch = '[none]'

    from .ops.op_builder import ALL_OPS
    installed_ops = dict.fromkeys(ALL_OPS.keys(), False)
    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
    torch_info = {'version': "0.0", "cuda_version": "0.0"}
