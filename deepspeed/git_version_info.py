try:
    #  This is populated by setup.py
    from .git_version_info_installed import *
except ModuleNotFoundError:
    # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
    version = '0.3.0+[none]'
    git_hash = '[none]'
    git_branch = '[none]'
    installed_ops = {
        'lamb': False,
        'transformer': False,
        'sparse-attn': False,
        'cpu-adam': False
    }
