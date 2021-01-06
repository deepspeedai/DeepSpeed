from . import adam
from . import lamb
from . import sparse_attention
from . import transformer

from .transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from .module_inject import replace_module

from ..git_version_info import compatible_ops as __compatible_ops__
