import deepspeed
from deepspeed.ops.aio import AsyncIOBuilder
assert AsyncIOBuilder().is_compatible()
