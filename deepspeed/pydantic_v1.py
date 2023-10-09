# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pydantic v1 compatibility module.

Pydantic v2 introduced breaking changes that hinder its adoption:
https://docs.pydantic.dev/latest/migration/. To provide deepspeed users the option to
migrate to pydantic v2 on their own timeline, deepspeed uses this compatibility module
as a pydantic-version-agnostic alias for pydantic's v1 API.
"""

try:
    from pydantic.v1 import *  # noqa: F401
except ImportError:
    from pydantic import *  # noqa: F401
