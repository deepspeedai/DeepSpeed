// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

enum ActivationType {
    GELU = 0,
    RELU = 1,
    SILU = 2,
    GEGLU = 3,
    ReGLU = 4,
    SiGLU = 5,
    IDENTITY = 6,
    InvalidType = -1
};
