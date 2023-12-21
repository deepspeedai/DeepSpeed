// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#define TOP_K_SWITCH(N_TOP_K, ...)         \
    [&] {                                  \
        if (1 == N_TOP_K) {                \
            constexpr int CONST_TOP_K = 1; \
            __VA_ARGS__();                 \
        } else if (2 == N_TOP_K) {         \
            constexpr int CONST_TOP_K = 2; \
            __VA_ARGS__();                 \
        }                                  \
    }()
