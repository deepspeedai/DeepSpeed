// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
    Utilities for type casting torch tensors without data movement.
*/

#include <torch/extension.h>
#include <vector>

using namespace std;
at::Tensor cast_to_byte_tensor(at::Tensor& src_tensor);

std::vector<at::Tensor> cast_to_byte_tensor(std::vector<at::Tensor>& tensor_list);
