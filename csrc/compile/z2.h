// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#pragma once

namespace dc {

void register_graph_z2(long graph_id, const std::vector<long>& ds_ids);

}  // namespace dc
