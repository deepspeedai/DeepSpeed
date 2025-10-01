// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#pragma once

namespace dc {

void register_graph_z3(long graph_id, const std::vector<long>& ds_ids);
void register_graph_ops_z3(long graph_id,
                           const std::vector<std::string>& op_names,
                           const std::vector<long>& n_args);
void register_bwd_graph_ops_z3(long graph_id,
                               const std::vector<std::string>& op_names,
                               const std::vector<long>& n_args);
void register_z3_param(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool persistent);
at::Tensor allgather_param(at::Tensor param_tensor,
                           long graph_id,
                           long ds_id,
                           std::optional<at::ScalarType> dtype);
void set_persistent(long ds_id);
void prefetch_params_fused(long graph_id,
                           const std::vector<at::Tensor>& params,
                           const std::vector<long>& ds_ids,
                           const std::optional<std::vector<at::ScalarType>>& dtypes);
void prefetch_params_fused_meta(long graph_id,
                                const std::vector<at::Tensor>& params,
                                const std::vector<long>& ds_ids,
                                const std::optional<std::vector<at::ScalarType>>& dtypes);
// for profiling
void invalidate_gathered_param(long ds_id);
void clear_all_gathered_params();
at::Tensor allgather_param_meta(at::Tensor param_tensor,
                                long graph_id,
                                long ds_id,
                                std::optional<at::ScalarType> dtype);
at::Tensor release_param(at::Tensor dummy, long graph_id, long ds_id, long n_users);
at::Tensor release_param_meta(at::Tensor dummy, long graph_id, long ds_id, long n_users);
at::Tensor wait_allgather(at::Tensor v, long graph_id, const long ds_id);
at::Tensor wait_allgather_meta(at::Tensor v, long graph_id, long ds_id);
at::Tensor offload_tensor(at::Tensor tensor, long graph_id, long id);
at::Tensor reload_tensor(at::Tensor tensor, long graph_id, long id);
at::Tensor wait_offload(at::Tensor tensor, long graph_id, long id);
at::Tensor wait_reload(at::Tensor tensor, long graph_id, long id);
void reload_parameter(at::Tensor tensor, long graph_id, long id);
void offload_parameter(at::Tensor tensor, long graph_id, long id);
void reload_parameter_meta(at::Tensor tensor, long graph_id, long id);
void offload_parameter_meta(at::Tensor tensor, long graph_id, long id);
void end_backward(long graph_id);
}  // namespace dc
