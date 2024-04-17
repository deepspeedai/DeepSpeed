// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include <oneapi/ccl.hpp>
#include "shm.h"

// #define DO_PROFILE
#ifdef DO_PROFILE
#include <cfloat>
#include <chrono>
#endif

// Communication settings
static int world_rank = -1;
static int world_size = -1;

static std::set<int> _comm_ids;
static std::set<int> _colors;
static std::vector<ccl::communicator> _ccl_comms;
static ccl::shared_ptr_class<ccl::kvs> sub_kvs;
static std::map<std::vector<int>, int> group_to_comm_id;

ccl::communicator& _get_comm_from_group() { return _ccl_comms[0]; }
ccl::communicator& _get_comm_from_group(py::object group) { return _ccl_comms[0]; }
ccl::communicator& _get_comm_from_group(std::vector<int> ranks)
{
    if (group_to_comm_id.find(ranks) != group_to_comm_id.end()) {
        auto id = group_to_comm_id.find(ranks);
        return _ccl_comms[id->second];
    }
    return _ccl_comms[0];
}

#define CCLCHECK(cmd) \
    do {              \
        cmd;          \
    } while (0)

#define KVS_CREATE_SUCCESS 0
#define KVS_CREATE_FAILURE -1

static bool is_initialized = 0;

static ccl::shared_ptr_class<ccl::kvs> kvs;

static bool all_ranks_local_p = false;

void initialize(int size, int rank, torch::Tensor& kvs_data)
{
    if (is_initialized) return;

    // Check whether all ranks is on the same physical machine.
    // If true, we will use an SHM based low latency allreduce

    auto ls_string = std::getenv("LOCAL_SIZE");
    int ls = 0;
    if (ls_string != NULL) { ls = std::stoi(std::getenv("LOCAL_SIZE")); }

    if (size >= 1 && size == ls) { all_ranks_local_p = true; }

    world_size = size;
    world_rank = rank;
    is_initialized = 1;

    ccl::kvs::address_type main_addr;

    if (rank != 0) {
        memcpy(main_addr.data(), kvs_data.data_ptr(), main_addr.size());
        kvs = ccl::create_kvs(main_addr);
    }

    _ccl_comms.emplace_back(ccl::create_communicator(size, rank, kvs));

    auto addr_string = std::getenv("MASTER_ADDR");
    if (addr_string == NULL) { addr_string = ""; }
    auto port_string = std::getenv("MASTER_PORT");
    if (port_string == NULL) { port_string = ""; }

    if (all_ranks_local_p) { shm_initialize(size, rank, addr_string, port_string); }
}

/*
    rank == 0: create main kvs and return its address
    rank == else: return an empty address
*/
std::vector<uint8_t> get_kvs_addr(int rank)
{
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        ccl::kvs::address_type main_addr = kvs->get_address();
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    } else {
        ccl::kvs::address_type main_addr;
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    }
}

int get_rank(int group = 0) { return world_rank; }

int get_world_size(int group = 0) { return world_size; }

// Find the next ordered, unique value to a set. E.g. <0,1,2,7> --> 3
int next_unique_val(std::set<int> s)
{
    std::set<int>::iterator itr;
    // Base case. Add 0 to start of set.
    if (s.empty() || *s.begin() != 0) {
        return 0;
        // second base case where s = {0} (the case of s = {n != 0} is caught above)
    } else if (s.size() == 1) {
        return 1;
    } else {
        int prev_val = *s.begin();
        for (itr = std::next(s.begin()); itr != s.end(); itr++) {
            if (*itr != prev_val + 1) { return prev_val + 1; }
            prev_val = *itr;
        }
        return *(s.end()) + 1;
    }
}

std::vector<uint8_t> get_sub_kvs_addr(bool first)
{
    if (first) {
        sub_kvs = ccl::create_main_kvs();
        ccl::kvs::address_type main_addr = sub_kvs->get_address();
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    } else {
        ccl::kvs::address_type main_addr;
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    }
}

void initialize_sub_comm(int size, int rank, torch::Tensor& kvs_data, std::vector<int> ranks)
{
    ccl::kvs::address_type main_addr;
    if (rank != 0) {
        memcpy(main_addr.data(), kvs_data.data_ptr(), main_addr.size());
        sub_kvs = ccl::create_kvs(main_addr);
    }
    _ccl_comms.push_back(ccl::create_communicator(size, rank, sub_kvs));
    group_to_comm_id[ranks] = _ccl_comms.size() - 1;
}

ccl::datatype get_ccl_datatype(c10::ScalarType type)
{
    ccl::datatype ccl_type;
    switch (type) {
        case c10::ScalarType::Int: ccl_type = ccl::datatype::int32; break;
        case c10::ScalarType::Long: ccl_type = ccl::datatype::int64; break;
        case c10::ScalarType::Float: ccl_type = ccl::datatype::float32; break;
        case c10::ScalarType::Double: ccl_type = ccl::datatype::float64; break;
        case c10::ScalarType::BFloat16: ccl_type = ccl::datatype::bfloat16; break;
        case c10::ScalarType::Half: ccl_type = ccl::datatype::float16; break;
        default: ccl_type = ccl::datatype::int8;
    }
    return ccl_type;
}

ccl::reduction get_ccl_reduce_op(py::object op, at::Tensor& input)
{
    py::object ReduceOp = py::module_::import("deepspeed.comm").attr("ReduceOp");
    if (!py::isinstance(op, ReduceOp)) {
        throw std::runtime_error("Error: Op must be of type ReduceOp");
    }

    int op_val = py::int_(op.attr("value"));
    ccl::reduction ccl_op;

    if (input.scalar_type() == at::kBool) {
        if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
            // For bool tensors, map sum to max, which both represent a bitwise or.
            // This is to prevent overflow issues with sum, since we use uint8 to
            // represent a bool (see cclDataType mapping).
            ccl_op = ccl::reduction::max;
        } else if (op_val == (int)py::int_(ReduceOp.attr("AVG").attr("value"))) {
            throw std::runtime_error("Error: For bool tensors, op must be of type ReduceOp");
        }
    }

    if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
        ccl_op = ccl::reduction::sum;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MIN").attr("value"))) {
        ccl_op = ccl::reduction::min;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MAX").attr("value"))) {
        ccl_op = ccl::reduction::max;
    } else if (op_val == (int)py::int_(ReduceOp.attr("PRODUCT").attr("value"))) {
        ccl_op = ccl::reduction::prod;
    } else {
        throw std::runtime_error("Error: Unrecognized ReduceOp type");
    }
    return ccl_op;
}

void broadcast(torch::Tensor& data, int src, std::vector<int> group, bool async_op)
{
    CCLCHECK(ccl::broadcast(data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            src,
                            _get_comm_from_group(group))
                 .wait());
}

// TODO: implement torch's async_op behavior, document it.
void all_reduce(torch::Tensor& data, py::object op, std::vector<int> group, bool async_op)
{
    CCLCHECK(ccl::allreduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            get_ccl_reduce_op(op, data),
                            _get_comm_from_group(group))
                 .wait());
}

void all_reduce_caching(torch::Tensor& data,
                        py::object op,
                        std::string match_id,
                        std::vector<int> group,
                        bool async_op)
{
    ccl::allreduce_attr attr = ccl::default_allreduce_attr;
    auto match_str = ccl::v1::string(match_id);
    attr.template set<ccl::operation_attr_id::to_cache>(true);
    attr.template set<ccl::operation_attr_id::match_id>(match_str);
    // To control this, use operation attribute and set true value for to_cache field and unique
    // string (for example, tensor name) for match_id field. Note that:
    //   match_id should be the same for a specific communication operation across all ranks.
    //   If the same tensor is a part of different communication operations, match_id should have
    //   different values for each of these operations.
    CCLCHECK(ccl::allreduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            get_ccl_reduce_op(op, data),
                            _get_comm_from_group(group),
                            attr)
                 .wait());
}

void inference_all_reduce(torch::Tensor& data, py::object op)
{
#ifdef DO_PROFILE
    static double total_time = 0.0;
    static double total_time_sq = 0.0;
    static int count = -16;  // warmup
    static double max_time = 0.0;
    static double min_time = DBL_MAX;
    // make sure all rank reach this point before measuring time
    // turn on this if you suspect each rank didn't reach here at the same time (stragger)
    // if (all_ranks_local_p) {
    // barrier_wait(0, world_size);
    //}
    auto start = std::chrono::system_clock::now();
#endif

    static py::object ReduceOp = py::module_::import("deepspeed.comm").attr("ReduceOp");
    static auto ReduceOpSum = (int)py::int_(ReduceOp.attr("SUM").attr("value"));

    assert(py::int_(op.attr("value")) == ReduceOpSum);

    auto numel = data.numel();

    int data_size = 0;
    bool data_type_fallback = false;

    switch (data.scalar_type()) {
        case c10::ScalarType::BFloat16: data_size = numel * 2; break;
        case c10::ScalarType::Float: data_size = numel * 4; break;
        default: data_type_fallback = true;
    }

    if (data_type_fallback || !all_ranks_local_p) {
        // fallback to oneccl allreduce
        CCLCHECK(ccl::allreduce(data.data_ptr(),
                                data.data_ptr(),
                                data.numel(),
                                get_ccl_datatype(data.scalar_type()),
                                get_ccl_reduce_op(op, data),
                                _get_comm_from_group())
                     .wait());
    } else {
        all_reduce_outer_loop(data, numel, data_size);
    }

#ifdef DO_PROFILE
    auto end = std::chrono::system_clock::now();
    count++;
    if (count > 0) {
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (elapsed > max_time) { max_time = elapsed; }
        if (elapsed < min_time) { min_time = elapsed; }
        total_time += elapsed;
        total_time_sq += elapsed * elapsed;
        if (world_rank == 0 && count == 1000) {
            auto avg = total_time / count;
            auto sd =
                sqrt(total_time_sq / count - total_time * total_time / (count * count)) / avg * 100;
            printf("      C++ kernel\t\t    %.2f\t  %.2f\t%.2f\t      %.2f\n",
                   min_time,
                   max_time,
                   total_time / count,
                   sd);
        }
    }
#endif
}

void barrier(std::vector<int> group, bool async_op)
{
    CCLCHECK(ccl::barrier(_get_comm_from_group(group)).wait());
}

std::vector<std::string> get_available_coll()
{
    std::vector<std::string> colls{
        "broadcast", "all_reduce", "inference_all_reduce", "all_reduce_caching", "barrier"};
    return colls;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_kvs_addr", &get_kvs_addr, "create and get main kvs addr");
    m.def("initialize", &initialize, "ccl initialize");
    m.def("get_rank", &get_rank, "get rank");
    m.def("get_world_size", &get_world_size, "get world size");
    m.def("broadcast", &broadcast, "ccl broadcast");
    m.def("all_reduce", &all_reduce, "ccl all_reduce");
    m.def("inference_all_reduce", &inference_all_reduce, "low latency all_reduce implementation");
    m.def("all_reduce_caching", &all_reduce_caching, "ccl all_reduce with caching");
    m.def("barrier", &barrier, "barrier");
    m.def("initialize_sub_comm", &initialize_sub_comm, "initialize_sub_comm");
    m.def("get_sub_kvs_addr", &get_sub_kvs_addr, "get_sub_kvs_addr");
    m.def("get_available_coll", &get_available_coll, "get_available_coll");
}
