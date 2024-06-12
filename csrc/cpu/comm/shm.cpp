// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <fcntl.h>
#include <immintrin.h>
#include <semaphore.h>
#include <sys/mman.h>
#include "shm.h"

// #define DO_PROFILE
#ifdef DO_PROFILE
#include <cfloat>
#include <chrono>
#endif

// states for collectives
enum coll_state {
    coll_begin = 0,
    coll_allreduce_naive__copy_in_done,
    coll_allreduce_naive__reduce_done,
    // alternative state when allreduce is working on alternative buffer
    // of the double buffer.
    coll_alt1_allreduce_naive__copy_in_done,
    coll_alt2_allreduce_naive__copy_in_done,
    coll_alt1_allreduce_naive__reduce_done,
};

// SHM building blocks
struct SharedData {
    const char* name;
    int descriptor;
    void* bytes;
    size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes)
{
    int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
        data->name = name;
        data->descriptor = d;
        data->bytes = bytes;
        data->nbytes = nbytes;
    } else {
        if (errno != ENOENT) {
            // don't print if shm can not be found because we want to loop over from
            // caller again until the other ranks created the shm
            printf("shared_open %s failed, errno=%d\n", name, errno);
        }
        data->descriptor = -1;
    }
}

void shared_create(SharedData* data, const char* name, void* bytes, size_t nbytes)
{
    int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        if (nbytes = write(d, bytes, nbytes)) { shared_open(data, name, nbytes); }
    } else {
        printf("shared_create %s failed\n", name);
    }
}

void shared_close(SharedData* data)
{
    if (data->descriptor != -1) {
        munmap(data->bytes, data->nbytes);
        shm_unlink(data->name);
    }
}

static int world_size;

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
    enum coll_state states[2];  // idx=0 -- state for symmetric_naive_all_reduce
                                // idx=1 -- state for distributed_naive_all_reduce
    // double buffer to avoid syncing between rounds
    // offset=0 -- 2*NAIVE_ALLREDUCE_THRESHOLD : buffer for symmetric_naive_all_reduce
    // after that : buffer for distributed_naive_all_reduce
    char buffer[2 * NAIVE_ALLREDUCE_THRESHOLD + 2 * MAX_BUF_SIZE];
};

#define BUFFER0_OFFSET(current_buffer) current_buffer* NAIVE_ALLREDUCE_THRESHOLD
#define BUFFER1_OFFSET(current_buffer) 2 * NAIVE_ALLREDUCE_THRESHOLD + current_buffer* MAX_BUF_SIZE

struct allreduce_workspace** workspace;

// buffer for small messages, double buffer
char** symmetric_buffer[2];
// buffer for large messages, double buffer
char** distributed_buffer[2];

void wait_buffer_state_until_2(int index,
                               enum coll_state state0,
                               enum coll_state state1,
                               int state_group)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->states[state_group]);

    while (1) {
        volatile enum coll_state cur_state = *state_ptr;
        if (cur_state == state0 || cur_state == state1) break;
    }
}

__m512 cvt_bf16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_bf16_to_fp32(const __m256i src)
{
    auto y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_bf16(const __m512 src)
{
    __m512i value = _mm512_castps_si512(src);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    // uint32_t lsb = (input >> 16) & 1;
    auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
    // uint32_t rounding_bias = 0x7fff + lsb;
    t_value = _mm512_add_epi32(t_value, vec_bias);
    // input += rounding_bias;
    t_value = _mm512_add_epi32(t_value, value);
    // input = input >> 16;
    t_value = _mm512_srli_epi32(t_value, 16);
    // Check NaN before converting back to bf16
    t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
    return _mm512_cvtusepi32_epi16(t_value);
}

void reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));

void reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));

void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
    __attribute__((target("avx512bw")));

void reduce_all_buffers(int start_elements,
                        int num_elements,
                        c10::ScalarType scalar_type,
                        int to_buffer_idx,
                        char* to_buffer,
                        char** buffers)
{
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            if (world_size == 2) {
                // add the other buffer to to_buffer
                reduce_2_bf16_buffers_iio(num_elements,
                                          buffers[1 - to_buffer_idx] + start_elements * 2,
                                          to_buffer + start_elements * 2,
                                          to_buffer + start_elements * 2);
            } else {
                reduce_bf16_buffers(start_elements, num_elements, to_buffer, buffers);
            }
            break;
        case c10::ScalarType::Float:
            if (world_size == 2) {
                reduce_2_fp32_buffers_iio(num_elements,
                                          buffers[1 - to_buffer_idx] + start_elements * 4,
                                          to_buffer + start_elements * 4,
                                          to_buffer + start_elements * 4);
            } else {
                assert(world_size > 2);
                reduce_fp32_buffers(start_elements, num_elements, to_buffer, buffers);
            }
            break;
        default: assert(!"Should not get here");
    }
}

#define CVT_ADD_BF16(x)                                                                      \
    do {                                                                                     \
        auto in##x##_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[x] + i))); \
        inout_val = _mm512_add_ps(inout_val, in##x##_val);                                   \
    } while (0)

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  256bit vector ==> 32 bytes, 512bit vector ==> 64 bytes
// If you change implementation of reduce_2_bf16_buffers_iio or reduce_2_fp32_buffers_iio, check
// whether this number needs to be changed
#define VECTOR_LENGTH_IN_BYTES 32

void reduce_bf16_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[0] + i)));
        switch (world_size) {
            case 16: CVT_ADD_BF16(15);
            case 15: CVT_ADD_BF16(14);
            case 14: CVT_ADD_BF16(13);
            case 13: CVT_ADD_BF16(12);
            case 12: CVT_ADD_BF16(11);
            case 11: CVT_ADD_BF16(10);
            case 10: CVT_ADD_BF16(9);
            case 9: CVT_ADD_BF16(8);
            case 8: CVT_ADD_BF16(7);
            case 7: CVT_ADD_BF16(6);
            case 6: CVT_ADD_BF16(5);
            case 5: CVT_ADD_BF16(4);
            case 4: CVT_ADD_BF16(3);
            case 3:
                CVT_ADD_BF16(2);
                CVT_ADD_BF16(1);
                break;
            default:
                for (int j = 1; j < world_size; j++) {
                    auto in_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[j] + i)));
                    inout_val = _mm512_add_ps(inout_val, in_val);
                }
        }
        _mm256_storeu_si256((__m256i*)(to_buffer + i), cvt_fp32_to_bf16(inout_val));
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < world_size; j++) { val += *(at::BFloat16*)(buffers[j] + i); }
        *(at::BFloat16*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

void reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < main_elements * element_size; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in0 + i)));
        auto in1_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in1 + i)));
        auto out_val = _mm512_add_ps(in0_val, in1_val);
        _mm256_storeu_si256((__m256i*)((char*)out + i), cvt_fp32_to_bf16(out_val));
    }

    // process remaining part
    int i = main_elements * element_size;
    while (remain_elements > 0) {
        float in0_val = *((at::BFloat16*)((char*)in0 + i));
        float in1_val = *((at::BFloat16*)((char*)in1 + i));
        *((at::BFloat16*)((char*)out + i)) = in0_val + in1_val;
        remain_elements--;
        i += element_size;
    }
}

#define CVT_ADD_F32(x)                                                \
    do {                                                              \
        auto in##x##_val = _mm256_loadu_ps((float*)(buffers[x] + i)); \
        inout_val = _mm256_add_ps(inout_val, in##x##_val);            \
    } while (0)

void reduce_fp32_buffers(int start_elements, int num_elements, char* to_buffer, char** buffers)
{
    const int element_size = 4;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)(buffers[0] + i));
        switch (world_size) {
            case 16: CVT_ADD_F32(15);
            case 15: CVT_ADD_F32(14);
            case 14: CVT_ADD_F32(13);
            case 13: CVT_ADD_F32(12);
            case 12: CVT_ADD_F32(11);
            case 11: CVT_ADD_F32(10);
            case 10: CVT_ADD_F32(9);
            case 9: CVT_ADD_F32(8);
            case 8: CVT_ADD_F32(7);
            case 7: CVT_ADD_F32(6);
            case 6: CVT_ADD_F32(5);
            case 5: CVT_ADD_F32(4);
            case 4: CVT_ADD_F32(3);
            case 3:
                CVT_ADD_F32(2);
                CVT_ADD_F32(1);
                break;
            default:
                for (int j = 1; j < world_size; j++) {
                    auto in_val = _mm256_loadu_ps((float*)(buffers[j] + i));
                    inout_val = _mm256_add_ps(inout_val, in_val);
                }
        }
        _mm256_storeu_ps((float*)(to_buffer + i), inout_val);
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < world_size; j++) { val += *(float*)(buffers[j] + i); }
        *(float*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

void reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
    const int element_size = 4;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < main_elements * element_size; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = _mm256_loadu_ps((float*)((char*)in0 + i));
        auto in1_val = _mm256_loadu_ps((float*)((char*)in1 + i));
        auto out_val = _mm256_add_ps(in0_val, in1_val);
        _mm256_storeu_ps((float*)((char*)out + i), out_val);
    }

    // process remaining part
    int i = main_elements * element_size;
    while (remain_elements > 0) {
        float in0_val = *((float*)((char*)in0 + i));
        float in1_val = *((float*)((char*)in1 + i));
        *((float*)((char*)out + i)) = in0_val + in1_val;
        remain_elements--;
        i += element_size;
    }
}

static bool is_initialized = 0;
static int world_rank;

void shm_initialize(int size, int rank, char* addr_string, char* port_string)
{
    if (is_initialized) return;
    is_initialized = 1;

    world_size = size;
    world_rank = rank;

    char shm_name_prefix[NAME_BUF_SIZE];
    char shm_name[NAME_BUF_SIZE];
    snprintf(shm_name_prefix,
             NAME_BUF_SIZE,
             "%s_%d_%s_%s",
             SHM_BUFFER_NAME,
             getuid(),
             addr_string,
             port_string);
    // create shared workspace for SHM based allreduce
    SharedData allreduce_buffer;
    // allocate workspace_buf for current rank
    struct allreduce_workspace* workspace_buf;
    struct allreduce_workspace* workspace_buf_other;
    workspace_buf = (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
    snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, rank);
    shared_create(&allreduce_buffer, shm_name, workspace_buf, sizeof(struct allreduce_workspace));
    workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
    workspace_buf->states[0] = coll_alt2_allreduce_naive__copy_in_done;
    workspace_buf->states[1] = coll_begin;

    // create the workspace pointer list
    workspace = (struct allreduce_workspace**)malloc(size * sizeof(struct allreduce_workspace*));
    symmetric_buffer[0] = (char**)malloc(size * sizeof(char**));
    symmetric_buffer[1] = (char**)malloc(size * sizeof(char**));
    distributed_buffer[0] = (char**)malloc(size * sizeof(char**));
    distributed_buffer[1] = (char**)malloc(size * sizeof(char**));

    // map shm of all ranks
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, i);
            // printf("open %s, %d\n", shm_name, rank);
            do {
                shared_open(&allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
            } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
            workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
            workspace[i] = workspace_buf_other;
        } else {
            workspace[i] = workspace_buf;
        }
        symmetric_buffer[0][i] = workspace[i]->buffer + BUFFER0_OFFSET(0);
        symmetric_buffer[1][i] = workspace[i]->buffer + BUFFER0_OFFSET(1);
        distributed_buffer[0][i] = workspace[i]->buffer + BUFFER1_OFFSET(0);
        distributed_buffer[1][i] = workspace[i]->buffer + BUFFER1_OFFSET(1);
    }
}

static void parallel_memcpy(void* to, void* from, size_t n_bytes)
    __attribute__((target("avx512bw")));
static void parallel_memcpy(void* to, void* from, size_t n_bytes)
{
    auto aligned_bytes = n_bytes - (n_bytes % VECTOR_LENGTH_IN_BYTES);
    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < aligned_bytes; i += VECTOR_LENGTH_IN_BYTES) {
        auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
        _mm256_storeu_si256((__m256i*)((char*)to + i), val);
    }

    // process remaining part
    for (int i = aligned_bytes; i < n_bytes; i++) { *((char*)to + i) = *((char*)from + i); }
}

#define positive_mod(num, mod) ((((num) % (mod)) + (mod)) % (mod))
#define rank_mod(rank) positive_mod(rank, world_size)
size_t slice_size(size_t chunk_el, int slice_idx)
{
    size_t slice_size = chunk_el / world_size;
    return slice_idx == world_size - 1 ? slice_size + (chunk_el % world_size) : slice_size;
}

char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx)
{
    size_t slice_size = chunk_el / world_size;
    size_t el_offset = slice_size * slice_idx;
    return data_ptr + el_offset * el_size;
}

size_t slice_el_start(size_t chunk_el, int slice_idx)
{
    size_t slice_size = chunk_el / world_size;
    return slice_size * slice_idx;
}

/*
    Symmetrical naive all_reduce
    step 0: before enter the function ith times, state is copy(i-1)
    step 1: each rank copy data from input (data_ptr) to SHM buffer[i]
    step 2: set own state to copy(i)
    step 3: wait each other rank's state equal or later than copy(i)
    step 4: reduce across SHM buffer(ith) directly into output (data_ptr)
*/
void symmetric_naive_all_reduce(char* data_ptr,
                                c10::ScalarType scalar_type,
                                size_t chunk_size,
                                size_t chunk_el)
{
#ifdef DO_PROFILE
    static double total_t1_t0 = 0.0;
    static double total_t2_t1 = 0.0;
    static double total_t3_t2 = 0.0;
    static int count = -16;  // warmup
    auto t0 = std::chrono::system_clock::now();
#endif

    /*
        We can't have infinite number of buffers and states.  2 sets of buffer
        and 3 sets of states is just enough.  Consider current rank is in step 3,
        with it's own state set to copy(i), the other rank will them have the
        following situations:
        ------------------------------------------------
        my state | can I proceed? | the other rank state
        ================================================
                 |       N        | copy(i-1)
                 |----------------|---------------------
        copy(i)  |       Y        | copy(i)
                 |----------------|---------------------
                 |       Y        | copy(i+1)
        ------------------------------------------------
        * When I have state as copy(i), the other rank cannot have state
          copy(i-2) or before. In that case I'll be in state copy(i-1) and cannot
          proceed to copy(i).
        * The other rank cannot have state copy(i+2) or beyond because my
          state is still copy(i), copy(i+1) is as far as the other rank could go.
        * From a rank's POV, all the other ranks can be divided into three sets:
          - Lagging ranks: ranks that are still working on previous iteration
          - Syncing ranks: ranks that are working on current iteration
          - Leading ranks: ranks that are working on next iteration
        * We can have 3 sets of states, one set for syncing ranks; one set for
          lagging ranks; one set of leading ranks.  With 3 sets of states, we can
          distinguish between lagging and leading ranks.
        * Note from any rank's POV, leading ranks and lagging ranks does not
          appear at the same time.  Either all other ranks are syncing or
          lagging, or all other ranks are syncing or leading.  Otherwise leading
          and lagging ranks will be 2 iterations apart and this should not happen.
        * So we have 2 sets of buffers, one buffer is used by current iter;
          one buffer used by either lagging ranks or leading ranks.
    */
    const int state_group = 0;
    static int current_buffer = 0;
    static int state_idx = 0;

    enum coll_state copy_current, copy_next;

    switch (state_idx) {
        case 0:
            copy_current = coll_allreduce_naive__copy_in_done;
            copy_next = coll_alt1_allreduce_naive__copy_in_done;
            break;
        case 1:
            copy_current = coll_alt1_allreduce_naive__copy_in_done;
            copy_next = coll_alt2_allreduce_naive__copy_in_done;
            break;
        case 2:
            copy_current = coll_alt2_allreduce_naive__copy_in_done;
            copy_next = coll_allreduce_naive__copy_in_done;
            break;
        default: assert(!"Should not get here.");
    }
    state_idx = (state_idx + 1) % 3;

    parallel_memcpy(symmetric_buffer[current_buffer][world_rank], data_ptr, chunk_size);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->states[state_group] = copy_current;

#ifdef DO_PROFILE
    auto t1 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < world_size; i++) {
        // wait until the other rank copy the buffer
        if (i != world_rank) { wait_buffer_state_until_2(i, copy_current, copy_next, state_group); }
    }
#ifdef DO_PROFILE
    auto t2 = std::chrono::system_clock::now();
#endif

    // each rank reduce the buffer independently so therre is no need for synchronization afterward
    reduce_all_buffers(
        0, chunk_el, scalar_type, world_rank, data_ptr, symmetric_buffer[current_buffer]);

    // switch buffer
    current_buffer = 1 - current_buffer;

#ifdef DO_PROFILE
    auto t3 = std::chrono::system_clock::now();

    count++;
    if (count > 0) {
        total_t1_t0 += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_t2_t1 += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        total_t3_t2 += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        if (world_rank == 0 && count == 1000) {
            printf("symmetric_naive_all_reduce time breakdown:\n");
            printf("\tcopy input buffer: %.2f\n", total_t1_t0 / count);
            printf("\twait for copy: %.2f\n", total_t2_t1 / count);
            printf("\treduce: %.2f\n", total_t3_t2 / count);
        }
    }
#endif
}

// naive allreduce distributed, each rank do naive reduce on its slice
void distributed_naive_reduce(char* data_ptr,
                              c10::ScalarType scalar_type,
                              size_t chunk_size,
                              size_t chunk_el)
{
#ifdef DO_PROFILE
    static double total_t1_t0 = 0.0;
    static double total_t2_t1 = 0.0;
    static double total_t3_t2 = 0.0;
    static double total_t4_t3 = 0.0;
    static double total_t5_t4 = 0.0;
    static int count = -16;  // warmup
    auto t0 = std::chrono::system_clock::now();
#endif

    const int state_group = 1;
    static int current_buffer = 0;
    static int state_idx = 0;

    enum coll_state copy_current, copy_next, reduce_current;

    // similar to symmetric_naive_allreduce, but here we only need two sets of
    // states, because distributed naive reduce has two barriers in the algorithm
    switch (state_idx) {
        case 0:
            copy_current = coll_allreduce_naive__copy_in_done;
            reduce_current = coll_allreduce_naive__reduce_done;
            copy_next = coll_alt1_allreduce_naive__copy_in_done;
            break;
        case 1:
            copy_current = coll_alt1_allreduce_naive__copy_in_done;
            reduce_current = coll_alt1_allreduce_naive__reduce_done;
            copy_next = coll_allreduce_naive__copy_in_done;
            break;
        default: assert(!"Should not get here.");
    }
    state_idx = (state_idx + 1) % 2;

    int data_size = chunk_size / chunk_el;
    parallel_memcpy(distributed_buffer[current_buffer][world_rank], data_ptr, chunk_size);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->states[state_group] = copy_current;

#ifdef DO_PROFILE
    auto t1 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < world_size; i++) {
        // wait until all the other ranks copy the buffer
        if (i != world_rank)
            wait_buffer_state_until_2(i, copy_current, reduce_current, state_group);
    }

#ifdef DO_PROFILE
    auto t2 = std::chrono::system_clock::now();
#endif

    // reduce scatter
    reduce_all_buffers(slice_el_start(chunk_el, world_rank),
                       slice_size(chunk_el, world_rank),
                       scalar_type,
                       world_rank,
                       distributed_buffer[current_buffer][world_rank],
                       distributed_buffer[current_buffer]);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->states[state_group] = reduce_current;

#ifdef DO_PROFILE
    auto t3 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < world_size; i++) {
        // wait until all the other ranks reduce the buffer
        if (i != world_rank) wait_buffer_state_until_2(i, reduce_current, copy_next, state_group);
    }

    auto t4 = std::chrono::system_clock::now();

    for (int i = 0; i < world_size; i++) {
        int rank = (i + world_rank) % world_size;
        parallel_memcpy(
            slice_data(data_ptr, chunk_el, data_size, rank),
            slice_data(
                distributed_buffer[current_buffer][rank], chunk_el, chunk_size / chunk_el, rank),
            slice_size(chunk_el, rank) * data_size);
    }

    current_buffer = 1 - current_buffer;

#ifdef DO_PROFILE
    auto t5 = std::chrono::system_clock::now();
    count++;
    if (count > 0) {
        total_t1_t0 += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_t2_t1 += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        total_t3_t2 += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        total_t4_t3 += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        total_t5_t4 += std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
        if (world_rank == 0 && count == 1000) {
            printf("distributed_naive_reduce time breakdown:\n");
            printf("\tcopy input buffer: %.2f\n", total_t1_t0 / count);
            printf("\twait for copy: %.2f\n", total_t2_t1 / count);
            printf("\treduce: %.2f\n", total_t3_t2 / count);
            printf("\twait for reduce finish: %.2f\n", total_t4_t3 / count);
            printf("\tcopy out: %.2f\n", total_t5_t4 / count);
        }
    }
#endif
}

void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size)
{
    for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
        auto data_ptr = ((char*)(data.data_ptr()) + offset);
        size_t chunk_size = data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
        size_t chunk_el = chunk_size / (data_size / numel);
        if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD)
            symmetric_naive_all_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
        else
            distributed_naive_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
    }
}
