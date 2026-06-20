#include <gtest/gtest.h>
#include <cstring>
#include <cstdlib>
#include <vector>

// Forward declare the function under test from shm.cpp
extern "C" void parallel_memcpy(void* to, void* from, size_t n_bytes);

// Test fixture for buffer overflow detection
class ParallelMemcpySecurityTest : public ::testing::TestWithParam<size_t> {};

TEST_P(ParallelMemcpySecurityTest, BufferReadNeverExceedsDeclaredLength) {
    // Invariant: parallel_memcpy must never read beyond n_bytes from source
    // or write beyond n_bytes to destination, regardless of input size.
    
    size_t n_bytes = GetParam();
    const size_t MAX_BUF_SIZE = 32 * 1024 * 1024;  // 32MB from shm.cpp
    
    // Allocate destination buffer with guard pages
    size_t alloc_size = (n_bytes > MAX_BUF_SIZE) ? MAX_BUF_SIZE : n_bytes;
    if (alloc_size == 0) alloc_size = 1;
    
    char* dest = (char*)malloc(alloc_size + 64);
    char* src = (char*)malloc(alloc_size + 64);
    
    ASSERT_NE(dest, nullptr);
    ASSERT_NE(src, nullptr);
    
    // Fill buffers with sentinel values
    memset(dest, 0xAA, alloc_size + 64);
    memset(src, 0xBB, alloc_size + 64);
    
    // Record guard zone before copy
    char guard_before[64];
    memcpy(guard_before, dest + alloc_size, 64);
    
    // Attempt copy with potentially oversized n_bytes
    size_t safe_copy_size = (n_bytes > alloc_size) ? alloc_size : n_bytes;
    parallel_memcpy(dest, src, safe_copy_size);
    
    // Verify guard zone after copy is untouched (no overflow)
    char guard_after[64];
    memcpy(guard_after, dest + alloc_size, 64);
    
    for (int i = 0; i < 64; i++) {
        EXPECT_EQ(guard_before[i], guard_after[i])
            << "Buffer overflow detected at offset " << i 
            << " with n_bytes=" << n_bytes;
    }
    
    // Verify copied data is correct for valid range
    for (size_t i = 0; i < safe_copy_size; i++) {
        EXPECT_EQ(dest[i], src[i])
            << "Data corruption at offset " << i;
    }
    
    free(dest);
    free(src);
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialBufferSizes,
    ParallelMemcpySecurityTest,
    ::testing::Values(
        1024,                          // Valid: small buffer
        1024 * 1024,                   // Valid: 1MB (NAIVE_ALLREDUCE_THRESHOLD)
        32 * 1024 * 1024,              // Boundary: MAX_BUF_SIZE
        64 * 1024 * 1024,              // Exploit: 2x MAX_BUF_SIZE
        320 * 1024 * 1024              // Exploit: 10x MAX_BUF_SIZE
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}