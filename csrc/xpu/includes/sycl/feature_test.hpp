// Wrapper: include real feature_test.hpp then disable BF16 math functions
// to avoid dependency on __devicelib_ConvertBF16ToFINTEL.
#include_next <sycl/feature_test.hpp>
#undef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
