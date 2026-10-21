include(FetchContent)

# 固定 GoogleTest v1.18.0，浅克隆并复用本地依赖缓存。
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "Use the parent project's Windows runtime" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tensor_tests test_tensor.cpp)
target_link_libraries(tensor_tests PRIVATE mini_llama GTest::gtest_main)

add_executable(ops_tests test_ops.cpp)
target_link_libraries(ops_tests PRIVATE mini_llama GTest::gtest_main)

add_executable(tokenizer_tests test_tokenizer.cpp)
target_link_libraries(tokenizer_tests PRIVATE mini_llama GTest::gtest_main)

# KV cache 尚未加入静态库，仅在测试目标中编译现有实现。
add_executable(kv_cache_tests test_kv_cache.cpp "${PROJECT_SOURCE_DIR}/src/kv_cache.cpp")
target_link_libraries(kv_cache_tests PRIVATE mini_llama GTest::gtest_main)

add_executable(radix_tree_tests test_radix_tree.cpp)
target_link_libraries(radix_tree_tests PRIVATE mini_llama GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tensor_tests PROPERTIES TIMEOUT 10)
gtest_discover_tests(ops_tests PROPERTIES TIMEOUT 10 LABELS ops)
gtest_discover_tests(tokenizer_tests PROPERTIES TIMEOUT 10 LABELS tokenizer)
gtest_discover_tests(kv_cache_tests PROPERTIES TIMEOUT 10 LABELS kv_cache)
gtest_discover_tests(radix_tree_tests PROPERTIES TIMEOUT 10 LABELS radix_tree)
