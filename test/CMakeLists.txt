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

# Windows 优先从程序目录加载 DLL，避免 PATH 中其他 MinGW 版本的运行库冲突。
if(MINGW)
    get_filename_component(mini_llama_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(mini_llama_test_runtime_dirs "${CMAKE_CURRENT_BINARY_DIR}")
    foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
        list(APPEND mini_llama_test_runtime_dirs "${CMAKE_CURRENT_BINARY_DIR}/${config}")
    endforeach()
    foreach(runtime IN ITEMS libstdc++-6.dll libwinpthread-1.dll
                             libgcc_s_seh-1.dll libgcc_s_sjlj-1.dll libgcc_s_dw2-1.dll)
        if(EXISTS "${mini_llama_mingw_bin}/${runtime}")
            foreach(runtime_dir IN LISTS mini_llama_test_runtime_dirs)
                file(COPY "${mini_llama_mingw_bin}/${runtime}" DESTINATION "${runtime_dir}")
            endforeach()
        endif()
    endforeach()
endif()

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

# 与 KV cache 一致，先在测试目标中编译尚未加入静态库的实现。
add_executable(batch_tests test_batch.cpp "${PROJECT_SOURCE_DIR}/src/batch.cpp")
target_link_libraries(batch_tests PRIVATE mini_llama GTest::gtest_main)

add_executable(request_context_tests test_request_context.cpp
               "${PROJECT_SOURCE_DIR}/src/request_context.cpp")
target_link_libraries(request_context_tests PRIVATE mini_llama GTest::gtest_main)

add_executable(context_tests test_context.cpp "${PROJECT_SOURCE_DIR}/src/kv_cache.cpp")
target_link_libraries(context_tests PRIVATE mini_llama GTest::gtest_main)

add_executable(forward_tests test_forward.cpp "${PROJECT_SOURCE_DIR}/src/kv_cache.cpp")
target_link_libraries(forward_tests PRIVATE mini_llama GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tensor_tests PROPERTIES TIMEOUT 10)
gtest_discover_tests(ops_tests PROPERTIES TIMEOUT 10 LABELS ops)
gtest_discover_tests(tokenizer_tests PROPERTIES TIMEOUT 10 LABELS tokenizer)
gtest_discover_tests(kv_cache_tests PROPERTIES TIMEOUT 10 LABELS kv_cache)
gtest_discover_tests(radix_tree_tests PROPERTIES TIMEOUT 10 LABELS radix_tree)
gtest_discover_tests(batch_tests PROPERTIES TIMEOUT 10 LABELS batch)
gtest_discover_tests(request_context_tests PROPERTIES TIMEOUT 10 LABELS request_context)
gtest_discover_tests(context_tests PROPERTIES TIMEOUT 10 LABELS context)
gtest_discover_tests(forward_tests PROPERTIES TIMEOUT 10 LABELS forward)
