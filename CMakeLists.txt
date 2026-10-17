cmake_minimum_required(VERSION 3.16)

project(mini_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mini_llama STATIC
    src/tensor.cpp
    src/ops.cpp
    src/tokenizer/ascii_tokenizer.cpp
    src/tokenizer/json_vocab_tokenizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
)

target_include_directories(mini_llama PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)

if(MSVC)
    target_compile_options(mini_llama PUBLIC /utf-8)
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
