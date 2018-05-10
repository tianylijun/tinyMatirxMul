cmake_minimum_required(VERSION 3.5)

set(LIB_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a/install")
message(LIB_INSTALL_DIR: ${LIB_INSTALL_DIR})

include_directories(${LIB_INSTALL_DIR}/include)
link_directories(${LIB_INSTALL_DIR}/libstatic)

add_executable(sgemmtest sgemm-test.cpp)
target_link_libraries(sgemmtest tinyMatrixMul)
