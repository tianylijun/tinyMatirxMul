cmake_minimum_required(VERSION 3.5)

set(LIB_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a/install")
message(LIB_INSTALL_DIR: ${LIB_INSTALL_DIR})

set(CMAKE_BUILD_TYPE Release)

option(ENABLE_OPENMP "openmp support" ON)

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    message(STATUS "OpenMP flags ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${LIB_INSTALL_DIR}/include)
link_directories(${LIB_INSTALL_DIR}/libstatic)

add_executable(sgemmtest sgemm-test.cpp)
target_link_libraries(sgemmtest tinyMatrixMul)

add_executable(ntcopytest ntcopy-test.cpp)
target_link_libraries(ntcopytest tinyMatrixMul)
