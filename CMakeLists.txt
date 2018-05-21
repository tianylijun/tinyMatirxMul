cmake_minimum_required(VERSION 3.5)
PROJECT(libtinyMatrixMul)

message(CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR})
message(CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR})
message(PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR})

option(ENABLE_OPENMP "openmp support" ON)

ENABLE_LANGUAGE(ASM)

file(GLOB LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB LIB_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
file(GLOB LIB_SSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/arm64/*.S)
message(STATUS "ARCH: ARM64")
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
file(GLOB LIB_SSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/arm32/*.S)
message(STATUS "ARCH: ARM32")
endif()

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-function -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

message(STATUS "C flags: ${CMAKE_C_FLAGS}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

ADD_LIBRARY(tinyMatrixMul SHARED ${LIB_CSRC} ${LIB_SSRC})
ADD_LIBRARY(tinyMatrixMul_static STATIC ${LIB_CSRC} ${LIB_SSRC})

SET_TARGET_PROPERTIES(tinyMatrixMul PROPERTIES VERSION 1.2 SOVERSION 1)
SET_TARGET_PROPERTIES(tinyMatrixMul_static PROPERTIES OUTPUT_NAME "tinyMatrixMul")

set(FEATHER_INSTALL_DIR "${PROJECT_BINARY_DIR}/install")

message(CMAKE_INSTALL_PREFIX: ${FEATHER_INSTALL_DIR})
message(Library headers: ${LIB_HEADERS})

install(TARGETS tinyMatrixMul tinyMatrixMul_static 
LIBRARY DESTINATION "${FEATHER_INSTALL_DIR}/lib"
ARCHIVE DESTINATION "${FEATHER_INSTALL_DIR}/libstatic"
)
install(FILES ${LIB_HEADERS} DESTINATION "${FEATHER_INSTALL_DIR}/include")
