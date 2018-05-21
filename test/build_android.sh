#!/bin/bash

mkdir -p build-android
pushd build-android
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-22 ..
make clean && make && cp sgemmtest /media/psf/Home/nfs && cp ntcopytest /media/psf/Home/nfs
popd
