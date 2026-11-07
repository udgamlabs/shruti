cmake_minimum_required(VERSION 3.16)
project(shruti_transcribe CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# get_sherpa.sh unpacks sherpa-onnx 2.0.1 here. Or: cmake -DSHERPA_DIR=/your/copy ..
set(SHERPA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx" CACHE PATH "unpacked sherpa-onnx release")

add_executable(transcribe transcribe.cc)
target_include_directories(transcribe PRIVATE "${SHERPA_DIR}/include")
target_link_directories(transcribe PRIVATE "${SHERPA_DIR}/lib")
target_link_libraries(transcribe PRIVATE sherpa-onnx-cxx-api sherpa-onnx-c-api)
set_target_properties(transcribe PROPERTIES BUILD_RPATH "${SHERPA_DIR}/lib")
