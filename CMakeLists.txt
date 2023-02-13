cmake_minimum_required(VERSION 2.8)
project(test_project)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)



find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})



add_executable(test main.cpp model.cpp)
target_link_libraries( test PRIVATE ${OpenCV_LIBS} )
target_link_libraries(test PRIVATE openvino::runtime)



