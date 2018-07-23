cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME FHOG_Demo)
project(${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src VPP_SRC)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(${PROJECT_NAME} ${VPP_SRC})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()
