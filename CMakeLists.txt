cmake_minimum_required(VERSION 3.16)

project(graph_paper_text_binarization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(graph_binarization
    src/binarization.cpp
)

target_include_directories(graph_binarization
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(graph_binarization
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(demo
    apps/demo.cpp
)

target_link_libraries(demo
    PRIVATE
        graph_binarization
)
