cmake_minimum_required(VERSION 3.15)
project(ProjectName)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# 启用CUDA语言支持
enable_language(CUDA)

find_package(TensorRT REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres 2 REQUIRED)
find_package(fmt REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

set(SOURCE_FILES 
    src/main.cpp
    src/TRTModule.cpp
    src/ThreadPool.cpp
    # 在这里添加其他的源文件
)

# 定义一个可执行文件目标
add_executable(ProjectName ${SOURCE_FILES})

# 链接所需的库
target_link_libraries(ProjectName ${OpenCV_LIBS} ${TensorRT_LIBS} ${CERES_LIBRARIES} fmt::fmt nvonnxparser cuda)

