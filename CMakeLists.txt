cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolo-app)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Torch REQUIRED)

aux_source_directory(. DIR_SRCS)

add_executable(yolo-app ${DIR_SRCS})
target_link_libraries(yolo-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET yolo-app PROPERTY CXX_STANDARD 11)
