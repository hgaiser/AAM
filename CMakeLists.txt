cmake_minimum_required(VERSION 2.8.3)
project(AAM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(TinyXML REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

if (CMAKE_COMPILER_IS_GNUCXX)
	message("Detected g++, using OpenMP.")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fopenmp")
endif()

include_directories(include 
  ${TINYXML_INCLUDE_DIRS}
)

add_executable(AAM src/AAM.cpp src/DetectFace.cpp src/MatlabIO.cpp)

target_link_libraries(AAM ${OpenCV_LIBS} ${TINYXML_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

