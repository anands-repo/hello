cmake_minimum_required(VERSION 3.12)
project(libCallability.so)

#### Correct these paths (platform specific) ####
#set(BOOST /opt/boost/1.68.0)   # Set boost if non-standard installation path
#include_directories(${BOOST}/include)   # Include boost directory if not standard installation path
set(PYTHON_INC /root/miniconda3/include/python3.7m)
#################################################

include_directories(${PYTHON_INC})
include_directories(c++/include)
link_directories(${BOOST}/lib)
set(CMAKE_CXX_STANDARD 14)
add_definitions(-DBOOST_LOG_DYN_LINK)
add_definitions(-mavx2)
add_definitions(-p)
add_definitions(-g)
add_library(Callability SHARED main.cpp c++/src/Trie.cpp c++/src/Reference.cpp c++/src/Read.cpp c++/src/utils.cpp c++/src/leftAlignCigars.cpp c++/src/AlleleSearcherLiteFiltered.cpp)
target_link_libraries(Callability -lboost_python37 -lboost_numpy37 -lboost_log -lpthread)
