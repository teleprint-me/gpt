cmake_minimum_required(VERSION 3.17)
project("gpt" LANGUAGES CXX C)
set(CMAKE_C_STANDARD 20)
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

add_executable(tokenizer unicode-data.cpp unicode.cpp tokenizer.cpp)
add_executable(model model.cpp)
