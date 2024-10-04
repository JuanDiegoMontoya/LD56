cmake_minimum_required(VERSION 3.15)

project(LD56)

set(CMAKE_CXX_STANDARD 20)

set(LD56_source_files
	src/main.cpp
	external/miniaudio.cpp
	external/tiny_obj_loader.cpp
)

add_executable(LD56_game
	${LD56_source_files}
)

# Determine whether we're compiling with clang++
string(FIND "${CMAKE_CXX_COMPILER}" "clang++" COMPILER_CLANGPP)
if(COMPILER_CLANGPP GREATER -1)
    set(COMPILER_CLANGPP 1)
else()
    set(COMPILER_CLANGPP 0)
endif()

# enable roughly equivalent (and strict!) compiler warnings depending on the compiler
target_compile_options(LD56_game PUBLIC
	$<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>,${COMPILER_CLANGPP}>:
	-Wall
	-Wextra
	-pedantic-errors
	-Wno-missing-field-initializers
	-Wno-unused-result
	>
	$<$<CXX_COMPILER_ID:MSVC>:
	-W4
	-WX
	-permissive-
	-wd4324 # disable warning indicating that a structure was padded
	>
)

find_package(OpenGL REQUIRED)

# enable asan for debug builds
if (DEBUG)
    target_compile_options(LD56_game PUBLIC -fsanitize=address)
endif()

add_subdirectory(external)

target_include_directories(LD56_game PUBLIC	src	external)

target_link_libraries(LD56_game glm EnTT::EnTT fwog glfw lib_glad imgui)

#add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/assets ${CMAKE_CURRENT_BINARY_DIR}/assets)
#add_dependencies(LD56_game copy_assets)
