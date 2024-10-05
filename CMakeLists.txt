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

include(FetchContent)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Distribution")
set(DOUBLE_PRECISION OFF)
#set(GENERATE_DEBUG_SYMBOLS ON)
#set(OVERRIDE_CXX_FLAGS ON)
set(CROSS_PLATFORM_DETERMINISTIC OFF)
set(INTERPROCEDURAL_OPTIMIZATION ON)
set(FLOATING_POINT_EXCEPTIONS_ENABLED OFF)
set(CPP_EXCEPTIONS_ENABLED ON)
set(CPP_RTTI_ENABLED ON)
set(OBJECT_LAYER_BITS 16)
set(USE_SSE4_1 ON)
set(USE_SSE4_2 ON)
set(USE_AVX ON)
set(USE_AVX2 ON)
set(USE_AVX512 OFF)
set(USE_LZCNT ON)
set(USE_TZCNT ON)
set(USE_F16C ON)
set(USE_FMADD ON)
FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics
    GIT_TAG v5.1.0
)
if(NOT JoltPhysics_POPULATED)
    FetchContent_MakeAvailable(JoltPhysics)

    set(PHYSICS_REPO_ROOT ${joltphysics_SOURCE_DIR})
    include(${joltphysics_SOURCE_DIR}/Jolt/Jolt.cmake)
    target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:RelWithDebInfo>:JPH_PROFILE_ENABLED;JPH_DEBUG_RENDERER>")
    if (FLOATING_POINT_EXCEPTIONS_ENABLED AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:RelWithDebInfo>:JPH_FLOATING_POINT_EXCEPTIONS_ENABLED>")
    endif()
endif()

message(asdfJoltIncludeDir="${joltphysics_SOURCE_DIR}")

target_include_directories(LD56_game PUBLIC	src	external ${joltphysics_SOURCE_DIR}/..)

target_link_libraries(LD56_game glm EnTT::EnTT fwog glfw lib_glad imgui Jolt)

#message(JoltIncludeDir="${JoltIncludeDir}")

#add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/assets ${CMAKE_CURRENT_BINARY_DIR}/assets)
#add_dependencies(LD56_game copy_assets)
