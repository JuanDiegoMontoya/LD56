# external content definitions
include(FetchContent)

set(GLFW_BUILD_TESTS false)
set(GLFW_BUILD_DOCS false)
set(GLFW_INSTALL false)
set(GLFW_BUILD_EXAMPLES false)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG        3.3.2
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm
    GIT_TAG        1.0.1
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG        v1.91.2
)

set(FWOG_BUILD_EXAMPLES false)
FetchContent_Declare(
    fwog
    GIT_REPOSITORY https://github.com/JuanDiegoMontoya/Fwog.git
    GIT_TAG        8eebdb03b2f7250cbe2e23e4953e8862266ef253
)

FetchContent_Declare(
    EnTT
    GIT_REPOSITORY https://github.com/skypjack/entt.git
    GIT_TAG        v3.13.2
)

FetchContent_MakeAvailable(glm glfw fwog EnTT)

FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)

    add_library(imgui
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)

    target_include_directories(imgui PUBLIC 
        ${imgui_SOURCE_DIR} 
        ${imgui_SOURCE_DIR}/backends
        ${glfw_SOURCE_DIR}/include)

    target_link_libraries(imgui PRIVATE glfw)
endif()

target_compile_definitions(glm INTERFACE GLM_FORCE_SILENT_WARNINGS)