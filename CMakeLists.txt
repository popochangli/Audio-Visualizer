cmake_minimum_required(VERSION 3.15)
project(AudioVisualizerSH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

add_executable(audio_visualizer
    src/main.cpp
)

target_link_libraries(audio_visualizer PRIVATE
    OpenGL::GL
    glfw
)

target_include_directories(audio_visualizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/miniaudio
)

# Copy shader directory next to the executable so relative paths like
# "shaders/sphere.vert" work when running from the build folder.
add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            ${CMAKE_CURRENT_BINARY_DIR}/shaders
)

add_dependencies(audio_visualizer copy_shaders)

