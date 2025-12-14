add_library(glad
    glad/src/glad.c
)

target_include_directories(glad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/glad/include
)

add_library(miniaudio INTERFACE)
target_include_directories(miniaudio INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/miniaudio
)

add_library(kissfft STATIC
    kissfft/kiss_fft.c
)

target_include_directories(kissfft PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/kissfft
)
