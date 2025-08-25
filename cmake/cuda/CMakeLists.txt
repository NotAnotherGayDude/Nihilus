enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE NIHILUS_CUDA_SOURCES CONFIGURE_DEPENDS 
    "${CMAKE_SOURCE_DIR}/src/cuda/*.cu"
)

file(GLOB_RECURSE NIHILUS_CUDA_HEADERS CONFIGURE_DEPENDS 
    "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cuda/*.cuh"
)

message(STATUS "CUDA: ${CUDAARCHS}")

add_library("${PROJECT_NAME}-cuda" STATIC "${NIHILUS_CUDA_SOURCES}")
add_library("${PROJECT_NAME}::${PROJECT_NAME}-cuda" ALIAS "${PROJECT_NAME}-cuda")

target_include_directories(
    "${PROJECT_NAME}-cuda" PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

target_link_options("${PROJECT_NAME}-cuda" PUBLIC
    "$<$<PLATFORM_ID:Linux>:${NIHILUS_LINUX_LINK_OPTIONS}>"
    "$<$<PLATFORM_ID:Darwin>:${NIHILUS_MACOS_LINK_OPTIONS}>"
    "$<$<PLATFORM_ID:Windows>:${NIHILUS_WINDOWS_LINK_OPTIONS}>"
)

target_compile_definitions(
    "${PROJECT_NAME}-cuda" PUBLIC
    "${NIHILUS_COMMON_COMPILE_DEFITIONS}"
    "${NIHILUS_MSVC_COMPILE_DEFITIONS}"
)
    
set_property(TARGET "${PROJECT_NAME}-cuda" PROPERTY CUDA_STANDARD 20)
set_property(TARGET "${PROJECT_NAME}-cuda" PROPERTY CUDA_STANDARD_REQUIRED ON)
    
target_link_libraries("${PROJECT_NAME}-cuda" PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)