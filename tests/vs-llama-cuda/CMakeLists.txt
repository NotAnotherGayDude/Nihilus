# Copyright (c) 2025 RealTimeChris (Chris M.)
# 
# This software is offered under a restricted-use license to a party (hereafter “the Licensee”) whose identity shall be disclosed and confirmed in writing by the Author.
# 
# The License is granted under the following terms:
# 
# 1. **Scope of Use**: The Licensee is granted an exclusive right to use this software for internal purposes only. Redistribution, 
#       sublicensing, public disclosure, or publication of this software or derivative works is prohibited without explicit written consent from the Author.
# 
# 2. **Ownership**: The Author retains full ownership of the software and all intellectual property rights.
# 
# 3. **Revocability**: The Author reserves the right to revoke this license at any time, for any reason, without compensation. Upon revocation, 
#       the Licensee must immediately cease all use of the software and destroy all copies in their possession.
# 
# 4. **Transferability**: This license is non-transferable and may not be reassigned without the Author’s written consent.
# 
# 5. **No Warranty**: The software is provided "as is", without warranty of any kind, express or implied.
# Signed,  
# RealTimeChris (Chris M.)  
# 2025

cmake_minimum_required(VERSION 3.18)

project(
  "nihilus_performance"
  VERSION "${PRODUCT_VERSION}"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(FetchContent)

FetchContent_Declare(
  llama.cpp
  GIT_REPOSITORY https://github.com/notanothergaydude/llama.cpp.git
  GIT_TAG "benchmark"
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama.cpp)

FetchContent_Declare(
  benchmarksuite
  GIT_REPOSITORY https://github.com/realtimechris/benchmarksuite.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(benchmarksuite)

add_executable(
  "nihilus_performance" 
  "./main.cpp"
 )

add_subdirectory("./common")

target_link_libraries(
	"nihilus_performance" PUBLIC 
	nihilus::nihilus llama common BenchmarkSuite::BenchmarkSuite ggml ggml-cpu
)

if (WIN32)
	install(
		FILES 
		"$<TARGET_PDB_FILE:nihilus_performance>"
		DESTINATION "bin"
		OPTIONAL
	)
endif()

install(
	FILES 
	"$<TARGET_FILE:nihilus_performance>"
	DESTINATION "bin"
	OPTIONAL
)

install(
      FILES "$<TARGET_FILE:ggml>"
      DESTINATION "$<TARGET_FILE_DIR:nihilus_performance>"
      OPTIONAL
    )
	install(
      FILES "$<TARGET_FILE:ggml-cuda>"
      DESTINATION "$<TARGET_FILE_DIR:nihilus_performance>"
      OPTIONAL
    )
    install(
      FILES "$<TARGET_FILE:ggml-cpu>"
      DESTINATION "$<TARGET_FILE_DIR:nihilus_performance>"
      OPTIONAL
    )
    install(
      FILES "$<TARGET_FILE:ggml-base>"
      DESTINATION "$<TARGET_FILE_DIR:nihilus_performance>"
      OPTIONAL
    )

install(
	FILES 
	"$<TARGET_FILE:llama>"
	DESTINATION "$<TARGET_FILE_DIR:nihilus_performance>"
	OPTIONAL
)
