/*
Copyright (c) 2025 RealTimeChris (Chris M.)

This file is part of software offered under a restricted-use license to a designated Licensee,
whose identity is confirmed in writing by the Author.

License Terms (Summary):
- Exclusive, non-transferable license for internal use only.
- Redistribution, sublicensing, or public disclosure is prohibited without written consent.
- Full ownership remains with the Author.
- License may terminate if unused for [X months], if materially breached, or by mutual agreement.
- No warranty is provided, express or implied.

Full license terms are provided in the LICENSE file distributed with this software.

Signed,
RealTimeChris (Chris M.)
2025
*/

#if NIHILUS_COMPILER_CUDA

	#pragma once

	#include <cuda_runtime.h>
	#include <cuda.h>
	#include <nihilus-incl/cuda/nihilus_gpu_properties.hpp>
	#include <nihilus-incl/common/array.hpp>
	#include <nihilus-incl/infra/core_bases.hpp>
	#include <nihilus-incl/common/config.hpp>
	#include <nihilus-incl/cuda/memory_buffer.hpp>
	#include <nihilus-incl/cuda/cuda_12.hpp>

namespace nihilus {

	void print_cuda_arch() {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	}

}
#endif
