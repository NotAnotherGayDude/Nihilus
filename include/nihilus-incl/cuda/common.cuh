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

#pragma once

#if NIHILUS_CUDA_ENABLED

	#include <cuda_runtime.h>
	#include <cuda.h>
	#include <nihilus-incl/cuda/nihilus_gpu_arch.hpp>
	#include <nihilus-incl/common/array.hpp>
	#include <nihilus-incl/common/kernel_traits.hpp>
	#include <nihilus-incl/infra/core_bases.hpp>
	#include <nihilus-incl/common/config.hpp>
	#include <nihilus-incl/cuda/cuda_12.cuh>

namespace nihilus {

	void print_cuda_arch() {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	}

	template<model_config config> struct memory_transfer;

	template<model_config config>
		requires(config.device_type == device_types::gpu)
	struct memory_transfer<config> {
		template<typename value_type> NIHILUS_INLINE static void host_to_device(const value_type* src, value_type* dst, uint64_t count) noexcept {
			auto result = cudaMemcpy(static_cast<void*>(dst), static_cast<const void*>(src), sizeof(value_type) * count, cudaMemcpyHostToDevice);
			if constexpr (config.dev) {
				if (result != cudaError::cudaSuccess) {
					log<log_levels::error>(std::string{ "Failed to copy from host to device (pointer types): " } + cudaGetErrorString(result));
				}
			}
		}
		template<not_pointer_types value_type> NIHILUS_INLINE static void host_to_device(const value_type& src, value_type* dst) noexcept {
			cudaMemcpy(dst, &src, sizeof(value_type), cudaMemcpyHostToDevice);
		}
		template<pointer_types value_type> NIHILUS_INLINE static void device_to_host(const value_type* src, value_type* dst, uint64_t count) noexcept {
			auto result = cudaMemcpy(dst, src, sizeof(value_type) * count, cudaMemcpyDeviceToHost);
			if constexpr (config.dev) {
				if (result != cudaError::cudaSuccess) {
					log<log_levels::error>(std::string{ "Failed to copy from device to host (pointer types): " } + cudaGetErrorString(result));
				}
			}
		}
		template<not_pointer_types value_type> NIHILUS_INLINE static void device_to_host(const value_type* src, value_type& dst) noexcept {
			cudaMemcpy(&dst, src, sizeof(value_type), cudaMemcpyDeviceToHost);
		}
	};

	template<model_config config>
		requires(config.device_type == device_types::gpu)
	struct memory_buffer<config> {
		using value_type   = uint8_t;
		using pointer	   = value_type*;
		using uint64_types = uint64_t;
		using size_type	   = uint64_t;

		NIHILUS_INLINE memory_buffer() noexcept								   = default;
		NIHILUS_INLINE memory_buffer& operator=(const memory_buffer&) noexcept = delete;
		NIHILUS_INLINE memory_buffer(const memory_buffer&) noexcept			   = delete;

		NIHILUS_INLINE memory_buffer& operator=(memory_buffer&& other) noexcept {
			if (this != &other) {
				std::swap(data_val, other.data_val);
				std::swap(size_val, other.size_val);
			}
			return *this;
		}

		NIHILUS_INLINE memory_buffer(memory_buffer&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE void init(uint64_t size) noexcept {
			if (data_val) {
				clear();
			}

			cudaError_t result = cudaMalloc(&data_val, size);
			if (result != cudaSuccess) {
				data_val					   = nullptr;
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "memory_buffer - failed to allocate GPU memory", location>::impl();
			}

			size_val = size;
		}

		NIHILUS_INLINE void deinit() noexcept {
			clear();
		}

		NIHILUS_INLINE uint64_types size() noexcept {
			return size_val;
		}

		NIHILUS_INLINE pointer data() noexcept {
			return data_val;
		}

		NIHILUS_INLINE void* claim_memory(uint64_t offset_to_claim) noexcept {
			uint64_t aligned_amount = round_up_to_multiple<64>(offset_to_claim);
			if (aligned_amount > size_val) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "memory_buffer - not enough memory allocated!", location>::impl();
			}
			pointer return_value = data_val + aligned_amount;
			return return_value;
		}

		NIHILUS_INLINE ~memory_buffer() noexcept {
			clear();
		}

	  protected:
		value_type* data_val{};
		size_type size_val{};

		NIHILUS_INLINE void clear() noexcept {
			if (data_val) {
				cudaError_t result = cudaFree(data_val);
				data_val		   = nullptr;
				size_val		   = 0;
			}
		}
	};

}
#endif
