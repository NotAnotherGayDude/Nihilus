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
// cuda_12.hpp

#if NIHILUS_CUDA_ENABLED

	#pragma once

	#include <nihilus-incl/infra/core_bases.hpp>
	#include <nihilus-incl/common/common.hpp>
	#include <nihilus-incl/common/utility.hpp>
	#include <nihilus-incl/common/type_traits.hpp>
	#include <nihilus-incl/infra/core_traits.hpp>
	#include <cuda_runtime.h>
	#include <cuda_fp16.h>

namespace nihilus {

	enum class scale_directions {
		up,
		down,
		scaled_separately,
	};

	struct scaling_results {
		scale_directions scale_direction{};
		uint32_t value_or_ratio01{};
		uint32_t value_or_ratio02{};
	};

	template<uint32_t dim_00> NIHILUS_HOST scaling_results conver_dimensions(uint32_t x2) {
		uint64_t initial_product = static_cast<uint64_t>(dim_00) * x2;

		if (initial_product <= gpu_properties::optimal_block_size) {
			double max_scale_factor = std::sqrt(static_cast<double>(gpu_properties::optimal_block_size) / initial_product);
			uint32_t n				= static_cast<uint32_t>(std::floor(max_scale_factor));
			return { scale_directions::up, static_cast<uint32_t>(n * dim_00), static_cast<uint32_t>(n * x2) };
		} else {
			double divisor_double	= std::sqrt(static_cast<double>(initial_product) / gpu_properties::optimal_block_size);
			uint32_t single_divisor = static_cast<uint32_t>(std::ceil(divisor_double));

			if (single_divisor > 0 && (dim_00 % single_divisor == 0) && (x2 % single_divisor == 0) && (dim_00 / single_divisor >= 1) && (x2 / single_divisor >= 1)) {
				return { scale_directions::down, static_cast<uint32_t>(single_divisor), 0 };
			} else {
				double required_divisor_product = static_cast<double>(initial_product) / gpu_properties::optimal_block_size;

				for (uint32_t d1 = 1; d1 <= dim_00; ++d1) {
					if (dim_00 % d1 != 0)
						continue;

					double d2_double	 = required_divisor_product / d1;
					uint32_t d2_required = static_cast<uint32_t>(std::ceil(d2_double));

					if (d2_required > x2)
						continue;
					if (x2 % d2_required != 0)
						continue;

					if ((dim_00 / d1) >= 1 && (x2 / d2_required) >= 1) {
						return { scale_directions::scaled_separately, static_cast<uint32_t>(d1), static_cast<uint32_t>(d2_required) };
					}
				}
				return { scale_directions::scaled_separately, 0, 0 };
			}
		}
	}

	struct cuda_launch_params {
		uint64_t block_chunk_size;
		uint64_t grid_chunk_size;
		uint64_t blocks_per_grid;
		uint64_t threads_per_block;
		uint64_t warp_aligned_size;
	};

	template<typename output_type, core_types kernel_type> NIHILUS_HOST static constexpr cuda_launch_params calculate_gpu_launch_params(output_type& output) {
		cuda_launch_params params;
		uint64_t total_required_bytes = output.total_required_bytes_rt;

		bool fits_in_l2 = total_required_bytes <= static_cast<uint64_t>(static_cast<float>(gpu_properties::l2_cache_size) * 0.6f);

		bool fits_in_shared = total_required_bytes <= static_cast<uint64_t>(static_cast<float>(gpu_properties::shared_mem_per_block) * 0.8f);

		if (fits_in_l2) {
			params.blocks_per_grid	 = gpu_properties::optimal_grid_size;
			params.threads_per_block = gpu_properties::optimal_block_size;
			params.grid_chunk_size	 = std::numeric_limits<uint64_t>::max();

			if (fits_in_shared) {
				params.block_chunk_size = gpu_properties::optimal_block_size * gpu_properties::warp_size;
			} else {
				params.block_chunk_size = gpu_properties::optimal_block_size;
			}

		} else {
			const uint64_t usable_l2 = static_cast<uint64_t>(static_cast<float>(gpu_properties::l2_cache_size) * 0.5f);

			const uint64_t chunks_needed = (total_required_bytes + usable_l2 - 1) / usable_l2;

			params.blocks_per_grid = detail::min(gpu_properties::optimal_grid_size, gpu_properties::total_threads / gpu_properties::optimal_block_size / chunks_needed);

			params.threads_per_block = gpu_properties::optimal_block_size;
			params.grid_chunk_size	 = usable_l2;
			params.block_chunk_size	 = usable_l2 / params.blocks_per_grid;
		}
		params.warp_aligned_size = ((params.block_chunk_size + gpu_properties::warp_size - 1) / gpu_properties::warp_size) * gpu_properties::warp_size;

		return params;
	}

	template<typename value_type> NIHILUS_DEVICE constexpr decltype(auto) device_forward(value_type&& arg) noexcept {
		return static_cast<value_type&&>(arg);
	}

	template<typename value_type> NIHILUS_DEVICE constexpr decltype(auto) device_forward(value_type& arg) noexcept {
		return static_cast<value_type&&>(arg);
	}

	enum class get_value_type_errors {
		invalid_type,
	};

	template<typename value_type> struct get_value_type {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) = delete;
	};

	template<int8_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_char1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_char2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_char3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_char4(device_forward<value_types>(args)...);
			}
		}
	};

	template<int16_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_short1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_short2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_short3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_short4(device_forward<value_types>(args)...);
			}
		}
	};

	template<int32_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_int1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_int2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_int3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_int4(device_forward<value_types>(args)...);
			}
		}
	};

	template<int64_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_long1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_long2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_long3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_long4(device_forward<value_types>(args)...);
			}
		}
	};

	template<uint8_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_uchar1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_uchar2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_uchar3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_uchar4(device_forward<value_types>(args)...);
			}
		}
	};

	template<uint16_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_ushort1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_ushort2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_ushort3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_ushort4(device_forward<value_types>(args)...);
			}
		}
	};

	template<uint32_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_uint1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_uint2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_uint3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_uint4(device_forward<value_types>(args)...);
			}
		}
	};

	template<uint64_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_ulong1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_ulong2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_ulong3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_ulong4(device_forward<value_types>(args)...);
			}
		}
	};

	template<float32_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_float1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_float2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_float3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_float4(device_forward<value_types>(args)...);
			}
		}
	};

	template<float64_cuda_types value_type> struct get_value_type<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			if constexpr (dim01_types<value_type>) {
				return make_double1(device_forward<value_types>(args)...);
			} else if constexpr (dim02_types<value_type>) {
				return make_double2(device_forward<value_types>(args)...);
			} else if constexpr (dim03_types<value_type>) {
				return make_double3(device_forward<value_types>(args)...);
			} else if constexpr (dim04_types<value_type>) {
				return make_double4(device_forward<value_types>(args)...);
			}
		}
	};

	enum class binary_op_types {
		add,
		mul,
		sub,
		div,
	};

	template<binary_op_types> struct binary_op_core;

	template<> struct binary_op_core<binary_op_types::add> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return device_forward<value_type01>(val01) + static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 += static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}
	};

	template<> struct binary_op_core<binary_op_types::mul> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return device_forward<value_type01>(val01) * static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 *= static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}
	};

	template<> struct binary_op_core<binary_op_types::sub> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return device_forward<value_type01>(val01) - static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 -= static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}
	};

	template<> struct binary_op_core<binary_op_types::div> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return device_forward<value_type01>(val01) / static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 /= static_cast<base_type<value_type01>>(device_forward<value_type02>(val02));
		}
	};

	template<typename value_type, binary_op_types binary_op_type> struct binary_op_base;

	template<dim01_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
		}
	};

	template<dim02_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x),
				op_core_type::impl(device_forward<value_type01>(val01).y, device_forward<value_type02>(val02).y));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
			op_core_type::impl_in_place(val01.y, device_forward<value_type02>(val02).y);
		}
	};

	template<dim03_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x),
				op_core_type::impl(device_forward<value_type01>(val01).y, device_forward<value_type02>(val02).y),
				op_core_type::impl(device_forward<value_type01>(val01).z, device_forward<value_type02>(val02).z));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
			op_core_type::impl_in_place(val01.y, device_forward<value_type02>(val02).y);
			op_core_type::impl_in_place(val01.z, device_forward<value_type02>(val02).z);
		}
	};

	template<dim04_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value_type<value_type01>::impl(op_core_type::impl(device_forward<value_type01>(val01).x, device_forward<value_type02>(val02).x),
				op_core_type::impl(device_forward<value_type01>(val01).y, device_forward<value_type02>(val02).y),
				op_core_type::impl(device_forward<value_type01>(val01).z, device_forward<value_type02>(val02).z),
				op_core_type::impl(device_forward<value_type01>(val01).w, device_forward<value_type02>(val02).w));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, device_forward<value_type02>(val02).x);
			op_core_type::impl_in_place(val01.y, device_forward<value_type02>(val02).y);
			op_core_type::impl_in_place(val01.z, device_forward<value_type02>(val02).z);
			op_core_type::impl_in_place(val01.w, device_forward<value_type02>(val02).w);
		}
	};

	template<binary_op_types binary_op_type> struct binary_op {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return binary_op_base<value_type01, binary_op_type>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl_in_place(value_type01& val01, value_type02&& val02) {
			return binary_op_base<value_type01, binary_op_type>::impl_in_place(val01, device_forward<value_type02>(val02));
		}
	};

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator+=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::add>::impl_in_place(val01, device_forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator+(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::add>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator*=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::mul>::impl_in_place(val01, device_forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator*(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::mul>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator-=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::sub>::impl_in_place(val01, device_forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator-(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::sub>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator/=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::div>::impl_in_place(val01, device_forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator/(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::div>::impl(device_forward<value_type01>(val01), device_forward<value_type02>(val02));
	}

	template<enum_types enum_type, enum_type enum_value, typename core_bases_type> NIHILUS_DEVICE decltype(auto) get_core(core_bases_type& core_bases) noexcept {
		return *static_cast<std::remove_cvref_t<typename core_bases_type::template core_base_type<enum_type, enum_value>>*>(&core_bases);
	}

	template<const model_config& config_new, auto kernel_type> NIHILUS_DEVICE static decltype(auto) get_adjacent_value_device(auto& parse_core) {
		using derived_type	   = core_traits<config_new, kernel_type>;
		using thread_pool_type = thread_pool<config_new>;
		using core_bases_type  = typename thread_pool<config_new>::core_bases_type;
		return *static_cast<derived_type*>(static_cast<core_bases_type*>(&parse_core));
	}

	template<typename core_traits_type> __global__ void token_embeddings_prompt_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs params) {
		static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
		static constexpr uint64_t block_size		   = core_traits_type::prof::weight_type::quant_count;
		static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

		const auto* __restrict__ weight_data = params.weight_data;
		const auto* __restrict__ token_ids	 = params.token_ids;
		auto* __restrict__ output_data		 = params.output_data;

		const uint64_t token_idx		 = blockIdx.x;
		const uint64_t thread_id		 = threadIdx.x;
		const uint64_t threads_per_block = blockDim.x;

		if (token_idx >= sequence_length)
			return;

		const auto token_id				 = token_ids[token_idx];
		const auto* __restrict__ src_row = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);
		auto* __restrict__ dst_row		 = output_data + (token_idx * embedding_length);

		for (uint64_t block_idx = thread_id; block_idx < blocks_per_embedding; block_idx += threads_per_block) {
			const auto& block				   = src_row[block_idx];
			const auto scale				   = __half2float(block.d);
			const auto* __restrict__ quantized = block.qs;
			const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
			for (uint64_t j = 0; j < block_size; ++j) {
				if (base_offset + j < embedding_length) {
					dst_row[base_offset + j] = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
				}
			}
		}
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
			auto& get_rows_op = params.values.template get_core<token_embeddings_types, token_embeddings_types::get_rows>();

			const uint64_t sequence_length = get_rows_op.get_mutable_dim();
			auto& weights_core			   = get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core			   = get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_op			   = weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op			   = inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();

			const auto* __restrict__ weight_data = token_embd_op.data;
			const auto* __restrict__ token_ids	 = inp_tokens_op.data;
			auto* __restrict__ output_data		 = get_rows_op.data;

			params.data_ptrs.set_ptrs(output_data, weight_data, token_ids);

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			if (sequence_length > 0) {
				static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
				static constexpr uint64_t block_size		   = core_traits_type::prof::weight_type::quant_count;
				static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

				constexpr uint64_t threads_per_block = blocks_per_embedding <= 32 ? 32
					: blocks_per_embedding <= 64								  ? 64
					: blocks_per_embedding <= 128								  ? 128
					: blocks_per_embedding <= 256								  ? 256
																				  : 512;

				const uint64_t blocks_per_grid = sequence_length;

				token_embeddings_prompt_eval_time<core_traits_type><<<blocks_per_grid, threads_per_block>>>(sequence_length, params.data_ptrs);

				if constexpr (config.dev ) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
					}
				}

				cudaDeviceSynchronize();

				if constexpr (config.dev ) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
					}
				}
			}
		}
	};

	template<typename core_traits_type> __global__ void token_embeddings_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs params) {
		static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
		static constexpr uint64_t block_size		   = core_traits_type::prof::weight_type::quant_count;
		static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

		const auto* __restrict__ weight_data = params.weight_data;
		const auto* __restrict__ token_ids	 = params.token_ids;
		auto* __restrict__ output_data		 = params.output_data;

		const auto token_id				 = token_ids[sequence_length - 1];
		const auto* __restrict__ src_row = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);

		const uint64_t thread_id	 = blockIdx.x * blockDim.x + threadIdx.x;
		const uint64_t total_threads = gridDim.x * blockDim.x;

		for (uint64_t elem_idx = thread_id; elem_idx < embedding_length; elem_idx += total_threads) {
			const uint64_t block_idx	 = elem_idx / block_size;
			const uint64_t elem_in_block = elem_idx % block_size;

			const auto& block		 = src_row[block_idx];
			const auto scale		 = __half2float(block.d);
			const auto quantized_val = block.qs[elem_in_block];

			output_data[elem_idx] = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized_val);
		}
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
			auto& get_rows_op			   = params.values.template get_core<token_embeddings_types, token_embeddings_types::get_rows>();
			auto& weights_core			   = get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core			   = get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_op			   = weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op			   = inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();
			const uint64_t sequence_length = get_rows_op.get_mutable_dim();

			const auto* __restrict__ weight_data = token_embd_op.data;
			const auto* __restrict__ token_ids	 = inp_tokens_op.data;
			auto* __restrict__ output_data		 = get_rows_op.data;

			params.data_ptrs.set_ptrs(output_data, weight_data, token_ids);

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			static constexpr uint64_t embedding_length = core_traits_type::mt::embedding_length;

			constexpr uint64_t threads_per_block = 256;
			const uint64_t blocks_per_grid		 = (embedding_length + threads_per_block - 1) / threads_per_block;

			token_embeddings_eval_time<core_traits_type><<<blocks_per_grid, threads_per_block>>>(sequence_length, params.data_ptrs);

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			cudaDeviceSynchronize();

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
				}
			}
		}
	};

	template<typename core_traits_type>
	__global__ void mega_qkv_prep_and_cache_publish_prompt_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs params) {
		static constexpr uint64_t embedding_length = core_traits_type::mt::embedding_length;
		static constexpr uint64_t rope_dim		   = core_traits_type::mt::rope_dimension_count;
		static constexpr uint64_t n_head		   = core_traits_type::mt::attention_head_count;
		static constexpr uint64_t n_head_kv		   = core_traits_type::mt::attention_head_count_kv;
		static constexpr uint64_t n_embd_kv_gqa	   = core_traits_type::mt::n_embd_kv_gqa;

		static constexpr uint64_t block_size	 = core_traits_type::prof::weight_type::quant_count;
		static constexpr uint64_t blocks_per_row = (embedding_length + block_size - 1) / block_size;

		const auto* __restrict__ inp_embd_data	  = params.inp_embd_data;
		const auto* __restrict__ attn_norm_w_data = params.attn_norm_w_data;
		const auto* __restrict__ attn_q_w_data	  = params.attn_q_w_data;
		const auto* __restrict__ attn_k_w_data	  = params.attn_k_w_data;
		const auto* __restrict__ attn_v_w_data	  = params.attn_v_w_data;
		const auto* __restrict__ inp_pos_data	  = params.inp_pos_data;
		const auto* __restrict__ rope_freqs_data  = params.rope_freqs_data;
		auto* __restrict__ cache_k_data			  = params.cache_k_data;
		auto* __restrict__ cache_v_data			  = params.cache_v_data;
		auto* __restrict__ q_output_data		  = params.q_output_data;

		const uint64_t thread_id	 = blockIdx.x * blockDim.x + threadIdx.x;
		const uint64_t total_threads = gridDim.x * blockDim.x;

		for (uint64_t token_idx = thread_id; token_idx < sequence_length; token_idx += total_threads) {
			// RMS Norm
			typename core_traits_type::compute_t sum_sq = 0;
			for (uint64_t i = 0; i < embedding_length; ++i) {
				const auto val = inp_embd_data[token_idx * embedding_length + i];
				sum_sq += val * val;
			}
			typename core_traits_type::compute_t rms = rsqrtf(sum_sq / embedding_length + 1e-6f);
			/*
			// Mul with attention norm weights (plain floats, no dequantization needed)
			typename core_traits_type::compute_t normed[embedding_length];
			for (uint64_t i = 0; i < embedding_length; ++i) {
				normed[i] = inp_embd_data[token_idx * embedding_length + i] * rms * attn_norm_w_data[i];
			}

			// Q: mul_mat, reshape, rope (dequantize Q weights)
			for (uint64_t h = 0; h < n_head; ++h) {
				for (uint64_t d = 0; d < rope_dim; ++d) {
					typename core_traits_type::compute_t q_val = 0;
					const auto* __restrict__ weight_row		   = attn_q_w_data + ((h * rope_dim + d) * blocks_per_row);

					for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
						const auto& block				   = weight_row[block_idx];
						const auto scale				   = __half2float(block.d);
						const auto* __restrict__ quantized = block.qs;
						const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
						for (uint64_t j = 0; j < block_size; ++j) {
							if (base_offset + j < embedding_length) {
								const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
								q_val += dequantized_weight * normed[base_offset + j];
							}
						}
					}

					// Apply RoPE
					const uint64_t pos								   = inp_pos_data[token_idx];
					const typename core_traits_type::compute_t freq	   = rope_freqs_data[d];
					const typename core_traits_type::compute_t cos_val = cosf(pos * freq);
					const typename core_traits_type::compute_t sin_val = sinf(pos * freq);

					if (d % 2 == 0) {
						typename core_traits_type::compute_t q_pair = 0;
						if (d + 1 < rope_dim) {
							const auto* __restrict__ weight_row_pair = attn_q_w_data + ((h * rope_dim + d + 1) * blocks_per_row);
							for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
								const auto& block				   = weight_row_pair[block_idx];
								const auto scale				   = __half2float(block.d);
								const auto* __restrict__ quantized = block.qs;
								const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
								for (uint64_t j = 0; j < block_size; ++j) {
									if (base_offset + j < embedding_length) {
										const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
										q_pair += dequantized_weight * normed[base_offset + j];
									}
								}
							}
						}
						q_output_data[token_idx * n_head * rope_dim + h * rope_dim + d] = q_val * cos_val - q_pair * sin_val;
					} else {
						typename core_traits_type::compute_t q_pair = 0;
						const auto* __restrict__ weight_row_pair	= attn_q_w_data + ((h * rope_dim + d - 1) * blocks_per_row);
						for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
							const auto& block				   = weight_row_pair[block_idx];
							const auto scale				   = __half2float(block.d);
							const auto* __restrict__ quantized = block.qs;
							const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
							for (uint64_t j = 0; j < block_size; ++j) {
								if (base_offset + j < embedding_length) {
									const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
									q_pair += dequantized_weight * normed[base_offset + j];
								}
							}
						}
						q_output_data[token_idx * n_head * rope_dim + h * rope_dim + d] = q_val * cos_val + q_pair * sin_val;
					}
				}
			}

			// K: mul_mat, reshape, rope, cache (dequantize K weights)
			for (uint64_t h = 0; h < n_head_kv; ++h) {
				for (uint64_t d = 0; d < rope_dim; ++d) {
					typename core_traits_type::compute_t k_val = 0;
					const auto* __restrict__ weight_row		   = attn_k_w_data + ((h * rope_dim + d) * blocks_per_row);

					for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
						const auto& block				   = weight_row[block_idx];
						const auto scale				   = __half2float(block.d);
						const auto* __restrict__ quantized = block.qs;
						const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
						for (uint64_t j = 0; j < block_size; ++j) {
							if (base_offset + j < embedding_length) {
								const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
								k_val += dequantized_weight * normed[base_offset + j];
							}
						}
					}

					// Apply RoPE and store to cache
					const uint64_t pos								   = inp_pos_data[token_idx];
					const typename core_traits_type::compute_t freq	   = rope_freqs_data[d];
					const typename core_traits_type::compute_t cos_val = cosf(pos * freq);
					const typename core_traits_type::compute_t sin_val = sinf(pos * freq);

					typename core_traits_type::kv_store_t k_cached;
					if (d % 2 == 0) {
						typename core_traits_type::compute_t k_pair = 0;
						if (d + 1 < rope_dim) {
							const auto* __restrict__ weight_row_pair = attn_k_w_data + ((h * rope_dim + d + 1) * blocks_per_row);
							for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
								const auto& block				   = weight_row_pair[block_idx];
								const auto scale				   = __half2float(block.d);
								const auto* __restrict__ quantized = block.qs;
								const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
								for (uint64_t j = 0; j < block_size; ++j) {
									if (base_offset + j < embedding_length) {
										const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
										k_pair += dequantized_weight * normed[base_offset + j];
									}
								}
							}
						}
						k_cached = static_cast<typename core_traits_type::kv_store_t>(k_val * cos_val - k_pair * sin_val);
					} else {
						typename core_traits_type::compute_t k_pair = 0;
						const auto* __restrict__ weight_row_pair	= attn_k_w_data + ((h * rope_dim + d - 1) * blocks_per_row);
						for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
							const auto& block				   = weight_row_pair[block_idx];
							const auto scale				   = __half2float(block.d);
							const auto* __restrict__ quantized = block.qs;
							const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
							for (uint64_t j = 0; j < block_size; ++j) {
								if (base_offset + j < embedding_length) {
									const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
									k_pair += dequantized_weight * normed[base_offset + j];
								}
							}
						}
						k_cached = static_cast<typename core_traits_type::kv_store_t>(k_val * cos_val + k_pair * sin_val);
					}
					cache_k_data[h * rope_dim * sequence_length + d * sequence_length + token_idx] = k_cached;
				}
			}

			// V: mul_mat, transpose, cache (dequantize V weights)
			for (uint64_t h = 0; h < n_head_kv; ++h) {
				for (uint64_t d = 0; d < rope_dim; ++d) {
					typename core_traits_type::compute_t v_val = 0;
					const auto* __restrict__ weight_row		   = attn_v_w_data + ((h * rope_dim + d) * blocks_per_row);

					for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
						const auto& block				   = weight_row[block_idx];
						const auto scale				   = __half2float(block.d);
						const auto* __restrict__ quantized = block.qs;
						const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
						for (uint64_t j = 0; j < block_size; ++j) {
							if (base_offset + j < embedding_length) {
								const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
								v_val += dequantized_weight * normed[base_offset + j];
							}
						}
					}
					cache_v_data[h * rope_dim * sequence_length + token_idx * rope_dim + d] = static_cast<typename core_traits_type::kv_store_t>(v_val);
				}
			}*/
		}
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
			auto& q_out_op		  = params.values.template get_core<mega_qkv_prep_and_cache_publish_types, mega_qkv_prep_and_cache_publish_types::q_out>();
			auto& weights_core	  = get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core	  = get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_core = get_adjacent_value<core_traits_type::config, core_types::token_embeddings>::impl(params);
			auto& inp_embd_op	  = token_embd_core.values.template get_core<token_embeddings_types, token_embeddings_types::get_rows>();
			auto& attn_norm_w_op  = weights_core.values.template get_core<weight_types, weight_types::attn_norm>();
			auto& attn_q_w_op	  = weights_core.values.template get_core<weight_types, weight_types::attn_q>();
			auto& attn_k_w_op	  = weights_core.values.template get_core<weight_types, weight_types::attn_k>();
			auto& attn_v_w_op	  = weights_core.values.template get_core<weight_types, weight_types::attn_v>();
			auto& inp_pos_op	  = inputs_core.values.template get_core<global_input_types, global_input_types::inp_pos>();
			auto& rope_freqs_op	  = weights_core.values.template get_core<weight_types, weight_types::rope_freqs>();
			auto& cache_k_op	  = inputs_core.values.template get_core<global_input_types, global_input_types::cache_k>();
			auto& cache_v_op	  = inputs_core.values.template get_core<global_input_types, global_input_types::cache_v>();

			const uint64_t sequence_length = q_out_op.get_mutable_dim();

			cuda_launch_params launch_params = calculate_gpu_launch_params<typename core_traits_type::q_out_type, core_types::mega_qkv_prep_and_cache_publish>(q_out_op);

			const uint64_t max_threads_needed = sequence_length;
			const uint64_t actual_blocks = detail::min(launch_params.blocks_per_grid, (max_threads_needed + launch_params.threads_per_block - 1) / launch_params.threads_per_block);

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			if (sequence_length > 0) {
				mega_qkv_prep_and_cache_publish_prompt_eval_time<core_traits_type><<<actual_blocks, launch_params.threads_per_block>>>(sequence_length, params.data_ptrs);

				if constexpr (config.dev ) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
					}
				}

				cudaDeviceSynchronize();

				if constexpr (config.dev ) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
					}
				}
			}
		}
	};

	template<typename core_traits_type>
	__global__ void mega_qkv_prep_and_cache_publish_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs params) {
		static constexpr uint64_t embedding_length = core_traits_type::mt::embedding_length;
		static constexpr uint64_t rope_dim		   = core_traits_type::mt::rope_dimension_count;
		static constexpr uint64_t n_head		   = core_traits_type::mt::attention_head_count;
		static constexpr uint64_t n_head_kv		   = core_traits_type::mt::attention_head_count_kv;
		static constexpr uint64_t n_embd_kv_gqa	   = core_traits_type::mt::n_embd_kv_gqa;

		static constexpr uint64_t block_size	 = core_traits_type::prof::weight_type::quant_count;
		static constexpr uint64_t blocks_per_row = (embedding_length + block_size - 1) / block_size;

		const auto* __restrict__ inp_embd_data	  = params.inp_embd_data;
		const auto* __restrict__ attn_norm_w_data = params.attn_norm_w_data;
		const auto* __restrict__ attn_q_w_data	  = params.attn_q_w_data;
		const auto* __restrict__ attn_k_w_data	  = params.attn_k_w_data;
		const auto* __restrict__ attn_v_w_data	  = params.attn_v_w_data;
		const auto* __restrict__ inp_pos_data	  = params.inp_pos_data;
		const auto* __restrict__ rope_freqs_data  = params.rope_freqs_data;
		auto* __restrict__ cache_k_data			  = params.cache_k_data;
		auto* __restrict__ cache_v_data			  = params.cache_v_data;
		auto* __restrict__ q_output_data		  = params.q_output_data;

		const uint64_t thread_id	 = blockIdx.x * blockDim.x + threadIdx.x;
		const uint64_t total_threads = gridDim.x * blockDim.x;

		// Process only the last token
		const uint64_t token_idx = sequence_length - 1;

		// Total output elements to compute: Q (n_head * rope_dim) + K (n_head_kv * rope_dim) + V (n_head_kv * rope_dim)
		const uint64_t total_elements = n_head * rope_dim + n_head_kv * rope_dim * 2;

		// Precompute RMS norm (all threads do this, but it's small and efficient)
		typename core_traits_type::compute_t sum_sq = 0;
		//for (uint64_t i = 0; i < embedding_length; ++i) {
		//const auto val = inp_embd_data[i];
		//			sum_sq += val * val;
		//}
		//typename core_traits_type::compute_t rms = rsqrtf(sum_sq / embedding_length + 1e-6f);
		/*
		// Precompute normalized input (shared across all threads)
		__shared__ typename core_traits_type::compute_t normed[embedding_length];
		for (uint64_t i = thread_id; i < embedding_length; i += total_threads) {
			normed[i] = inp_embd_data[i] * rms * attn_norm_w_data[i];
		}
		__syncthreads();

		// Partition work across threads
		for (uint64_t elem_idx = thread_id; elem_idx < total_elements; elem_idx += total_threads) {
			if (elem_idx < n_head * rope_dim) {
				// Process Q
				const uint64_t h = elem_idx / rope_dim;
				const uint64_t d = elem_idx % rope_dim;

				typename core_traits_type::compute_t q_val = 0;
				const auto* __restrict__ weight_row		   = attn_q_w_data + ((h * rope_dim + d) * blocks_per_row);

				for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
					const auto& block				   = weight_row[block_idx];
					const auto scale				   = __half2float(block.d);
					const auto* __restrict__ quantized = block.qs;
					const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
					for (uint64_t j = 0; j < block_size; ++j) {
						if (base_offset + j < embedding_length) {
							const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
							q_val += dequantized_weight * normed[base_offset + j];
						}
					}
				}

				// Apply RoPE
				const uint64_t pos								   = inp_pos_data[token_idx];
				const typename core_traits_type::compute_t freq	   = rope_freqs_data[d];
				const typename core_traits_type::compute_t cos_val = cosf(pos * freq);
				const typename core_traits_type::compute_t sin_val = sinf(pos * freq);

				if (d % 2 == 0) {
					typename core_traits_type::compute_t q_pair = 0;
					if (d + 1 < rope_dim) {
						const auto* __restrict__ weight_row_pair = attn_q_w_data + ((h * rope_dim + d + 1) * blocks_per_row);
						for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
							const auto& block				   = weight_row_pair[block_idx];
							const auto scale				   = __half2float(block.d);
							const auto* __restrict__ quantized = block.qs;
							const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
							for (uint64_t j = 0; j < block_size; ++j) {
								if (base_offset + j < embedding_length) {
									const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
									q_pair += dequantized_weight * normed[base_offset + j];
								}
							}
						}
					}
					q_output_data[h * rope_dim + d] = q_val * cos_val - q_pair * sin_val;
				} else {
					typename core_traits_type::compute_t q_pair = 0;
					const auto* __restrict__ weight_row_pair	= attn_q_w_data + ((h * rope_dim + d - 1) * blocks_per_row);
					for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
						const auto& block				   = weight_row_pair[block_idx];
						const auto scale				   = __half2float(block.d);
						const auto* __restrict__ quantized = block.qs;
						const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
						for (uint64_t j = 0; j < block_size; ++j) {
							if (base_offset + j < embedding_length) {
								const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
								q_pair += dequantized_weight * normed[base_offset + j];
							}
						}
					}
					q_output_data[h * rope_dim + d] = q_val * cos_val + q_pair * sin_val;
				}
			} else if (elem_idx < n_head * rope_dim + n_head_kv * rope_dim) {
				// Process K
				const uint64_t k_elem_idx = elem_idx - n_head * rope_dim;
				const uint64_t h		  = k_elem_idx / rope_dim;
				const uint64_t d		  = k_elem_idx % rope_dim;

				typename core_traits_type::compute_t k_val = 0;
				const auto* __restrict__ weight_row		   = attn_k_w_data + ((h * rope_dim + d) * blocks_per_row);

				for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
					const auto& block				   = weight_row[block_idx];
					const auto scale				   = __half2float(block.d);
					const auto* __restrict__ quantized = block.qs;
					const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
					for (uint64_t j = 0; j < block_size; ++j) {
						if (base_offset + j < embedding_length) {
							const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
							k_val += dequantized_weight * normed[base_offset + j];
						}
					}
				}

				// Apply RoPE and store to cache
				const uint64_t pos								   = inp_pos_data[token_idx];
				const typename core_traits_type::compute_t freq	   = rope_freqs_data[d];
				const typename core_traits_type::compute_t cos_val = cosf(pos * freq);
				const typename core_traits_type::compute_t sin_val = sinf(pos * freq);

				typename core_traits_type::kv_store_t k_cached;
				if (d % 2 == 0) {
					typename core_traits_type::compute_t k_pair = 0;
					if (d + 1 < rope_dim) {
						const auto* __restrict__ weight_row_pair = attn_k_w_data + ((h * rope_dim + d + 1) * blocks_per_row);
						for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
							const auto& block				   = weight_row_pair[block_idx];
							const auto scale				   = __half2float(block.d);
							const auto* __restrict__ quantized = block.qs;
							const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
							for (uint64_t j = 0; j < block_size; ++j) {
								if (base_offset + j < embedding_length) {
									const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
									k_pair += dequantized_weight * normed[base_offset + j];
								}
							}
						}
					}
					k_cached = static_cast<typename core_traits_type::kv_store_t>(k_val * cos_val - k_pair * sin_val);
				} else {
					typename core_traits_type::compute_t k_pair = 0;
					const auto* __restrict__ weight_row_pair	= attn_k_w_data + ((h * rope_dim + d - 1) * blocks_per_row);
					for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
						const auto& block				   = weight_row_pair[block_idx];
						const auto scale				   = __half2float(block.d);
						const auto* __restrict__ quantized = block.qs;
						const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
						for (uint64_t j = 0; j < block_size; ++j) {
							if (base_offset + j < embedding_length) {
								const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
								k_pair += dequantized_weight * normed[base_offset + j];
							}
						}
					}
					k_cached = static_cast<typename core_traits_type::kv_store_t>(k_val * cos_val + k_pair * sin_val);
				}
				cache_k_data[h * rope_dim * sequence_length + d * sequence_length + token_idx] = k_cached;
			} else {
				// Process V
				const uint64_t v_elem_idx = elem_idx - n_head * rope_dim - n_head_kv * rope_dim;
				const uint64_t h		  = v_elem_idx / rope_dim;
				const uint64_t d		  = v_elem_idx % rope_dim;

				typename core_traits_type::compute_t v_val = 0;
				const auto* __restrict__ weight_row		   = attn_v_w_data + ((h * rope_dim + d) * blocks_per_row);

				for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
					const auto& block				   = weight_row[block_idx];
					const auto scale				   = __half2float(block.d);
					const auto* __restrict__ quantized = block.qs;
					const uint64_t base_offset		   = block_idx * block_size;

	#pragma unroll
					for (uint64_t j = 0; j < block_size; ++j) {
						if (base_offset + j < embedding_length) {
							const auto dequantized_weight = scale * static_cast<typename core_traits_type::prof::compute_type>(quantized[j]);
							v_val += dequantized_weight * normed[base_offset + j];
						}
					}
				}
				cache_v_data[h * rope_dim * sequence_length + token_idx * rope_dim + d] = static_cast<typename core_traits_type::kv_store_t>(v_val);
			}
		}*/
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
			auto& q_out_op		  = params.values.template get_core<mega_qkv_prep_and_cache_publish_types, mega_qkv_prep_and_cache_publish_types::q_out>();
			auto& weights_core	  = get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core	  = get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_core = get_adjacent_value<core_traits_type::config, core_types::token_embeddings>::impl(params);
			auto& inp_embd_op	  = token_embd_core.values.template get_core<token_embeddings_types, token_embeddings_types::get_rows>();
			auto& attn_norm_w_op  = weights_core.values.template get_core<weight_types, weight_types::attn_norm>();
			auto& attn_q_w_op	  = weights_core.values.template get_core<weight_types, weight_types::attn_q>();
			auto& attn_k_w_op	  = weights_core.values.template get_core<weight_types, weight_types::attn_k>();
			auto& attn_v_w_op	  = weights_core.values.template get_core<weight_types, weight_types::attn_v>();
			auto& inp_pos_op	  = inputs_core.values.template get_core<global_input_types, global_input_types::inp_pos>();
			auto& rope_freqs_op	  = weights_core.values.template get_core<weight_types, weight_types::rope_freqs>();
			auto& cache_k_op	  = inputs_core.values.template get_core<global_input_types, global_input_types::cache_k>();
			auto& cache_v_op	  = inputs_core.values.template get_core<global_input_types, global_input_types::cache_v>();

			const uint64_t sequence_length = q_out_op.get_mutable_dim();

			static constexpr uint64_t rope_dim	= core_traits_type::mt::rope_dimension_count;
			static constexpr uint64_t n_head	= core_traits_type::mt::attention_head_count;
			static constexpr uint64_t n_head_kv = core_traits_type::mt::attention_head_count_kv;

			cuda_launch_params launch_params  = calculate_gpu_launch_params<typename core_traits_type::q_out_type, core_types::mega_qkv_prep_and_cache_publish>(q_out_op);
			const uint64_t max_threads_needed = n_head * rope_dim + n_head_kv * rope_dim * 2;
			const uint64_t actual_blocks = detail::min(launch_params.blocks_per_grid, (max_threads_needed + launch_params.threads_per_block - 1) / launch_params.threads_per_block);

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			mega_qkv_prep_and_cache_publish_eval_time<core_traits_type><<<actual_blocks, launch_params.threads_per_block>>>(sequence_length, params.data_ptrs);

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			cudaDeviceSynchronize();

			if constexpr (config.dev ) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
		}
	};

}
#endif
