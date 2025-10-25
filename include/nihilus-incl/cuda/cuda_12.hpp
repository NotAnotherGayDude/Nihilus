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

#if NIHILUS_COMPILER_CUDA

	#pragma once

	#include <nihilus-incl/infra/nihilus_cathedral.hpp>
	#include <nihilus-incl/common/common.hpp>
	#include <nihilus-incl/common/utility.hpp>
	#include <nihilus-incl/common/type_traits.hpp>
	#include <nihilus-incl/infra/core_traits.hpp>
	#include <cuda_runtime.h>
	#include <cuda_fp16.h>

namespace nihilus {

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

	enum class get_value_type_errors {
		invalid_type,
	};

	template<typename value_type> struct get_value {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) = delete;
	};

	template<int8_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_char1(detail::forward<value_types>(args)...);
		}
	};

	template<int8_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_char2(detail::forward<value_types>(args)...);
		}
	};

	template<int8_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_char3(detail::forward<value_types>(args)...);
		}
	};

	template<int8_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_char4(detail::forward<value_types>(args)...);
		}
	};

	template<int16_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_short1(detail::forward<value_types>(args)...);
		}
	};

	template<int16_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_short2(detail::forward<value_types>(args)...);
		}
	};

	template<int16_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_short3(detail::forward<value_types>(args)...);
		}
	};

	template<int16_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_short4(detail::forward<value_types>(args)...);
		}
	};

	template<int32_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_int1(detail::forward<value_types>(args)...);
		}
	};

	template<int32_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_int2(detail::forward<value_types>(args)...);
		}
	};

	template<int32_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_int3(detail::forward<value_types>(args)...);
		}
	};

	template<int32_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_int4(detail::forward<value_types>(args)...);
		}
	};

	template<int64_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_long1(detail::forward<value_types>(args)...);
		}
	};

	template<int64_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_long2(detail::forward<value_types>(args)...);
		}
	};

	template<int64_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_long3(detail::forward<value_types>(args)...);
		}
	};

	template<int64_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_long4(detail::forward<value_types>(args)...);
		}
	};

	template<uint8_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uchar1(detail::forward<value_types>(args)...);
		}
	};

	template<uint8_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uchar2(detail::forward<value_types>(args)...);
		}
	};

	template<uint8_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uchar3(detail::forward<value_types>(args)...);
		}
	};

	template<uint8_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uchar4(detail::forward<value_types>(args)...);
		}
	};

	template<uint16_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ushort1(detail::forward<value_types>(args)...);
		}
	};

	template<uint16_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ushort2(detail::forward<value_types>(args)...);
		}
	};

	template<uint16_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ushort3(detail::forward<value_types>(args)...);
		}
	};

	template<uint16_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ushort4(detail::forward<value_types>(args)...);
		}
	};

	template<uint32_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uint1(detail::forward<value_types>(args)...);
		}
	};

	template<uint32_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uint2(detail::forward<value_types>(args)...);
		}
	};

	template<uint32_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uint3(detail::forward<value_types>(args)...);
		}
	};

	template<uint32_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_uint4(detail::forward<value_types>(args)...);
		}
	};

	template<uint64_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ulong1(detail::forward<value_types>(args)...);
		}
	};

	template<uint64_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ulong2(detail::forward<value_types>(args)...);
		}
	};

	template<uint64_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ulong3(detail::forward<value_types>(args)...);
		}
	};

	template<uint64_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_ulong4(detail::forward<value_types>(args)...);
		}
	};

	template<float32_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_float1(detail::forward<value_types>(args)...);
		}
	};

	template<float32_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_float2(detail::forward<value_types>(args)...);
		}
	};

	template<float32_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_float3(detail::forward<value_types>(args)...);
		}
	};

	template<float32_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_float4(detail::forward<value_types>(args)...);
		}
	};

	template<float64_cuda_types value_type>
		requires(dim01_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_double1(detail::forward<value_types>(args)...);
		}
	};

	template<float64_cuda_types value_type>
		requires(dim02_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_double2(detail::forward<value_types>(args)...);
		}
	};

	template<float64_cuda_types value_type>
		requires(dim03_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_double3(detail::forward<value_types>(args)...);
		}
	};

	template<float64_cuda_types value_type>
		requires(dim04_types<value_type>)
	struct get_value<value_type> {
		template<typename... value_types> NIHILUS_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
			return make_double4(detail::forward<value_types>(args)...);
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
			return detail::forward<value_type01>(val01) + static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 += static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}
	};

	template<> struct binary_op_core<binary_op_types::mul> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return detail::forward<value_type01>(val01) * static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 *= static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}
	};

	template<> struct binary_op_core<binary_op_types::sub> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return detail::forward<value_type01>(val01) - static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 -= static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}
	};

	template<> struct binary_op_core<binary_op_types::div> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return detail::forward<value_type01>(val01) / static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			val01 /= static_cast<base_type<value_type01>>(detail::forward<value_type02>(val02));
		}
	};

	template<typename value_type, binary_op_types binary_op_type> struct binary_op_base;

	template<dim01_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value<value_type01>::impl(op_core_type::impl(detail::forward<value_type01>(val01).x, detail::forward<value_type02>(val02).x));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, detail::forward<value_type02>(val02).x);
		}
	};

	template<dim02_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value<value_type01>::impl(op_core_type::impl(detail::forward<value_type01>(val01).x, detail::forward<value_type02>(val02).x),
				op_core_type::impl(detail::forward<value_type01>(val01).y, detail::forward<value_type02>(val02).y));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, detail::forward<value_type02>(val02).x);
			op_core_type::impl_in_place(val01.y, detail::forward<value_type02>(val02).y);
		}
	};

	template<dim03_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value<value_type01>::impl(op_core_type::impl(detail::forward<value_type01>(val01).x, detail::forward<value_type02>(val02).x),
				op_core_type::impl(detail::forward<value_type01>(val01).y, detail::forward<value_type02>(val02).y),
				op_core_type::impl(detail::forward<value_type01>(val01).z, detail::forward<value_type02>(val02).z));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, detail::forward<value_type02>(val02).x);
			op_core_type::impl_in_place(val01.y, detail::forward<value_type02>(val02).y);
			op_core_type::impl_in_place(val01.z, detail::forward<value_type02>(val02).z);
		}
	};

	template<dim04_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			return get_value<value_type01>::impl(op_core_type::impl(detail::forward<value_type01>(val01).x, detail::forward<value_type02>(val02).x),
				op_core_type::impl(detail::forward<value_type01>(val01).y, detail::forward<value_type02>(val02).y),
				op_core_type::impl(detail::forward<value_type01>(val01).z, detail::forward<value_type02>(val02).z),
				op_core_type::impl(detail::forward<value_type01>(val01).w, detail::forward<value_type02>(val02).w));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
			using op_core_type = binary_op_core<binary_op_type>;
			op_core_type::impl_in_place(val01.x, detail::forward<value_type02>(val02).x);
			op_core_type::impl_in_place(val01.y, detail::forward<value_type02>(val02).y);
			op_core_type::impl_in_place(val01.z, detail::forward<value_type02>(val02).z);
			op_core_type::impl_in_place(val01.w, detail::forward<value_type02>(val02).w);
		}
	};

	template<binary_op_types binary_op_type> struct binary_op {
		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
			return binary_op_base<value_type01, binary_op_type>::impl(detail::forward<value_type01>(val01), detail::forward<value_type02>(val02));
		}

		template<typename value_type01, typename value_type02> NIHILUS_DEVICE static decltype(auto) impl_in_place(value_type01& val01, value_type02&& val02) {
			return binary_op_base<value_type01, binary_op_type>::impl_in_place(val01, detail::forward<value_type02>(val02));
		}
	};

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator+=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::add>::impl_in_place(val01, detail::forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator+(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::add>::impl(detail::forward<value_type01>(val01), detail::forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator*=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::mul>::impl_in_place(val01, detail::forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator*(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::mul>::impl(detail::forward<value_type01>(val01), detail::forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator-=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::sub>::impl_in_place(val01, detail::forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator-(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::sub>::impl(detail::forward<value_type01>(val01), detail::forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator/=(value_type01& val01, value_type02&& val02) {
		return binary_op<binary_op_types::div>::impl_in_place(val01, detail::forward<value_type02>(val02));
	}

	template<dim_types value_type01, dim_types value_type02> NIHILUS_DEVICE decltype(auto) operator/(value_type01&& val01, value_type02&& val02) {
		return binary_op<binary_op_types::div>::impl(detail::forward<value_type01>(val01), detail::forward<value_type02>(val02));
	}

	template<typename core_traits_type>
		requires(core_traits_type::kernel_profile_type::type == kernel_type_profiles::q8_gqa)
	NIHILUS_GLOBAL void token_embeddings_prompt_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs_type params) {
		using weight_type  = const typename core_traits_type::kernel_profile_type::weight_type;
		using index_type   = const typename core_traits_type::kernel_profile_type::index_type;
		using compute_type = typename core_traits_type::kernel_profile_type::compute_type;

		static constexpr uint64_t embedding_length	   = core_traits_type::mtt::embedding_length;
		static constexpr uint64_t block_size		   = type_traits<typename core_traits_type::kernel_profile_type::weight_type>::block_size;
		static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

		weight_type* __restrict weight_data	 = params.template get_weight_data<weight_type>();
		index_type* __restrict token_ids	 = params.template get_token_data<index_type>();
		compute_type* __restrict output_data = params.template get_output_data<compute_type>();

		const uint64_t token_idx = blockIdx.x;

		if (token_idx >= sequence_length) {
			return;
		}

		index_type token_id				 = token_ids[token_idx];
		weight_type* __restrict src_row	 = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);
		compute_type* __restrict dst_row = output_data + (token_idx * embedding_length);

		const uint64_t thread_id		 = threadIdx.x;
		const uint64_t threads_per_block = blockDim.x;

		for (uint64_t block_idx = thread_id; block_idx < blocks_per_embedding; block_idx += threads_per_block) {
			const auto& block				 = src_row[block_idx];
			const auto scale				 = __half2float(block.d);
			const auto* __restrict quantized = block.qs;
			const uint64_t base_offset		 = block_idx * block_size;

	#pragma unroll
			for (uint64_t j = 0; j < block_size; ++j) {
				if (base_offset + j < embedding_length) {
					dst_row[base_offset + j] = scale * static_cast<compute_type>(quantized[j]);
				}
			}
		}
	}

	template<typename core_traits_type>
		requires(core_traits_type::kernel_profile_type::type == kernel_type_profiles::fp16_mha)
	NIHILUS_GLOBAL void token_embeddings_prompt_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs_type params) {
		using weight_type  = const typename core_traits_type::kernel_profile_type::weight_type;
		using index_type   = const typename core_traits_type::kernel_profile_type::index_type;
		using compute_type = typename core_traits_type::kernel_profile_type::compute_type;

		static constexpr uint64_t embedding_length	   = core_traits_type::mtt::embedding_length;
		static constexpr uint64_t block_size		   = type_traits<typename core_traits_type::kernel_profile_type::weight_type>::block_size;
		static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

		weight_type* __restrict weight_data	 = params.template get_weight_data<weight_type>();
		index_type* __restrict token_ids	 = params.template get_token_data<index_type>();
		compute_type* __restrict output_data = params.template get_output_data<compute_type>();

		const uint64_t token_idx = blockIdx.x;

		if (token_idx >= sequence_length) {
			return;
		}

		index_type token_id				 = token_ids[token_idx];
		weight_type* __restrict src_row	 = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);
		compute_type* __restrict dst_row = output_data + (token_idx * embedding_length);

		const uint64_t thread_id		 = threadIdx.x;
		const uint64_t threads_per_block = blockDim.x;

		for (uint64_t block_idx = thread_id; block_idx < blocks_per_embedding; block_idx += threads_per_block) {
			const uint64_t base_offset = block_idx * block_size;
			dst_row[base_offset]	   = src_row[block_idx];
		}
	}

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
			auto& get_rows_op = params.values.template get_core<token_embeddings_types::get_rows>();

			const uint64_t sequence_length = get_rows_op.get_seq_length_dim();
			auto& weights_core			   = get_adjacent_value<typename core_traits_type::config_type, core_types::weights>::impl(params);
			auto& inputs_core			   = get_adjacent_value<typename core_traits_type::config_type, core_types::global_inputs>::impl(params);
			auto& token_embd_op			   = weights_core.values.template get_core<weight_types::token_embd>();
			auto& inp_tokens_op			   = inputs_core.values.template get_core<global_input_types::inp_tokens>();

			params.data_ptrs.set_ptrs(get_rows_op.get_data(), token_embd_op.get_data(), inp_tokens_op.get_data());

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			if (sequence_length > 0) {
				static constexpr uint64_t embedding_length	   = core_traits_type::mtt::embedding_length;
				static constexpr uint64_t block_size		   = type_traits<typename core_traits_type::kernel_profile_type::weight_type>::block_size;
				static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

				constexpr uint64_t threads_per_block = blocks_per_embedding <= 32 ? 32
					: blocks_per_embedding <= 64								  ? 64
					: blocks_per_embedding <= 128								  ? 128
					: blocks_per_embedding <= 256								  ? 256
																				  : 512;

				const uint64_t blocks_per_grid = sequence_length;

				token_embeddings_prompt_eval_time<core_traits_type><<<blocks_per_grid, threads_per_block>>>(sequence_length, params.data_ptrs);

				if constexpr (config_type::dev) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config_type::exceptions, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
					}
				}

				cudaDeviceSynchronize();

				if constexpr (config_type::dev) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config_type::exceptions, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
					}
				}
			}
		}
	};

	template<typename core_traits_type>
		requires(core_traits_type::kernel_profile_type::type == kernel_type_profiles::q8_gqa)
	NIHILUS_GLOBAL void token_embeddings_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs_type params) {
		using weight_type  = half;
		using index_type   = uint32_t;
		using compute_type = half;

		static constexpr uint64_t embedding_length	   = core_traits_type::mtt::embedding_length;
		static constexpr uint64_t block_size		   = type_traits<typename core_traits_type::kernel_profile_type::weight_type>::block_size;
		static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

		const block_q8_0<half>* __restrict weight_data = params.template get_weight_data<block_q8_0<half>>();
		const uint32_t* __restrict token_ids		   = params.template get_token_data<uint32_t>();
		float* __restrict output_data				   = params.template get_output_data<float>();

		const uint64_t thread_id		 = threadIdx.x;
		const uint64_t threads_per_block = blockDim.x;

		for (uint64_t block_idx = thread_id; block_idx < blocks_per_embedding; block_idx += threads_per_block) {
			const auto& block				 = weight_data[block_idx];
			const auto scale				 = __half2float(block.d);
			const auto* __restrict quantized = block.qs;
			const uint64_t base_offset		 = block_idx * block_size;

	#pragma unroll
			for (uint64_t j = 0; j < block_size; ++j) {
				if (base_offset + j < embedding_length) {
					output_data[base_offset + j] = scale * static_cast<compute_type>(quantized[j]);
				}
			}
		}
	}

	template<typename core_traits_type>
		requires(core_traits_type::kernel_profile_type::type == kernel_type_profiles::fp16_mha)
	NIHILUS_GLOBAL void token_embeddings_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs_type params) {
		using weight_type  = half;
		using index_type   = uint32_t;
		using compute_type = half;

		static constexpr uint64_t embedding_length = core_traits_type::mtt::embedding_length;

		const half* __restrict weight_data	 = params.template get_weight_data<half>();
		const uint32_t* __restrict token_ids = params.template get_token_data<uint32_t>();
		half* __restrict output_data		 = params.template get_output_data<half>();

		const uint64_t thread_id		 = threadIdx.x;
		const uint64_t threads_per_block = blockDim.x;

		constexpr uint64_t elems_per_vec = sizeof(float4) / sizeof(compute_type);
		const uint64_t vec_length		 = embedding_length / elems_per_vec;

		auto* __restrict vec_weights = weight_data;
		auto* __restrict vec_output	 = output_data;

		for (uint64_t i = thread_id; i < vec_length; i += threads_per_block) {
			vec_output[i] = vec_weights[i];
		}

		const uint64_t remainder_start = vec_length * elems_per_vec;
		for (uint64_t i = remainder_start + thread_id; i < embedding_length; i += threads_per_block) {
			output_data[i] = weight_data[i];
		}
	}

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
			auto& get_rows_op			   = params.values.template get_core<token_embeddings_types::get_rows>();
			auto& weights_core			   = get_adjacent_value<typename core_traits_type::config_type, core_types::weights>::impl(params);
			auto& inputs_core			   = get_adjacent_value<typename core_traits_type::config_type, core_types::global_inputs>::impl(params);
			auto& token_embd_op			   = weights_core.values.template get_core<weight_types::token_embd>();
			auto& inp_tokens_op			   = inputs_core.values.template get_core<global_input_types::inp_tokens>();
			const uint64_t sequence_length = get_rows_op.get_seq_length_dim();

			params.data_ptrs.set_ptrs(get_rows_op.get_data(), token_embd_op.get_data(), inp_tokens_op.get_data());

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			static constexpr uint64_t embedding_length	   = core_traits_type::mtt::embedding_length;
			static constexpr uint64_t block_size		   = type_traits<typename core_traits_type::kernel_profile_type::weight_type>::block_size;
			static constexpr uint64_t blocks_per_embedding = (embedding_length + block_size - 1) / block_size;

			constexpr uint64_t threads_per_block = blocks_per_embedding <= 64 ? 64 : blocks_per_embedding <= 128 ? 128 : blocks_per_embedding <= 256 ? 256 : 512;

			const uint64_t blocks_per_grid = 1;

			token_embeddings_eval_time<core_traits_type><<<blocks_per_grid, threads_per_block>>>(sequence_length, params.data_ptrs);

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			cudaDeviceSynchronize();

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
				}
			}
		}
	};

	template<typename core_traits_type>
	NIHILUS_GLOBAL void mega_qkv_prep_and_cache_publish_prompt_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs_type params) {
	}

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
			auto& q_out_op		  = params.values.template get_core<mega_qkv_prep_and_cache_publish_types::q_out>();
			auto& weights_core	  = get_adjacent_value<typename core_traits_type::config_type, core_types::weights>::impl(params);
			auto& inputs_core	  = get_adjacent_value<typename core_traits_type::config_type, core_types::global_inputs>::impl(params);
			auto& token_embd_core = get_adjacent_value<typename core_traits_type::config_type, core_types::token_embeddings>::impl(params);
			auto& inp_embd_op	  = token_embd_core.values.template get_core<token_embeddings_types::get_rows>();
			auto& attn_norm_w_op  = weights_core.values.template get_core<weight_types::attn_norm>();
			auto& attn_q_w_op	  = weights_core.values.template get_core<weight_types::attn_q>();
			auto& attn_k_w_op	  = weights_core.values.template get_core<weight_types::attn_k>();
			auto& attn_v_w_op	  = weights_core.values.template get_core<weight_types::attn_v>();
			auto& inp_pos_op	  = inputs_core.values.template get_core<global_input_types::inp_pos>();
			auto& rope_freqs_op	  = weights_core.values.template get_core<weight_types::rope_freqs>();
			auto& cache_k_op	  = inputs_core.values.template get_core<global_input_types::cache_k>();
			auto& cache_v_op	  = inputs_core.values.template get_core<global_input_types::cache_v>();

			const uint64_t sequence_length = q_out_op.get_seq_length_dim();

			cuda_launch_params launch_params = calculate_gpu_launch_params<typename core_traits_type::q_out_type, core_types::mega_qkv_prep_and_cache_publish>(q_out_op);

			const uint64_t max_threads_needed = sequence_length;
			const uint64_t actual_blocks = detail::min(launch_params.blocks_per_grid, (max_threads_needed + launch_params.threads_per_block - 1) / launch_params.threads_per_block);

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			if (sequence_length > 0) {
				//mega_qkv_prep_and_cache_publish_prompt_eval_time<core_traits_type><<<actual_blocks, launch_params.threads_per_block>>>(sequence_length, params.data_ptrs);

				if constexpr (config_type::dev) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config_type::exceptions, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
					}
				}

				cudaDeviceSynchronize();

				if constexpr (config_type::dev) {
					if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config_type::exceptions, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
					}
				}
			}
		}
	};

	template<typename core_traits_type>
	NIHILUS_GLOBAL void mega_qkv_prep_and_cache_publish_eval_time(uint64_t sequence_length, typename core_traits_type::kernel_data_ptrs_type params) {
	}

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
			auto& q_out_op		  = params.values.template get_core<mega_qkv_prep_and_cache_publish_types::q_out>();
			auto& weights_core	  = get_adjacent_value<typename core_traits_type::config_type, core_types::weights>::impl(params);
			auto& inputs_core	  = get_adjacent_value<typename core_traits_type::config_type, core_types::global_inputs>::impl(params);
			auto& token_embd_core = get_adjacent_value<typename core_traits_type::config_type, core_types::token_embeddings>::impl(params);
			auto& inp_embd_op	  = token_embd_core.values.template get_core<token_embeddings_types::get_rows>();
			auto& attn_norm_w_op  = weights_core.values.template get_core<weight_types::attn_norm>();
			auto& attn_q_w_op	  = weights_core.values.template get_core<weight_types::attn_q>();
			auto& attn_k_w_op	  = weights_core.values.template get_core<weight_types::attn_k>();
			auto& attn_v_w_op	  = weights_core.values.template get_core<weight_types::attn_v>();
			auto& inp_pos_op	  = inputs_core.values.template get_core<global_input_types::inp_pos>();
			auto& rope_freqs_op	  = weights_core.values.template get_core<weight_types::rope_freqs>();
			auto& cache_k_op	  = inputs_core.values.template get_core<global_input_types::cache_k>();
			auto& cache_v_op	  = inputs_core.values.template get_core<global_input_types::cache_v>();

			const uint64_t sequence_length = q_out_op.get_seq_length_dim();

			static constexpr uint64_t rope_dim	= core_traits_type::mtt::rope_dimension_count;
			static constexpr uint64_t n_head	= core_traits_type::mtt::attention_head_count;
			static constexpr uint64_t n_head_kv = core_traits_type::mtt::attention_head_count_kv;

			cuda_launch_params launch_params  = calculate_gpu_launch_params<typename core_traits_type::q_out_type, core_types::mega_qkv_prep_and_cache_publish>(q_out_op);
			const uint64_t max_threads_needed = n_head * rope_dim + n_head_kv * rope_dim * 2;
			const uint64_t actual_blocks = detail::min(launch_params.blocks_per_grid, (max_threads_needed + launch_params.threads_per_block - 1) / launch_params.threads_per_block);

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			//mega_qkv_prep_and_cache_publish_eval_time<core_traits_type><<<actual_blocks, launch_params.threads_per_block>>>(sequence_length, params.data_ptrs);

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Kernel Launch Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			cudaDeviceSynchronize();

			if constexpr (config_type::dev) {
				if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Synchronization Error: ", location>::impl(cudaGetErrorString(err));
				}
			}
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void impl(core_traits_type& params) {
		}
	};

}
#endif
