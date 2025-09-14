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

#pragma once

#if NIHILUS_CUDA_ENABLED

	#include <nihilus-incl/infra/core_bases.hpp>
	#include <nihilus-incl/common/common.hpp>
	#include <nihilus-incl/common/utility.hpp>
	#include <nihilus-incl/common/type_traits.hpp>
	#include <nihilus-incl/infra/core_traits.hpp>
	#include <cuda_runtime.h>
	#include <cuda_fp16.h>

namespace nihilus {

	template<core_types kernel_type> struct kernel_scaling_factors {
		static constexpr float memory_bound_factor	= 1.0f;
		static constexpr float compute_bound_factor = 1.0f;
	};

	template<typename output_type, core_types kernel_type> NIHILUS_INLINE static void calculate_gpu_launch_params(output_type& output, dim3& grid_dims, dim3& block_dims) {
		const auto dims					= output.get_array_rt();
		const uint64_t total_elems		= dims[0] * dims[1] * dims[2] * dims[3];
		constexpr uint64_t element_size = sizeof(typename output_type::output_type);
		const uint64_t total_bytes		= total_elems * element_size;

		constexpr auto memory_bound_factor	= kernel_scaling_factors<kernel_type>::memory_bound_factor;
		constexpr auto compute_bound_factor = kernel_scaling_factors<kernel_type>::compute_bound_factor;

		constexpr uint64_t base_block_size		 = gpu_properties::optimal_block_size;
		constexpr uint64_t warp_size			 = gpu_properties::warp_size;
		constexpr uint64_t max_threads_per_block = gpu_properties::max_threads_per_block;
		constexpr uint64_t sm_count				 = gpu_properties::sm_count;
		constexpr uint64_t max_threads_per_sm	 = gpu_properties::max_threads_per_sm;
		constexpr uint64_t l2_cache_size		 = gpu_properties::l2_cache_size;
		uint64_t optimal_block_size;
		if constexpr (memory_bound_factor > compute_bound_factor) {
			const uint64_t memory_optimal_size = (total_bytes <= l2_cache_size) ? detail::min(base_block_size * 2, max_threads_per_block) : base_block_size;
			optimal_block_size				   = ((memory_optimal_size + warp_size - 1) / warp_size) * warp_size;
		} else if constexpr (compute_bound_factor > memory_bound_factor) {
			const uint64_t occupancy_optimal_size = detail::min(max_threads_per_sm / 4, max_threads_per_block);
			optimal_block_size					  = ((occupancy_optimal_size + warp_size - 1) / warp_size) * warp_size;
		} else {
			optimal_block_size = base_block_size;
		}
		optimal_block_size			 = detail::max(warp_size, detail::min(optimal_block_size, max_threads_per_block));
		const uint64_t blocks_needed = (total_elems + optimal_block_size - 1) / optimal_block_size;
		uint64_t optimal_grid_size;
		if (blocks_needed > sm_count * 4) {
			optimal_grid_size = detail::min(blocks_needed, static_cast<uint64_t>(gpu_properties::max_grid_size_x));
		} else {
			optimal_grid_size = detail::min(blocks_needed, sm_count * 2);
		}
		if (dims[0] > 1 && dims[1] > 1) {
			const uint64_t dim0_blocks = (dims[0] + optimal_block_size - 1) / optimal_block_size;
			const uint64_t dim1_blocks = (dims[1] + optimal_block_size - 1) / optimal_block_size;

			if (dim0_blocks * dim1_blocks <= optimal_grid_size) {
				grid_dims.x = static_cast<unsigned int>(dim0_blocks);
				grid_dims.y = static_cast<unsigned int>(dim1_blocks);
				grid_dims.z = 1;

				const uint64_t block_dim_x		 = detail::min(dims[0], optimal_block_size);
				const uint64_t remaining_threads = optimal_block_size / block_dim_x;
				const uint64_t block_dim_y		 = detail::min(dims[1], remaining_threads);

				block_dims.x = static_cast<unsigned int>(block_dim_x);
				block_dims.y = static_cast<unsigned int>(block_dim_y);
				block_dims.z = 1;
				return;
			}
		}
		grid_dims.x = static_cast<unsigned int>(optimal_grid_size);
		grid_dims.y = 1;
		grid_dims.z = 1;

		block_dims.x = static_cast<unsigned int>(optimal_block_size);
		block_dims.y = 1;
		block_dims.z = 1;
	}

	template<typename value_type> struct get_value_type {
		NIHILUS_INLINE static constexpr auto impl() {
			if constexpr (int8_types<value_type>) {
				if constexpr (dim01_types<value_type>) {
					return &make_char1;
				} else if constexpr (dim02_types<value_type>) {
					return &make_char2;
				} else if constexpr (dim03_types<value_type>) {
					return &make_char3;
				} else if constexpr (dim04_types<value_type>) {
					return &make_char4;
				}
			} else if (int16_types<value_type>) {
				if constexpr (dim01_types<value_type>) {
					return &make_short1;
				} else if constexpr (dim02_types<value_type>) {
					return &make_short2;
				} else if constexpr (dim03_types<value_type>) {
					return &make_short3;
				} else if constexpr (dim04_types<value_type>) {
					return &make_short4;
				}
			} else if (int32_types<value_type>) {
				if constexpr (dim01_types<value_type>) {
					return &make_int1;
				} else if constexpr (dim02_types<value_type>) {
					return &make_int2;
				} else if constexpr (dim03_types<value_type>) {
					return &make_int3;
				} else if constexpr (dim04_types<value_type>) {
					return &make_int4;
				}
			} else if (float_types<value_type>) {
				if constexpr (dim01_types<value_type>) {
					return &make_float1;
				} else if constexpr (dim02_types<value_type>) {
					return &make_float2;
				} else if constexpr (dim03_types<value_type>) {
					return &make_float3;
				} else if constexpr (dim04_types<value_type>) {
					return &make_float4;
				}
			}
		}
		static constexpr auto type = impl();
	};

	struct add {
		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_one(value_type01 val01, value_type02 val02) {
			return val01 + val02;
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_two(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x + val02.x, val01.y + val02.y });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_three(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x + val02.x, val01.y + val02.y, val01.z + val02.z });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_four(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x + val02.x, val01.y + val02.y, val01.z + val02.z, val01.w + val02.w });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl(value_type01 val01, value_type02 val02) {
			if constexpr (dim04_types<value_type01>) {
				return impl_four(val01, val02);
			} else if constexpr (dim03_types<value_type01>) {
				return impl_three(val01, val02);
			} else if constexpr (dim02_types<value_type01>) {
				return impl_two(val01, val02);
			} else {
				return impl_one(val01, val02);
			}
		}
	};

	struct mul {
		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_one(value_type01 val01, value_type02 val02) {
			return val01 * val02;
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_two(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x * val02.x, val01.y * val02.y });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_three(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x * val02.x, val01.y * val02.y, val01.z * val02.z });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_four(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x * val02.x, val01.y * val02.y, val01.z * val02.z, val01.w * val02.w });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl(value_type01 val01, value_type02 val02) {
			if constexpr (dim04_types<value_type01>) {
				return impl_four(val01, val02);
			} else if constexpr (dim03_types<value_type01>) {
				return impl_three(val01, val02);
			} else if constexpr (dim02_types<value_type01>) {
				return impl_two(val01, val02);
			} else {
				return impl_one(val01, val02);
			}
		}
	};

	struct sub {
		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_one(value_type01 val01, value_type02 val02) {
			return val01 - val02;
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_two(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x - val02.x, val01.y - val02.y });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_three(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x - val02.x, val01.y - val02.y, val01.z - val02.z });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_four(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x - val02.x, val01.y - val02.y, val01.z - val02.z, val01.w - val02.w });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl(value_type01 val01, value_type02 val02) {
			if constexpr (dim04_types<value_type01>) {
				return impl_four(val01, val02);
			} else if constexpr (dim03_types<value_type01>) {
				return impl_three(val01, val02);
			} else if constexpr (dim02_types<value_type01>) {
				return impl_two(val01, val02);
			} else {
				return impl_one(val01, val02);
			}
		}
	};

	struct div {
		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_one(value_type01 val01, value_type02 val02) {
			return val01 / val02;
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_two(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x / val02.x, val01.y / val02.y });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_three(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x / val02.x, val01.y / val02.y, val01.z / val02.z });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl_four(value_type01 val01, value_type02 val02) {
			return get_value_type<value_type01>::type({ val01.x / val02.x, val01.y / val02.y, val01.z / val02.z, val01.w / val02.w });
		}

		template<typename value_type01, detail::convertible_to<value_type01> value_type02>
		NIHILUS_INLINE static __device__ value_type01 impl(value_type01 val01, value_type02 val02) {
			if constexpr (dim04_types<value_type01>) {
				return impl_four(val01, val02);
			} else if constexpr (dim03_types<value_type01>) {
				return impl_three(val01, val02);
			} else if constexpr (dim02_types<value_type01>) {
				return impl_two(val01, val02);
			} else {
				return impl_one(val01, val02);
			}
		}
	};

	template<enum_types enum_type, enum_type enum_value, typename core_bases_type> NIHILUS_INLINE __device__ decltype(auto) get_core(core_bases_type& core_bases) noexcept {
		return *static_cast<std::remove_cvref_t<typename core_bases_type::template core_base_type<enum_type, enum_value>>*>(&core_bases);
	}

	template<const model_config& config_new, auto kernel_type> NIHILUS_INLINE __device__ static decltype(auto) get_adjacent_value_device(auto& parse_core) {
		using derived_type	   = core_traits<config_new, kernel_type>;
		using thread_pool_type = thread_pool<config_new>;
		using core_bases_type  = typename thread_pool<config_new>::core_bases_type;
		return *static_cast<derived_type*>(static_cast<core_bases_type*>(&parse_core));
	}

	template<typename core_traits_type> __global__ void token_embeddings_prompt_eval_time(uint64_t sequence_length, core_traits_type& params) {
		static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
		static constexpr uint64_t blocks_per_embedding = embedding_length / 32;
		auto& get_rows_op							   = get_core<token_embedding_types, token_embedding_types::get_rows>(params.values);
		auto& weights_core							   = get_adjacent_value_device<core_traits_type::config, core_types::weights>(params);
		auto& inputs_core							   = get_adjacent_value_device<core_traits_type::config, core_types::global_inputs>(params);
		auto& token_embd_op							   = get_core<weight_types, weight_types::token_embd>(weights_core.values);
		auto& inp_tokens_op							   = get_core<global_input_types, global_input_types::inp_tokens>(inputs_core.values);
		const auto* __restrict__ weight_data		   = token_embd_op.data;
		const auto* __restrict__ token_ids			   = inp_tokens_op.data;
		auto* __restrict__ output_data				   = get_rows_op.data;

		const uint64_t global_tid	 = blockIdx.x * blockDim.x + threadIdx.x;
		const uint64_t total_threads = gridDim.x * blockDim.x;

		for (uint64_t token_idx = global_tid; token_idx < sequence_length; token_idx += total_threads) {
			const typename core_traits_type::prof::index_type token_id					= token_ids[token_idx];
			const typename core_traits_type::prof::weight_type* __restrict__ src_blocks = weight_data + (token_id * blocks_per_embedding);
			auto* __restrict__ dst_row													= output_data + (token_idx * embedding_length);

			for (uint64_t block_idx = 0; block_idx < blocks_per_embedding; ++block_idx) {
				const typename core_traits_type::prof::weight_type& block	  = src_blocks[block_idx];
				const typename core_traits_type::prof::compute_type scale_f32 = __half2float(block.d);
				const uint64_t base_elem									  = block_idx * 32;

	#pragma unroll
				for (uint64_t i = 0; i < 32; ++i) {
					dst_row[base_elem + i] = scale_f32 * static_cast<typename core_traits_type::prof::compute_type>(block.qs[i]);
				}
			}
		}
	}

	template<typename core_traits_type> __global__ void token_embeddings_prompt_eval_time_optimized(uint64_t sequence_length, core_traits_type& params) {
		static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
		static constexpr uint64_t blocks_per_embedding = embedding_length / 32;
		auto& get_rows_op							   = get_core<token_embedding_types, token_embedding_types::get_rows>(params.values);
		auto& weights_core							   = get_adjacent_value_device<core_traits_type::config, core_types::weights>(params);
		auto& inputs_core							   = get_adjacent_value_device<core_traits_type::config, core_types::global_inputs>(params);
		auto& token_embd_op							   = get_core<weight_types, weight_types::token_embd>(weights_core.values);
		auto& inp_tokens_op							   = get_core<global_input_types, global_input_types::inp_tokens>(inputs_core.values);
		const auto* __restrict__ weight_data		   = token_embd_op.data;
		const auto* __restrict__ token_ids			   = inp_tokens_op.data;
		auto* __restrict__ output_data				   = get_rows_op.data;

		extern __shared__ char shared_mem[];
		auto* shared_tokens = reinterpret_cast<typename core_traits_type::prof::index_type*>(shared_mem);

		const uint64_t block_token_start = blockIdx.x * blockDim.y;
		const uint64_t block_token_end	 = min(block_token_start + blockDim.y, sequence_length);
		const uint64_t tokens_in_block	 = block_token_end - block_token_start;

		if (threadIdx.x < tokens_in_block) {
			shared_tokens[threadIdx.x] = token_ids[block_token_start + threadIdx.x];
		}
		__syncthreads();

		const uint64_t warp_id		   = threadIdx.x / 32;
		const uint64_t lane_id		   = threadIdx.x % 32;
		const uint64_t warps_per_block = blockDim.x / 32;

		for (uint64_t token_offset = warp_id; token_offset < tokens_in_block; token_offset += warps_per_block) {
			const typename core_traits_type::prof::index_type token_idx = add::impl(block_token_start, token_offset);
			const typename core_traits_type::prof::index_type token_id	= shared_tokens[token_offset];

			const auto* __restrict__ src_blocks = weight_data + mul::impl(token_id, blocks_per_embedding);
			auto* __restrict__ dst_row			= output_data + mul::impl(token_idx, embedding_length);

			for (uint64_t block_idx = lane_id; block_idx < blocks_per_embedding; block_idx += 32) {
				const auto& block											  = src_blocks[block_idx];
				const typename core_traits_type::prof::compute_type scale_f32 = __half2float(block.d);
				auto* dst_ptr												  = dst_row + mul::impl(block_idx, 32);
	#pragma unroll
				for (uint64_t i = 0; i < 32; i += 4) {
					float4 result;
					result.x								= mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[i]));
					result.y								= mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[i + 1]));
					result.z								= mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[i + 2]));
					result.w								= mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[i + 3]));
					*reinterpret_cast<float4*>(&dst_ptr[i]) = result;
				}
			}
		}
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void impl(core_traits_type& params) {
			auto& get_rows_op	= params.values.template get_core<token_embedding_types, token_embedding_types::get_rows>();
			auto& weights_core	= get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core	= get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_op = weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op = inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();

			const auto* __restrict__ weight_data = token_embd_op.data;
			const auto* __restrict__ token_ids	 = inp_tokens_op.data;
			auto* __restrict__ output_data		 = get_rows_op.data;
			const uint64_t sequence_length		 = inp_tokens_op.get_mutable_dim();

			static constexpr model_arches model_arch{ core_traits_type::config.arch };
			static constexpr model_sizes model_size{ core_traits_type::config.model_size };
			static constexpr model_generations model_generation{ core_traits_type::config.model_generation };

			dim3 grid_dims{}, block_dims{};
			calculate_gpu_launch_params<typename core_traits_type::token_embeddings_type, core_types::token_embeddings>(params.values, grid_dims, block_dims);

			constexpr uint64_t embedding_length = model_traits<model_arch, model_size, model_generation>::embedding_length;

			if constexpr (embedding_length >= 1024) {
				const uint64_t tokens_per_block	 = min(static_cast<uint64_t>(16), sequence_length);
				const uint64_t threads_per_token = min(static_cast<uint64_t>(32), embedding_length / 32);

				block_dims.x = threads_per_token * 32;
				block_dims.y = 1;
				block_dims.z = 1;

				grid_dims.x = (sequence_length + tokens_per_block - 1) / tokens_per_block;
				grid_dims.y = 1;
				grid_dims.z = 1;

				const size_t shared_mem_size = tokens_per_block * sizeof(typename kernel_type_profile_traits<config.kernel_profile>::index_type);
				token_embeddings_prompt_eval_time_optimized<<<grid_dims, block_dims, shared_mem_size>>>(sequence_length, params);
			} else {
				token_embeddings_prompt_eval_time<<<grid_dims, block_dims>>>(sequence_length, params);
			}
			if constexpr (config.dev) {
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}
			cudaDeviceSynchronize();
		}
	};

	template<typename core_traits_type> __global__ void token_embeddings_eval_time(core_traits_type& params) {
		static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
		static constexpr uint64_t blocks_per_embedding = embedding_length / 32;
		auto& get_rows_op							   = get_core<token_embedding_types, token_embedding_types::get_rows>(params.values);
		auto& weights_core							   = get_adjacent_value_device<core_traits_type::config, core_types::weights>(params);
		auto& inputs_core							   = get_adjacent_value_device<core_traits_type::config, core_types::global_inputs>(params);
		auto& token_embd_op							   = get_core<weight_types, weight_types::token_embd>(weights_core.values);
		auto& inp_tokens_op							   = get_core<global_input_types, global_input_types::inp_tokens>(inputs_core.values);
		const auto* __restrict__ weight_data		   = token_embd_op.data;
		const auto* __restrict__ token_ids			   = inp_tokens_op.data;
		auto* __restrict__ output_data				   = get_rows_op.data;

		const typename core_traits_type::prof::index_type token_id					= token_ids[0];
		const typename core_traits_type::prof::weight_type* __restrict__ src_blocks = weight_data + (token_id * blocks_per_embedding);

		const uint64_t tid				 = blockIdx.x * blockDim.x + threadIdx.x;
		const uint64_t blocks_per_thread = (blocks_per_embedding + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
		const uint64_t start_block		 = tid * blocks_per_thread;
		const uint64_t end_block		 = min(start_block + blocks_per_thread, blocks_per_embedding);

		for (uint64_t block_idx = start_block; block_idx < end_block; ++block_idx) {
			const typename core_traits_type::prof::weight_type& block	  = src_blocks[block_idx];
			const typename core_traits_type::prof::compute_type scale_f32 = __half2float(block.d);
			const uint64_t base_elem									  = block_idx * 32;

	#pragma unroll
			for (uint64_t i = 0; i < 32; ++i) {
				output_data[base_elem + i] = mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[i]));
			}
		}
	}

	template<typename core_traits_type> __global__ void token_embeddings_eval_time_optimized(core_traits_type& params) {
		static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
		static constexpr uint64_t blocks_per_embedding = embedding_length / 32;
		auto& get_rows_op							   = get_core<token_embedding_types, token_embedding_types::get_rows>(params.values);
		auto& weights_core							   = get_adjacent_value_device<core_traits_type::config, core_types::weights>(params);
		auto& inputs_core							   = get_adjacent_value_device<core_traits_type::config, core_types::global_inputs>(params);
		auto& token_embd_op							   = get_core<weight_types, weight_types::token_embd>(weights_core.values);
		auto& inp_tokens_op							   = get_core<global_input_types, global_input_types::inp_tokens>(inputs_core.values);
		const auto* __restrict__ weight_data		   = token_embd_op.data;
		const auto* __restrict__ token_ids			   = inp_tokens_op.data;
		auto* __restrict__ output_data				   = get_rows_op.data;

		const typename core_traits_type::prof::index_type token_id					= token_ids[0];
		const typename core_traits_type::prof::weight_type* __restrict__ src_blocks = weight_data + (token_id * blocks_per_embedding);

		const uint64_t warp_id	   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
		const uint64_t lane_id	   = threadIdx.x % 32;
		const uint64_t total_warps = (blockDim.x * gridDim.x) / 32;

		for (uint64_t block_idx = warp_id; block_idx < blocks_per_embedding; block_idx += total_warps) {
			const typename core_traits_type::prof::weight_type& block	  = src_blocks[block_idx];
			const typename core_traits_type::prof::compute_type scale_f32 = __half2float(block.d);

			const uint64_t base_elem = block_idx * 32;

			if (lane_id < 8) {
				const uint64_t vec_idx = lane_id * 4;
				float4 result;
				result.x = mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[vec_idx]));
				result.y = mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[vec_idx + 1]));
				result.z = mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[vec_idx + 2]));
				result.w = mul::impl(scale_f32, static_cast<typename core_traits_type::prof::compute_type>(block.qs[vec_idx + 3]));
				*reinterpret_cast<float4*>(&output_data[base_elem + vec_idx]) = result;
			}
		}
	}

	template<typename core_traits_type> __global__ void token_embeddings_eval_time_single_block(core_traits_type& params) {
		static constexpr uint64_t embedding_length	   = core_traits_type::mt::embedding_length;
		static constexpr uint64_t blocks_per_embedding = embedding_length / 32;
		auto& get_rows_op							   = get_core<token_embedding_types, token_embedding_types::get_rows>(params.values);
		auto& weights_core							   = get_adjacent_value_device<core_traits_type::config, core_types::weights>(params);
		auto& inputs_core							   = get_adjacent_value_device<core_traits_type::config, core_types::global_inputs>(params);
		auto& token_embd_op							   = get_core<weight_types, weight_types::token_embd>(weights_core.values);
		auto& inp_tokens_op							   = get_core<global_input_types, global_input_types::inp_tokens>(inputs_core.values);

		const auto* __restrict__ weight_data = token_embd_op.data;
		const auto* __restrict__ token_ids	 = inp_tokens_op.data;
		auto* __restrict__ output_data		 = get_rows_op.data;
		extern __shared__ typename core_traits_type::prof::weight_type shared_blocks[];

		const typename core_traits_type::prof::index_type token_id					= token_ids[0];
		const typename core_traits_type::prof::weight_type* __restrict__ src_blocks = weight_data + (token_id * blocks_per_embedding);

		const uint64_t tid			  = threadIdx.x;
		const uint64_t blocks_to_load = (blocks_per_embedding + blockDim.x - 1) / blockDim.x;

		for (uint64_t i = 0; i < blocks_to_load; ++i) {
			const uint64_t block_idx = tid + i * blockDim.x;
			if (block_idx < blocks_per_embedding) {
				shared_blocks[block_idx] = src_blocks[block_idx];
			}
		}
		__syncthreads();

		const uint64_t elems_per_thread = (embedding_length + blockDim.x - 1) / blockDim.x;
		const uint64_t start_elem		= tid * elems_per_thread;
		const uint64_t end_elem			= min(start_elem + elems_per_thread, embedding_length);

		for (uint64_t elem_idx = start_elem; elem_idx < end_elem; ++elem_idx) {
			const uint64_t block_idx									  = elem_idx / 32;
			const uint64_t elem_in_block								  = elem_idx % 32;
			const typename core_traits_type::prof::weight_type& block	  = shared_blocks[block_idx];
			const typename core_traits_type::prof::compute_type scale_f32 = __half2float(block.d);
			output_data[elem_idx]										  = scale_f32 * static_cast<typename core_traits_type::prof::compute_type>(block.qs[elem_in_block]);
		}
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::eval_time> {
		NIHILUS_INLINE static void impl(core_traits_type& params) {
			auto& get_rows_op	= params.values.template get_core<token_embedding_types, token_embedding_types::get_rows>();
			auto& weights_core	= get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core	= get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_op = weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op = inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();

			static constexpr kernel_type_profiles kernel_type_profile{ core_traits_type::config.kernel_profile };
			static constexpr model_arches model_arch{ core_traits_type::config.arch };
			static constexpr model_sizes model_size{ core_traits_type::config.model_size };
			static constexpr model_generations model_generation{ core_traits_type::config.model_generation };

			constexpr uint64_t embedding_length		= model_traits<model_arch, model_size, model_generation>::embedding_length;
			constexpr uint64_t blocks_per_embedding = embedding_length / 32;

			if constexpr (embedding_length <= 512) {
				const uint64_t threads		 = min(static_cast<uint64_t>(256), embedding_length);
				const size_t shared_mem_size = blocks_per_embedding * sizeof(typename kernel_type_profile_traits<kernel_type_profile>::weight_type);

				token_embeddings_eval_time_single_block<<<1, threads, shared_mem_size>>>(params);
			} else if constexpr (embedding_length <= 2048) {
				const uint64_t threads = 256;
				const uint64_t blocks  = min(static_cast<uint64_t>(4), (blocks_per_embedding + 7) / 8);

				token_embeddings_eval_time<<<blocks, threads>>>(params);
			} else {
				const uint64_t threads = 256;
				const uint64_t blocks  = min(static_cast<uint64_t>(8), (blocks_per_embedding + 3) / 4);

				token_embeddings_eval_time_optimized<<<blocks, threads>>>(params);
			}
			if constexpr (config.dev) {
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}

			cudaDeviceSynchronize();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk, int64_t current_block) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk, int64_t current_block) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t current_block) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params) {
		}
	};

}
#endif