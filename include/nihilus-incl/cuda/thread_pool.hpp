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

#include <nihilus-incl/infra/monolithic_dispatcher.hpp>
#include <nihilus-incl/cpu/nihilus_cpu_properties.hpp>
#include <nihilus-incl/infra/core_bases.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <atomic>
#include <thread>
#include <latch>

namespace nihilus {

	template<const model_config& config>
		requires(config.device_type == device_types::gpu)
	struct thread_pool<config> : public get_core_bases_t<config, core_types>, public perf_base<config> {
		using core_bases_type											   = get_core_bases_t<config, core_types>;
		NIHILUS_INLINE thread_pool() noexcept							   = default;
		NIHILUS_INLINE thread_pool& operator=(const thread_pool&) noexcept = delete;
		NIHILUS_INLINE thread_pool(const thread_pool&) noexcept			   = delete;

		NIHILUS_INLINE thread_pool(int64_t) {}

		template<processing_phases processing_phase, size_t... indices> NIHILUS_INLINE void execute_blocks(std::index_sequence<indices...>) {
			(core_bases_type::template impl_thread<per_block_thread_function, processing_phase>(static_cast<int64_t>(indices), 1), ...);
		}

		template<processing_phases phase_new>
		NIHILUS_INLINE void thread_function() {
			core_bases_type::template impl_thread<global_input_thread_function, phase_new>(1);
			execute_blocks<phase_new>(std::make_index_sequence<static_cast<size_t>(model_traits_type<config>::block_count)>{});
			core_bases_type::template impl_thread<global_output_thread_function, phase_new>(1);
		}

		template<processing_phases phase_new> NIHILUS_INLINE void execute_tasks(uint64_t runtime_dimensions_new) {
			core_bases_type::template impl<sync_resetter>(1);
			core_bases_type::template impl<dim_updater>(runtime_dimensions_new);
			thread_function<phase_new>();
			if constexpr (config.dev) {
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}
		}

	  protected:
		NIHILUS_INLINE ~thread_pool(){};
	};


}
#endif
