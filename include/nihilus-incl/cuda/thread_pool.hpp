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

#if NIHILUS_COMPILER_CUDA

	#include <nihilus-incl/infra/monolithic_dispatcher.hpp>
	#include <nihilus-incl/cpu/nihilus_cpu_properties.hpp>
	#include <nihilus-incl/infra/nihilus_cathedral.hpp>
	#include <nihilus-incl/common/common.hpp>
	#include <nihilus-incl/common/tuple.hpp>
	#include <atomic>
	#include <thread>
	#include <latch>

namespace nihilus {

	template<gpu_device_config_types config_type> struct thread_pool<config_type> : public get_nihilus_cathedral_t<config_type>, public perf_base<config_type> {
		using nihilus_cathedral_type = get_nihilus_cathedral_t<config_type>;
		NIHILUS_HOST thread_pool() noexcept {
		}
		NIHILUS_HOST thread_pool& operator=(const thread_pool&) noexcept = delete;
		NIHILUS_HOST thread_pool(const thread_pool&) noexcept			 = delete;

		NIHILUS_HOST thread_pool(int64_t) {
		}

		template<processing_phases processing_phase, size_t... indices> NIHILUS_HOST void execute_blocks(std::index_sequence<indices...>) {
			(nihilus_cathedral_type::template impl_thread<per_block_thread_function, processing_phase>(static_cast<int64_t>(indices)), ...);
		}

		template<processing_phases phase_new> NIHILUS_HOST void thread_function() {
			nihilus_cathedral_type::template impl_thread<global_input_thread_function, phase_new>();
			execute_blocks<phase_new>(std::make_index_sequence<static_cast<uint64_t>(model_traits_type<config_type>::block_count)>{});
			nihilus_cathedral_type::template impl_thread<global_output_thread_function, phase_new>();
		}

		template<processing_phases phase_new> NIHILUS_HOST void execute_tasks(uint64_t sequence_length, uint64_t batch_size) {
			nihilus_cathedral_type::template impl<dim_updater>(sequence_length);
			nihilus_cathedral_type::template impl<batched_dim_updater>(sequence_length, batch_size);
			thread_function<phase_new>();
			if constexpr (config_type::dev) {
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config_type::exceptions, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
				}
			}
		}

	  protected:
		NIHILUS_HOST ~thread_pool(){};
	};


}
#endif
