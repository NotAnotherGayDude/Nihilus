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

#include <nihilus/common/arch_traits.hpp>
#include <nihilus/common/model_traits.hpp>
#include <nihilus/common/model_parser.hpp>
#include <nihilus/cpu/thread_pool.hpp>
#include <nihilus/common/h_params.hpp>
#include <nihilus/common/memory.hpp>
#include <nihilus/common/tuple.hpp>

namespace nihilus {

	template<typename model_generation_type, typename model_size_type> struct model_base {
		model_config<model_generation_type, model_size_type> config_new{};
		virtual void execute_model(execution_parameters& params) = 0;
		virtual void init(cli_params params)					 = 0;
		virtual ~model_base()									 = default;
	};

	template<model_config config> struct model : public model_base<decltype(config.model_size), decltype(config.model_generation)>,
												 public get_core_bases_config_base_t<config>,
												 public thread_pool<config, model<config>>,
												 public hyper_parameters<config.arch> {
		using core_bases_config_type		  = get_core_bases_config_base_t<config>;
		using model_traits_type				  = model_traits_type<config>;
		using op_type_type					  = model_traits_type::op_type_type;
		using kernel_type_profile_traits_type = kernel_type_profile_traits<config.kernel_profile>;
		using base_type						  = model_base<decltype(config.model_size), decltype(config.model_generation)>;
		template<typename model_type> friend struct input_session;
		static constexpr uint64_t total_required_bytes{ [] {
			uint64_t return_value{};
			core_bases_config_type::template impl_constexpr<memory_allocator>(return_value);
			return return_value;
		}() };
		NIHILUS_FORCE_INLINE model()						= default;
		NIHILUS_FORCE_INLINE model& operator=(model&&)		= delete;
		NIHILUS_FORCE_INLINE model(model&&)					= delete;
		NIHILUS_FORCE_INLINE model& operator=(const model&) = delete;
		NIHILUS_FORCE_INLINE model(const model&)			= delete;
		NIHILUS_FORCE_INLINE model(cli_params params) : thread_pool<config, model>{ params.thread_count } {
			init(params);
		}

		NIHILUS_FORCE_INLINE void init(cli_params params) {
			memory.init(total_required_bytes);
			std::cout << "Total required bytes: " << total_required_bytes << std::endl;
			weight_memory = memory_mapped_file{ params.model_file };
			thread_pool<config, model>::init();
			array<array<void*, model_traits_type::block_count>, op_type_type::count> data{};
			//core_bases_config_type::template impl<memory_mapper>(memory);
			//core_bases_config_type::template impl<execution_planner>(params.thread_count, data);
			stop_watch_val_nihilus.reset();
			//model_graph_data<config> model_construction_data = model_parser<config>::parse_model(data, &weight_memory);
			std::cout << "Nihilus model Load time: " << stop_watch_val_nihilus.total_time_elapsed() << std::endl;
			//core_bases_config_type::template impl<tensor_debugger_impl>();
		}

		NIHILUS_FORCE_INLINE void deinit(cli_params params) {
			memory.deinit();
		}

		template<op_type_type type> NIHILUS_FORCE_INLINE auto& get_core() {
			return *static_cast<core_traits<config, type>*>(this);
		}

		NIHILUS_FORCE_INLINE void execute_model(execution_parameters& params) {
			for (uint64_t x = 0; x < params.token_count + 1; ++x) {
				stop_watch_val_nihilus.reset();
				this->execute_tasks();
				stop_watch_val_nihilus.add_time();
			}
			// Perform all of the necessary stuff to execute the model - along with all of the constexpr values stored globally inside the class LOL!.
			// Because we only pay the "virtual overhead @ the top here == totally negligible.
		};

	  protected:
		memory_mapped_file weight_memory{};
		memory_buffer<config> memory{};
	};

}
