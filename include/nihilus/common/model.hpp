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

	template<typename model_generation_type, typename model_uint64_type> struct model_base {
		model_config<model_generation_type, model_uint64_type> config_new{};
		virtual void execute_model(execution_parameters& params) = 0;
		virtual void init(cli_params params)					 = 0;
		virtual ~model_base()									 = default;
	};

	template<nihilus::model_config config> struct model : public thread_pool<config, model<config>>,
														  public model_base<decltype(config.model_generation), decltype(config.model_size)> {
		using thread_pool_type		= thread_pool<config, model<config>>;
		using model_traits_type = nihilus::model_traits_type<config>;
		using core_bases_t		= get_core_bases_t<config>;
		using op_type_type		= typename model_traits_type ::op_type_type;
		using base_type			= model_base<decltype(config.model_generation), decltype(config.model_size)>;

		static constexpr uint64_t total_required_bytes{ []() {
			uint64_t return_value{};
			thread_pool_type ::template impl_static<memory_calculator>(return_value);
			return return_value;
		}() };
		template<auto op_type> auto& get_core() {
			return *static_cast<nihilus::core_traits<config, op_type>*>(static_cast<get_core_bases_t<config>*>(this));
		}

		NIHILUS_FORCE_INLINE model(nihilus::cli_params params) : thread_pool<config, model>{ params.thread_count } {
			init(params);
		}

		NIHILUS_FORCE_INLINE void init(nihilus::cli_params params) {
			memory.init(total_required_bytes);
			weight_memory = nihilus::memory_mapped_file{ params.model_file };
			nihilus::array<nihilus::array<void*, model_traits_type::block_count>, op_type_type::count> data{};
			this->template impl<weight_mapper>(data);
			this->template impl<memory_mapper>(memory);
			nihilus::stop_watch_val_nihilus.reset();
			nihilus::model_graph_data<config> model_construction_data = nihilus::model_parser<config>::parse_model(data, &weight_memory);
			std::cout << "Nihilus model Load time: " << nihilus::stop_watch_val_nihilus.total_time_elapsed() << std::endl;
			this->template impl<tensor_debugger_impl>();
		}

		NIHILUS_FORCE_INLINE void deinit(nihilus::cli_params params) {
			memory.deinit();
		}

		NIHILUS_FORCE_INLINE void execute_model(nihilus::execution_parameters& params) {
			for (uint64_t x = 0; x < params.token_count + 1; ++x) {
				nihilus::stop_watch_val_nihilus.reset();
				static_cast<thread_pool<config, model>*>(this)->execute_tasks();
				nihilus::stop_watch_val_nihilus.add_time();
			}
			// Perform all of the necessary stuff to execute the model - along with all of the constexpr values stored globally inside the class LOL!.
			// Because we only pay the "virtual overhead @ the top here == totally negligible.
		};

	  protected:
		nihilus::memory_mapped_file weight_memory{};
		nihilus::memory_buffer<config> memory{};
	};

}
