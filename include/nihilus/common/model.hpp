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

#include <nihilus/common/model_traits.hpp>
#include <nihilus/common/model_parser.hpp>
#include <nihilus/common/input_session.hpp>
#include <nihilus/common/core_bases.hpp>
#include <nihilus/cpu/thread_pool.hpp>
#include <nihilus/common/tuple.hpp>

namespace nihilus {

	struct model_base {
		NIHILUS_FORCE_INLINE model_base() noexcept = default;
		NIHILUS_FORCE_INLINE model_base(model_config config_new) : config{ config_new } {};
		model_config config{};
		virtual void execute_model(execution_parameters& params) = 0;
		virtual bool process_input(const std::string& params)	 = 0;
		virtual void init(cli_params params)					 = 0;
		virtual ~model_base()									 = default;
	};

	template<model_config config_new>
	struct model : public thread_pool<config_new, model<config_new>>, public model_base, public input_session<config_new, model<config_new>>, core_bases_traits<config_new> {
		using thread_pool_type	= thread_pool<config_new, model<config_new>>;
		using model_traits_type = nihilus::model_traits_type<config_new>;
		using core_bases_t		= get_core_bases_t<config_new>;
		using core_bases_traits = core_bases_traits<config_new>;
		using op_type_type		= typename model_traits_type ::op_type_type;
		using tokenizer_type	= tokenizer<config_new, model, config_new.arch, config_new.tokenizer_type>;
		using base_type			= model_base;
		template<auto op_type> auto& get_core() {
			return *static_cast<nihilus::core_traits<config_new, op_type>*>(static_cast<get_core_bases_t<config_new>*>(this));
		}

		NIHILUS_FORCE_INLINE model() noexcept = default;

		NIHILUS_FORCE_INLINE model(nihilus::cli_params params)
			: thread_pool<config_new, model>{ params.thread_count }, model_base{ config_new }, input_session<config_new, model<config_new>>{ params } {
			init(params);
		}

		model& operator=(const model&) = delete;
		model(const model&)			   = delete;

		NIHILUS_FORCE_INLINE bool process_input(const std::string& input) {
			return input_session<config_new, model>::process_input_impl(input);
		}

		NIHILUS_FORCE_INLINE void init(nihilus::cli_params params) {
			memory.init(core_bases_traits::total_required_bytes);
			weight_memory = nihilus::memory_mapped_file{ params.model_file };
			nihilus::array<nihilus::array<void*, model_traits_type::block_count>, op_types::count> data{};
			this->template impl<weight_mapper>(data);
			this->template impl<memory_mapper>(memory);
			nihilus::stop_watch_val_nihilus.reset();
			gguf_metadata<config_new.arch, config_new.tokenizer_type, config_new.tokenizer_pre_type> model_construction_data =
				nihilus::model_parser<config_new>::parse_model(data, &weight_memory, *static_cast<tokenizer_type*>(this));
			int64_t total_time{ nihilus::stop_watch_val_nihilus.total_time_elapsed().count() };
			std::cout << "Nihilus model Load time: " << total_time << std::endl;
		}

		NIHILUS_FORCE_INLINE void deinit(nihilus::cli_params params) {
			memory.deinit();
		}

		NIHILUS_FORCE_INLINE void execute_model(nihilus::execution_parameters& params) {
			current_iteration = 0;
			nihilus::stop_watch_val_nihilus.reset();
			this->template impl<dim_updater>(2ull);
#if defined(NIHILUS_DEBUG)
			this->template impl<tensor_debugger_impl>();
#endif
			++current_iteration;
			this->template impl<dim_updater>(12);
#if defined(NIHILUS_DEBUG)
			this->template impl<tensor_debugger_impl>();
#endif
			nihilus::stop_watch_val_nihilus.reset();
			static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();
			nihilus::stop_watch_val_nihilus.add_time();
			++current_iteration;
			this->template impl<dim_updater>(1ull);

			for (uint64_t x = 0; x < params.token_count - 1; ++x) {
				nihilus::stop_watch_val_nihilus.reset();
#if defined(NIHILUS_DEBUG)
				this->template impl<tensor_debugger_impl>();
				++current_iteration;
#endif
				static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();
				nihilus::stop_watch_val_nihilus.add_time();
			}
			// Perform all of the necessary stuff to execute the model - along with all of the constexpr values stored globally inside the class LOL!.
			// Because we only pay the "virtual overhead @ the top here == totally negligible.
		};

	  protected:
		nihilus::memory_mapped_file weight_memory{};
		nihilus::memory_buffer<config_new> memory{};
	};

}
