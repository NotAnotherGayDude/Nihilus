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

#if defined(NIHILUS_BENCHMARK)
		struct benchmark_stats {
			int64_t total_load_time_ms		  = 0;
			int64_t total_prompt_eval_time_ms = 0;
			int64_t total_eval_time_ms		  = 0;
			int64_t total_sampling_time_ms	  = 0;
			int32_t prompt_token_count		  = 0;
			int32_t generated_token_count	  = 0;
			int32_t total_sampling_runs		  = 0;
		} perf_stats;
#endif

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
#if defined(NIHILUS_BENCHMARK)
			auto load_start = std::chrono::high_resolution_clock::now();
#endif

			memory.init(core_bases_traits::total_required_bytes);
			weight_memory = nihilus::memory_mapped_file{ params.model_file };
			nihilus::array<nihilus::array<void*, model_traits_type::block_count>, op_types::count> data{};
			this->template impl<weight_mapper>(data);
			this->template impl<memory_mapper>(memory);

			gguf_metadata<config_new.arch, config_new.tokenizer_type, config_new.tokenizer_pre_type> model_construction_data =
				nihilus::model_parser<config_new>::parse_model(data, &weight_memory, *static_cast<tokenizer_type*>(this));

#if defined(NIHILUS_BENCHMARK)
			auto load_end				  = std::chrono::high_resolution_clock::now();
			perf_stats.total_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();

			std::cout << "nihilus_perf_context_print:        load time = " << perf_stats.total_load_time_ms << " ms" << std::endl;
#endif
		}

		NIHILUS_FORCE_INLINE void deinit(nihilus::cli_params params) {
			memory.deinit();
		}

		NIHILUS_FORCE_INLINE void execute_model(nihilus::execution_parameters& params) {
			current_iteration = 0;

#if defined(NIHILUS_BENCHMARK)
			perf_stats.prompt_token_count	 = params.sequence_length;
			perf_stats.generated_token_count = params.token_count - 1;
			perf_stats.total_sampling_runs	 = params.token_count;

			auto prompt_start = std::chrono::high_resolution_clock::now();
#endif

			this->template impl<dim_updater>(2ull);
#if defined(NIHILUS_DEV)
			this->template impl<tensor_debugger_impl>();
#endif
			++current_iteration;
			this->template impl<dim_updater>(12);
#if defined(NIHILUS_DEV)
			this->template impl<tensor_debugger_impl>();
#endif

			static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();

#if defined(NIHILUS_BENCHMARK)
			auto prompt_end						 = std::chrono::high_resolution_clock::now();
			perf_stats.total_prompt_eval_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prompt_end - prompt_start).count();

			std::cout << "nihilus_perf_context_print: prompt eval time = " << perf_stats.total_prompt_eval_time_ms << " ms / " << perf_stats.prompt_token_count << " tokens ("
					  << ( float )perf_stats.total_prompt_eval_time_ms / perf_stats.prompt_token_count << " ms per token, "
					  << (1000.0f * perf_stats.prompt_token_count) / perf_stats.total_prompt_eval_time_ms << " tokens per second)" << std::endl;
#endif

			++current_iteration;
			this->template impl<dim_updater>(1ull);

#if defined(NIHILUS_BENCHMARK)
			auto eval_start				   = std::chrono::high_resolution_clock::now();
			int64_t total_eval_time_ms	   = 0;
			int64_t total_sampling_time_ms = 0;
#endif

			for (uint64_t x = 0; x < params.token_count - 1; ++x) {
#if defined(NIHILUS_BENCHMARK)
				auto token_start = std::chrono::high_resolution_clock::now();
#endif

#if defined(NIHILUS_DEV)
				this->template impl<tensor_debugger_impl>();
				++current_iteration;
#endif
				static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();

#if defined(NIHILUS_BENCHMARK)
				auto token_end	   = std::chrono::high_resolution_clock::now();
				auto token_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(token_end - token_start).count();
				total_eval_time_ms += token_time_ms;
				auto sampling_start = std::chrono::high_resolution_clock::now();
				sample_next_token(params);

				auto sampling_end	  = std::chrono::high_resolution_clock::now();
				auto sampling_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sampling_end - sampling_start).count();
				total_sampling_time_ms += sampling_time_ms;
#endif
			}

#if defined(NIHILUS_BENCHMARK)
			perf_stats.total_eval_time_ms	  = total_eval_time_ms;
			perf_stats.total_sampling_time_ms = total_sampling_time_ms;
#endif

#if defined(NIHILUS_DEV)
			this->template impl<execution_checker>(params.thread_count);
#endif

#if defined(NIHILUS_BENCHMARK)
			print_performance_stats();
#endif

			std::cout << "OP COUNT: " << current_count.load() << std::endl;
		}

	  private:
#if defined(NIHILUS_BENCHMARK)
		NIHILUS_FORCE_INLINE void sample_next_token(nihilus::execution_parameters& params) {
			volatile int dummy_work = 0;
			for (int i = 0; i < 100; ++i) {
				dummy_work += i;
			}
		}

		NIHILUS_FORCE_INLINE void print_performance_stats() {
			int64_t total_time_ms = perf_stats.total_load_time_ms + perf_stats.total_prompt_eval_time_ms + perf_stats.total_eval_time_ms + perf_stats.total_sampling_time_ms;

			int32_t total_tokens = perf_stats.prompt_token_count + perf_stats.generated_token_count;
			std::cout << "nihilus_perf_sampler_print:    sampling time = " << perf_stats.total_sampling_time_ms << " ms / " << perf_stats.total_sampling_runs << " runs   ("
					  << ( float )perf_stats.total_sampling_time_ms / perf_stats.total_sampling_runs << " ms per token, "
					  << (1000.0f * perf_stats.total_sampling_runs) / perf_stats.total_sampling_time_ms << " tokens per second)" << std::endl;

			std::cout << "nihilus_perf_context_print:        eval time = " << perf_stats.total_eval_time_ms << " ms / " << perf_stats.generated_token_count << " runs   ("
					  << ( float )perf_stats.total_eval_time_ms / perf_stats.generated_token_count << " ms per token, "
					  << (1000.0f * perf_stats.generated_token_count) / perf_stats.total_eval_time_ms << " tokens per second)" << std::endl;

			std::cout << "nihilus_perf_context_print:       total time = " << total_time_ms << " ms / " << total_tokens << " tokens" << std::endl;
		}
#endif

	  protected:
		nihilus::memory_mapped_file weight_memory{};
		nihilus::memory_buffer<config_new> memory{};
	};

}
