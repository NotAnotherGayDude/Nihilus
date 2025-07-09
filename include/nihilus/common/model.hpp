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
#include <nihilus/common/model_serializer.hpp>
#include <nihilus/common/core_bases.hpp>
#include <nihilus/cpu/thread_pool.hpp>
#include <nihilus/common/tuple.hpp>

namespace nihilus {

	struct model_base {
		NIHILUS_FORCE_INLINE model_base() noexcept = default;
		NIHILUS_FORCE_INLINE model_base(model_config config_new) : config{ config_new } {};
		model_config config{};
		virtual bool process_input(const std::string& params)	 = 0;
		virtual void init(cli_params params)					 = 0;
		virtual ~model_base()									 = default;
	};

	template<model_config config_new> struct model
		: public thread_pool<config_new, model<config_new>>,
		  public model_base,
		  public tokenizer<config_new, model<config_new>, model_traits<config_new.arch, config_new.model_size, config_new.model_generation>::arch, config_new.tokenizer_type> {
		using thread_pool_type	= thread_pool<config_new, model<config_new>>;
		using model_traits_type = nihilus::model_traits_type<config_new>;
		using core_bases_t		= get_core_bases_t<config_new>;
		using core_bases_traits = core_bases_traits<config_new>;
		using op_type_type		= typename model_traits_type ::op_type_type;
		using tokenizer_type	= tokenizer<config_new, model, config_new.arch, config_new.tokenizer_type>;
		using base_type			= model_base;

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
		struct benchmark_stats {
			double total_load_time_ns		 = 0;
			double total_prompt_eval_time_ns = 0;
			double total_eval_time_ns		 = 0;
			double total_sampling_time_ns	 = 0;
			int32_t prompt_token_count		 = 0;
			int32_t generated_token_count	 = 0;
			int32_t total_sampling_runs		 = 0;

			// llama.cpp-style timing variables
			bool has_evaluated_once = false;
			std::chrono::high_resolution_clock::time_point t_start;
			double llama_style_load_time_ns = 0;
			double llama_style_eval_time_ns = 0;
		} perf_stats;
#endif

		template<auto op_type> auto& get_core() {
			return *static_cast<nihilus::core_traits<config_new, op_type>*>(static_cast<get_core_bases_t<config_new>*>(this));
		}

		NIHILUS_FORCE_INLINE model() noexcept = default;
		NIHILUS_FORCE_INLINE model(nihilus::cli_params params) : thread_pool<config_new, model>{ params.thread_count }, model_base{ config_new } {
			exec_params.token_count = params.n_tokens;
			init(params);
		}

		model& operator=(const model&) = delete;
		model(const model&)			   = delete;

		NIHILUS_FORCE_INLINE bool process_input(const std::string& input) {
			tokenizer_type::tokenize_init(get_core<op_type_type::inp_tokens>().data);
			get_core<op_type_type::inp_pos>().data[1] = 1;
			get_core<op_type_type::inp_out_ids>().data[0] = 1;
			execute_model(input);
			return false;
		}

		NIHILUS_FORCE_INLINE void init(nihilus::cli_params params) {
#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
			auto load_start	   = std::chrono::high_resolution_clock::now();
			perf_stats.t_start = load_start;// Start timing like llama.cpp
#endif

			memory.init(core_bases_traits::total_required_bytes);
			weight_memory = nihilus::memory_mapped_file{ params.model_file };
			nihilus::array<nihilus::array<void*, model_traits_type::block_count>, op_types::count> data{};
			this->template impl<weight_mapper>(data);
			this->template impl<memory_mapper>(memory);

			gguf_metadata<config_new.arch, config_new.tokenizer_type, config_new.tokenizer_pre_type> model_construction_data =
				nihilus::model_parser<config_new>::parse_model(data, &weight_memory, *static_cast<tokenizer_type*>(this));

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
			auto load_end				  = std::chrono::high_resolution_clock::now();
			perf_stats.total_load_time_ns = std::chrono::duration<double, std::nano>(load_end - load_start).count();
			// Don't set llama_style_load_time_ns here - it gets set after first inference!
#endif
		}

		NIHILUS_FORCE_INLINE void deinit() {
			memory.deinit();
		}

		NIHILUS_FORCE_INLINE void execute_model(std::string_view input) {

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)

			auto prompt_start = std::chrono::high_resolution_clock::now();
			current_iteration = 0;
#endif

			this->template impl<dim_updater>(2ull);
#if defined(NIHILUS_DEV)
			this->template impl<tensor_debugger_impl>();
			++current_iteration;
#endif

			exec_params.sequence_length = tokenizer_type::tokenize(input, get_core<op_type_type::inp_tokens>().data);
			this->template impl<dim_updater>(exec_params.sequence_length);
#if defined(NIHILUS_DEV)
			this->template impl<tensor_debugger_impl>();
#endif

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
			perf_stats.prompt_token_count	 = exec_params.sequence_length;
			perf_stats.generated_token_count = exec_params.token_count - 1;
			perf_stats.total_sampling_runs	 = exec_params.token_count;
#endif

			static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
			auto prompt_end						 = std::chrono::high_resolution_clock::now();
			perf_stats.total_prompt_eval_time_ns = std::chrono::duration<double, std::nano>(prompt_end - prompt_start).count();

			// llama.cpp-style: set "load time" to include everything up to first inference completion
			if (!perf_stats.has_evaluated_once) {
				perf_stats.llama_style_load_time_ns = std::chrono::duration<double, std::nano>(prompt_end - perf_stats.t_start).count();
				perf_stats.has_evaluated_once		= true;
			}
#endif

			++current_iteration;
			this->template impl<dim_updater>(1ull);

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
			auto eval_start				   = std::chrono::high_resolution_clock::now();
			int64_t total_eval_time_ns	   = 0;
			int64_t total_sampling_time_ns = 0;
#endif

			for (uint64_t x = 0; x < exec_params.token_count; ++x) {
#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
				auto token_start = std::chrono::high_resolution_clock::now();
#endif

#if defined(NIHILUS_DEV)
				this->template impl<tensor_debugger_impl>();
				++current_iteration;
#endif
				static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
				auto token_end	   = std::chrono::high_resolution_clock::now();
				auto token_time_ns = std::chrono::duration<double, std::nano>(token_end - token_start).count();
				total_eval_time_ns += token_time_ns;

				// llama.cpp-style: only count eval time for subsequent tokens (after first inference)
				if (perf_stats.has_evaluated_once) {
					perf_stats.llama_style_eval_time_ns += token_time_ns;
				}

				auto sampling_start = std::chrono::high_resolution_clock::now();
#endif
				auto new_token = sample_next_token();
#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
				auto sampling_end	  = std::chrono::high_resolution_clock::now();
				auto sampling_time_ns = std::chrono::duration<double, std::nano>(sampling_end - sampling_start).count();
				total_sampling_time_ns += sampling_time_ns;
#endif
			}

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
			perf_stats.total_eval_time_ns	  = total_eval_time_ns;
			perf_stats.total_sampling_time_ns = total_sampling_time_ns;
#endif

#if defined(NIHILUS_DEV)
			this->template impl<execution_checker>(exec_params.thread_count);
#endif

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
			print_performance_stats();
#endif

			std::cout << "OP COUNT: " << current_count.load() << std::endl;
		}

	  protected:
		NIHILUS_FORCE_INLINE int32_t sample_next_token() {
			auto& result_output_tensor = get_core<op_types::result_output>();
			float* logits			   = static_cast<float*>(result_output_tensor.data);
			uint64_t vocab_size		   = model_traits_type::tokenizer_size;
			return tokenizer_type::sample_next_token(logits, vocab_size);
		}

#if defined(NIHILUS_BENCHMARK) || defined(NIHILUS_DEV)
		NIHILUS_FORCE_INLINE void print_performance_stats() {
			// Print HONEST timing first
			int64_t total_time_ns = perf_stats.total_load_time_ns + perf_stats.total_prompt_eval_time_ns + perf_stats.total_eval_time_ns + perf_stats.total_sampling_time_ns;
			double current_rate	  = perf_stats.total_sampling_time_ns / static_cast<double>(perf_stats.total_sampling_runs);
			int32_t total_tokens  = perf_stats.prompt_token_count + perf_stats.generated_token_count;

			std::cout << "\n=== HONEST NIHILUS TIMING ===" << std::endl;
			std::cout << "nihilus_perf_context_print:        load time = " << perf_stats.total_load_time_ns * 1e-6 << " ms" << std::endl;
			std::cout << "nihilus_perf_context_print: prompt eval time = " << perf_stats.total_prompt_eval_time_ns * 1e-6 << " ms / " << perf_stats.prompt_token_count
					  << " tokens (" << perf_stats.total_prompt_eval_time_ns * 1e-6 / static_cast<double>(perf_stats.prompt_token_count) << " ms per token, "
					  << static_cast<double>(perf_stats.prompt_token_count) / (perf_stats.total_prompt_eval_time_ns * 1e-9) << " tokens per second)" << std::endl;
			std::cout << "nihilus_perf_context_print:        eval time = " << perf_stats.total_eval_time_ns * 1e-6 << " ms / " << perf_stats.generated_token_count << " runs   ("
					  << perf_stats.total_eval_time_ns * 1e-6 / static_cast<double>(perf_stats.generated_token_count) << " ms per token, "
					  << static_cast<double>(perf_stats.generated_token_count) / (perf_stats.total_eval_time_ns * 1e-9) << " tokens per second)" << std::endl;
			std::cout << "nihilus_perf_context_print:       total time = " << total_time_ns * 1e-6 << " ms / " << total_tokens << " tokens" << std::endl;

			// Print llama.cpp-style DECEPTIVE timing
			std::cout << "\n=== LLAMA.CPP-STYLE DECEPTIVE TIMING ===" << std::endl;
			std::cout << "llama_print_timings:        load time = " << perf_stats.llama_style_load_time_ns * 1e-6 << " ms" << std::endl;
			std::cout << "llama_print_timings:       total time = " << perf_stats.llama_style_eval_time_ns * 1e-6 << " ms" << std::endl;

			std::cout << "\n=== DECEPTION ANALYSIS ===" << std::endl;
			std::cout << "llama.cpp 'load time' includes: " << (perf_stats.llama_style_load_time_ns - perf_stats.total_load_time_ns) * 1e-6 << " ms of inference work!"
					  << std::endl;
			std::cout << "Real load time advantage: " << (perf_stats.llama_style_load_time_ns / perf_stats.total_load_time_ns) << "x faster than their reported 'load time'"
					  << std::endl;

			//std::cout << "nihilus_perf_sampler_print:    sampling time = " << perf_stats.total_sampling_time_ns * 1e-6 << " ms / " << perf_stats.total_sampling_runs << " runs   ("
			//<< perf_stats.total_sampling_time_ns * 1e-6 / static_cast<double>(perf_stats.total_sampling_runs) << " ms per token, "
			//<< static_cast<double>(perf_stats.total_sampling_runs) / (perf_stats.total_sampling_time_ns * 1e-9) << " tokens per second)" << std::endl;
		}

		int64_t t_compute_start_us = 0;
		int64_t n_queued_tokens	   = 0;
		int64_t t_start_us		   = 0;
		int64_t t_load_us		   = 0;
		int64_t t_p_eval_us		   = 0;
		int64_t t_eval_us		   = 0;
		int32_t n_p_eval		   = 0;
		int32_t n_eval			   = 0;

#endif
		nihilus::memory_mapped_file weight_memory{};
		nihilus::memory_buffer<config_new> memory{};
		execution_parameters exec_params{};
	};

}
