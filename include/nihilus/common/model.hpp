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
		NIHILUS_INLINE model_base() noexcept = default;
		NIHILUS_INLINE model_base(model_config config_new) : config{ config_new } {};
		model_config config{};
		virtual bool process_input(const std::string& params) = 0;
		virtual ~model_base()								  = default;
	};

	template<model_config config_new> struct model
		: public thread_pool<config_new, model<config_new>>,
		  public model_base,
		  public tokenizer<config_new, model<config_new>, model_traits<config_new.arch, config_new.model_size, config_new.model_generation>::arch, config_new.tokenizer_type> {
		using thread_pool_type		 = thread_pool<config_new, model<config_new>>;
		using model_traits_type		 = model_traits_type<config_new>;
		using core_bases_type		 = get_core_bases_t<config_new>;
		using core_bases_traits_type = core_bases_traits_type<config_new>;
		using op_type_type			 = typename model_traits_type ::op_type_type;
		using tokenizer_type		 = tokenizer<config_new, model, config_new.arch, config_new.tokenizer_type>;

		struct benchmark_stats {
			std::chrono::high_resolution_clock::time_point sampling_start;
			std::chrono::high_resolution_clock::time_point prompt_start;
			std::chrono::high_resolution_clock::time_point token_start;
			std::chrono::high_resolution_clock::time_point eval_start;
			std::chrono::high_resolution_clock::time_point load_start;
			bool has_evaluated_once			 = false;
			double total_load_time_ns		 = 0;
			double total_prompt_eval_time_ns = 0;
			double total_eval_time_ns		 = 0;
			double total_sampling_time_ns	 = 0;
			int32_t prompt_token_count		 = 0;
			int32_t generated_token_count	 = 0;
			int32_t total_sampling_runs		 = 0;
			uint64_t current_iteration		 = 0;
			double llama_style_load_time_ns	 = 0;
			double llama_style_eval_time_ns	 = 0;
		} perf_stats;

		template<auto op_type> auto& get_core() {
			return *static_cast<core_traits<config_new, op_type>*>(static_cast<get_core_bases_t<config_new>*>(this));
		}

		NIHILUS_INLINE model() noexcept = default;
		NIHILUS_INLINE model(cli_params params) : thread_pool<config_new, model>{ params.thread_count }, model_base{ config_new } {
			exec_params.token_count = params.n_tokens;
			init(params);
		}

		model& operator=(const model&) = delete;
		model(const model&)			   = delete;

		NIHILUS_INLINE bool process_input(const std::string& input) {
			tokenizer_type::tokenize_init(get_core<op_type_type::inp_tokens>().data);
			get_core<op_type_type::inp_pos>().data[1]	  = 1;
			get_core<op_type_type::inp_out_ids>().data[0] = 1;
			execute_model(input);
			return false;
		}

		NIHILUS_INLINE void init(cli_params params) {
			if constexpr (config_new.benchmark || config_new.dev) {
				perf_stats.load_start = std::chrono::high_resolution_clock::now();
			}
			memory.init(core_bases_traits_type::total_required_bytes);
			weight_memory = memory_mapped_file{ params.model_file };
			array<array<void*, model_traits_type::block_count>, op_types::count> data{};
			this->template impl<weight_mapper>(data);
			this->template impl<memory_mapper>(memory);

			gguf_metadata<config_new> model_construction_data = model_parser<config_new>::parse_model(data, &weight_memory, *static_cast<tokenizer_type*>(this));

			if constexpr (config_new.benchmark || config_new.dev) {
				auto load_end				  = std::chrono::high_resolution_clock::now();
				perf_stats.total_load_time_ns = std::chrono::duration<double, std::nano>(load_end - perf_stats.load_start).count();
			}
		}

		NIHILUS_INLINE void deinit() {
			memory.deinit();
		}

		NIHILUS_INLINE void execute_model(std::string_view input) {
			this->template impl<dim_updater>(2ull);
			if constexpr (config_new.dev) {
				this->template impl<tensor_debugger_impl>(perf_stats.current_iteration);
				perf_stats.current_iteration = 1;
			}

			exec_params.sequence_length = tokenizer_type::tokenize(input, get_core<op_type_type::inp_tokens>().data);
			this->template impl<dim_updater>(exec_params.sequence_length);

			for (size_t x = 0; x < exec_params.sequence_length; ++x) {
				get_core<op_type_type::inp_pos>().data[x] = x;
			}
			get_core<op_type_type::inp_out_ids>().data[0] = exec_params.sequence_length - 1;

			if constexpr (config_new.dev) {
				this->template impl<tensor_debugger_impl>(perf_stats.current_iteration);
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_stats.prompt_token_count	 = exec_params.sequence_length;
				perf_stats.generated_token_count = exec_params.token_count - 1;
				perf_stats.total_sampling_runs	 = exec_params.token_count;
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_stats.prompt_start		 = std::chrono::high_resolution_clock::now();
			}

			static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();

			if constexpr (config_new.benchmark || config_new.dev) {
				auto prompt_end						 = std::chrono::high_resolution_clock::now();
				perf_stats.total_prompt_eval_time_ns = std::chrono::duration<double, std::nano>(prompt_end - perf_stats.prompt_start).count();

				if (!perf_stats.has_evaluated_once) {
					perf_stats.llama_style_load_time_ns = std::chrono::duration<double, std::nano>(prompt_end - perf_stats.load_start).count();
					perf_stats.has_evaluated_once		= true;
				}
				++perf_stats.current_iteration;
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_stats.eval_start = std::chrono::high_resolution_clock::now();
			}

			for (uint64_t x = 0; x < exec_params.token_count - 1; ++x) {
				if constexpr (config_new.benchmark || config_new.dev) {
					perf_stats.token_start = std::chrono::high_resolution_clock::now();
				}

				if constexpr (config_new.dev) {
					this->template impl<tensor_debugger_impl>(perf_stats.current_iteration);
					++perf_stats.current_iteration;
				}
				static_cast<thread_pool<config_new, model>*>(this)->execute_tasks();

				this->template impl<dim_updater>(1ull);

				if constexpr (config_new.benchmark || config_new.dev) {
					auto token_end	   = std::chrono::high_resolution_clock::now();
					auto token_time_ns = std::chrono::duration<double, std::nano>(token_end - perf_stats.token_start).count();
					perf_stats.total_eval_time_ns += token_time_ns;

					if (perf_stats.has_evaluated_once) {
						perf_stats.llama_style_eval_time_ns += token_time_ns;
					}

					auto sampling_start = std::chrono::high_resolution_clock::now();
				}
				auto new_token = sample_next_token();
				if constexpr (config_new.benchmark || config_new.dev) {
					auto sampling_end	  = std::chrono::high_resolution_clock::now();
					auto sampling_time_ns = std::chrono::duration<double, std::nano>(sampling_end - perf_stats.sampling_start).count();
					perf_stats.total_sampling_time_ns += sampling_time_ns;
				}
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_stats.total_eval_time_ns	  = perf_stats.total_eval_time_ns;
				perf_stats.total_sampling_time_ns = perf_stats.total_sampling_time_ns;
			}

			if constexpr (config_new.dev) {
				this->template impl<execution_checker>(exec_params.thread_count);
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				print_performance_stats();
			}

			std::cout << "OP COUNT: " << count.load() << std::endl;
		}

	  protected:
		NIHILUS_INLINE int32_t sample_next_token() {
			auto& result_output_tensor = get_core<op_types::result_output>();
			float* logits			   = static_cast<float*>(result_output_tensor.data);
			uint64_t vocab_size		   = model_traits_type::vocab_size;
			return tokenizer_type::sample_next_token(logits, vocab_size);
		}

		NIHILUS_INLINE void print_performance_stats() {
			if constexpr (config_new.benchmark || config_new.dev) {
				int64_t total_time_ns = perf_stats.total_load_time_ns + perf_stats.total_prompt_eval_time_ns + perf_stats.total_eval_time_ns;
				double current_rate	  = perf_stats.total_sampling_time_ns / static_cast<double>(perf_stats.total_sampling_runs);
				int32_t total_tokens  = perf_stats.prompt_token_count + perf_stats.generated_token_count;

				std::cout << "\n=== HONEST NIHILUS TIMING ===" << std::endl;
				std::cout << "nihilus_perf_context_print:        load time = " << perf_stats.total_load_time_ns * 1e-6 << " ms" << std::endl;
				std::cout << "nihilus_perf_context_print: prompt eval time = " << perf_stats.total_prompt_eval_time_ns * 1e-6 << " ms / " << perf_stats.prompt_token_count
						  << " tokens (" << perf_stats.total_prompt_eval_time_ns * 1e-6 / static_cast<double>(perf_stats.prompt_token_count) << " ms per token, "
						  << static_cast<double>(perf_stats.prompt_token_count) / (perf_stats.total_prompt_eval_time_ns * 1e-9) << " tokens per second)" << std::endl;
				std::cout << "nihilus_perf_context_print:        eval time = " << perf_stats.total_eval_time_ns * 1e-6 << " ms / " << perf_stats.generated_token_count
						  << " runs   (" << perf_stats.total_eval_time_ns * 1e-6 / static_cast<double>(perf_stats.generated_token_count) << " ms per token, "
						  << static_cast<double>(perf_stats.generated_token_count) / (perf_stats.total_eval_time_ns * 1e-9) << " tokens per second)" << std::endl;
				std::cout << "nihilus_perf_context_print:       total time = " << total_time_ns * 1e-6 << " ms / " << total_tokens << " tokens" << std::endl;

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
		}

		memory_mapped_file weight_memory{};
		memory_buffer<config_new> memory{};
		execution_parameters exec_params{};
	};
}
