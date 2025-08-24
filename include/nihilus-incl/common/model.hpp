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

#include <nihilus-incl/common/model_traits.hpp>
#include <nihilus-incl/common/model_parser.hpp>
#include <nihilus-incl/common/model_serializer.hpp>
#include <nihilus-incl/common/core_bases.hpp>
#include <nihilus-incl/cpu/thread_pool.hpp>
#include <nihilus-incl/common/tuple.hpp>

namespace nihilus {

	struct model_base {
		NIHILUS_INLINE model_base() noexcept = default;
		NIHILUS_INLINE model_base(model_config config_new) : config{ config_new } {
		}
		model_config config{};
		virtual bool process_input(const std::string_view params) = 0;
		virtual ~model_base();
	};

	model_base::~model_base() {};

	template<model_config config_new> struct model
		: public thread_pool<config_new>,
		  public model_base,
		  public tokenizer<config_new, model_traits<config_new.arch, config_new.model_size, config_new.model_generation>::arch, config_new.tokenizer_type> {
		using thread_pool_type		 = thread_pool<config_new>;
		using core_bases_type		 = get_core_bases_t<config_new, core_types>;
		using core_bases_traits_type = core_bases_traits<config_new>;
		using tokenizer_type		 = tokenizer<config_new, config_new.arch, config_new.tokenizer_type>;

		NIHILUS_INLINE model() noexcept {
		}

		NIHILUS_INLINE model(cli_params params) : thread_pool<config_new>{ static_cast<int64_t>(params.thread_count) }, model_base{ config_new } {
			exec_params.token_count = params.n_tokens;
			init(params);
		}

		model& operator=(const model&) = delete;
		model(const model&)			   = delete;

		NIHILUS_INLINE bool process_input(const std::string_view input) override {
			tokenizer_type::tokenize_init(
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_tokens>().data);
			this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>().data[1]	 = 1;
			this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>().data[0] = 1;
			generate_causal_mask();
			execute_model(input);
			return false;
		}

		NIHILUS_INLINE void init(cli_params params) {
			std::cout << "(Nihilus) Total Bytes Required for Intermediate Tensors at Context Length Of: " << config.default_max_sequence_length << ": "
					  << core_bases_traits_type::total_required_bytes.peak_allocated_bytes << std::endl;
			memory.init(core_bases_traits_type::total_required_bytes.peak_allocated_bytes);
			array<array<void*, model_traits_type<config_new>::block_count>, weight_types::count> data{};
			weight_mapper<config_new, core_traits<config_new, core_types::weights>>::impl(*static_cast<core_traits<config_new, core_types::weights>*>(this), data);
			this->template impl<memory_mapper>(core_bases_traits_type::total_required_bytes, memory);

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_base<config_new>::perf_stats.load_start = clock_type::now();
			}
			weight_memory									  = memory_mapped_file{ params.model_file };
			gguf_metadata<config_new> model_construction_data = model_parser<config_new>::parse_model(data, &weight_memory, *static_cast<tokenizer_type*>(this));

			if constexpr (config_new.benchmark || config_new.dev) {
				auto load_end										 = clock_type::now();
				perf_base<config_new>::perf_stats.total_load_time_ns = std::chrono::duration<double, std::nano>(load_end - perf_base<config_new>::perf_stats.load_start).count();
			}
		}

		NIHILUS_INLINE void deinit() {
			memory.deinit();
		}

		NIHILUS_INLINE void execute_model(std::string_view input) {
			static_cast<thread_pool<config_new>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(2);

			if constexpr (config_new.dev) {
				++perf_base<config_new>::perf_stats.current_iteration;
			}

			exec_params.sequence_length = tokenizer_type::tokenize(input,
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_tokens>().data);

			for (uint64_t x = 0; x < exec_params.sequence_length; ++x) {
				using core_type = detail::remove_cvref_t<
					decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>())>;
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>().data[x] =
					static_cast<typename core_type::output_type>(x);
			}
			using core_type = detail::remove_cvref_t<decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>())>;
			this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>().data[0] =
				static_cast<typename core_type::output_type>(exec_params.sequence_length - 1);
				
			if constexpr (config_new.benchmark || config_new.dev) {
				perf_base<config_new>::perf_stats.prompt_token_count	= exec_params.sequence_length;
				perf_base<config_new>::perf_stats.generated_token_count = exec_params.token_count - 1;
				perf_base<config_new>::perf_stats.total_sampling_runs	= exec_params.token_count;
				perf_base<config_new>::perf_stats.total_eval_time_ns	= 0;
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_base<config_new>::perf_stats.prompt_start = clock_type::now();
			}

			static_cast<thread_pool<config_new>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(exec_params.sequence_length);

			if constexpr (config_new.dev) {
				++perf_base<config_new>::perf_stats.current_iteration;
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				auto prompt_end = clock_type::now();
				perf_base<config_new>::perf_stats.total_prompt_eval_time_ns =
					std::chrono::duration<double, std::nano>(prompt_end - perf_base<config_new>::perf_stats.prompt_start).count();
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_base<config_new>::perf_stats.eval_start = clock_type::now();
			}

			for (uint64_t x = 0; x < exec_params.token_count - 1; ++x) {
				if constexpr (config_new.benchmark || config_new.dev) {
					perf_base<config_new>::perf_stats.token_start = clock_type::now();
				}
				static_cast<thread_pool<config_new>*>(this)->template execute_tasks<processing_phases::eval_time>(1);

				if constexpr (config_new.benchmark || config_new.dev) {
					++perf_base<config_new>::perf_stats.current_iteration;
					auto token_end	   = clock_type::now();
					auto token_time_ns = std::chrono::duration<double, std::nano>(token_end - perf_base<config_new>::perf_stats.token_start).count();
					perf_base<config_new>::perf_stats.total_eval_time_ns += token_time_ns;
				}
				auto new_token = sample_next_token();
				( void )(new_token);
				if constexpr (config_new.benchmark || config_new.dev) {
					auto sampling_end	  = clock_type::now();
					auto sampling_time_ns = std::chrono::duration<double, std::nano>(sampling_end - perf_base<config_new>::perf_stats.sampling_start).count();
					perf_base<config_new>::perf_stats.total_sampling_time_ns += sampling_time_ns;
				}
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				perf_base<config_new>::perf_stats.total_eval_time_ns	 = perf_base<config_new>::perf_stats.total_eval_time_ns;
				perf_base<config_new>::perf_stats.total_sampling_time_ns = perf_base<config_new>::perf_stats.total_sampling_time_ns;
			}

			if constexpr (config_new.benchmark || config_new.dev) {
				print_performance_stats();
			}
		}

		NIHILUS_INLINE ~model() {
			deinit();
		}

	  protected:
		NIHILUS_INLINE int32_t sample_next_token() {
			//auto& result_output_tensor = get_core<core_types, core_types::result_output>();
			//float* logits			   = static_cast<float*>(result_output_tensor.data);
			//uint64_t vocab_size		   = model_traits_type::vocab_size;
			return {};
		}

		NIHILUS_INLINE void generate_causal_mask() {
			using core_type = detail::remove_cvref_t<
				decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::kq_mask>())>;
			float* mask_data = this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::kq_mask>().data;
			static constexpr auto dims = core_type::get_array();
			const uint64_t total_dims  = dims[0] * dims[1] * dims[2] * dims[3];
			for (uint64_t x = 0; x < total_dims; ++x) {
				if (x == 0 || x == 32 || x == 33) {
					mask_data[x] = 0.0f;
				} else {
					mask_data[x] = -std::numeric_limits<float>::infinity();
				}
			}
		}

		NIHILUS_INLINE void print_performance_stats() {
			if constexpr (config_new.benchmark || config_new.dev) {
				int64_t total_time_ns = perf_base<config_new>::perf_stats.total_prompt_eval_time_ns + perf_base<config_new>::perf_stats.total_eval_time_ns;
				int32_t total_tokens  = perf_base<config_new>::perf_stats.prompt_token_count + perf_base<config_new>::perf_stats.generated_token_count;

				std::cout << "\n=== HONEST NIHILUS TIMING ===" << std::endl;
				std::cout << "nihilus_perf_context_print:        load time = " << perf_base<config_new>::perf_stats.total_load_time_ns * 1e-6 << " ms" << std::endl;
				std::cout << "nihilus_perf_context_print: prompt eval time = " << perf_base<config_new>::perf_stats.total_prompt_eval_time_ns * 1e-6 << " ms / "
						  << perf_base<config_new>::perf_stats.prompt_token_count << " tokens ("
						  << perf_base<config_new>::perf_stats.total_prompt_eval_time_ns * 1e-6 / static_cast<double>(perf_base<config_new>::perf_stats.prompt_token_count)
						  << " ms per token, "
						  << static_cast<double>(perf_base<config_new>::perf_stats.prompt_token_count) / (perf_base<config_new>::perf_stats.total_prompt_eval_time_ns * 1e-9)
						  << " tokens per second)" << std::endl;
				std::cout << "nihilus_perf_context_print:        eval time = " << perf_base<config_new>::perf_stats.total_eval_time_ns * 1e-6 << " ms / "
						  << perf_base<config_new>::perf_stats.generated_token_count << " runs   ("
						  << perf_base<config_new>::perf_stats.total_eval_time_ns * 1e-6 / static_cast<double>(perf_base<config_new>::perf_stats.generated_token_count)
						  << " ms per token, "
						  << static_cast<double>(perf_base<config_new>::perf_stats.generated_token_count) / (perf_base<config_new>::perf_stats.total_eval_time_ns * 1e-9)
						  << " tokens per second)" << std::endl;
				std::cout << "nihilus_perf_context_print:       total time = " << static_cast<double>(total_time_ns) * 1e-6 << " ms / " << static_cast<double>(total_tokens)
						  << " tokens" << std::endl;

				/*
				std::cout << "nihilus_perf_sampler_print:    sampling time = " << perf_base<config_new>::perf_stats.total_sampling_time_ns * 1e-6 << " ms / " << perf_base<config_new>::perf_stats.total_sampling_runs << " runs   ("
				<< perf_base<config_new>::perf_stats.total_sampling_time_ns * 1e-6 / static_cast<double>(perf_base<config_new>::perf_stats.total_sampling_runs) << " ms per token, "
				<< static_cast<double>(perf_base<config_new>::perf_stats.total_sampling_runs) / (perf_base<config_new>::perf_stats.total_sampling_time_ns * 1e-9) << " tokens per second)" << std::endl;
				*/
			}
		}

		memory_mapped_file weight_memory{};
		memory_buffer<config_new> memory{};
		execution_parameters exec_params{};
	};
}
