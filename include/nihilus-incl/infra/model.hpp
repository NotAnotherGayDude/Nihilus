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

#include <nihilus-incl/infra/model_traits.hpp>
#include <nihilus-incl/infra/model_parser.hpp>
#include <nihilus-incl/cpu/memory_mapped_file.hpp>
#include <nihilus-incl/infra/model_serializer.hpp>
#include <nihilus-incl/infra/core_bases.hpp>
#include <nihilus-incl/cpu/thread_pool.hpp>
#include <nihilus-incl/cuda/thread_pool.hpp>
#include <nihilus-incl/common/input_collector.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <nihilus-incl/infra/jitter_generator.hpp>
#include <span>

namespace nihilus {

	template<typename... bases> struct model_collection : public bases... {
		template<typename... arg_types> NIHILUS_HOST explicit model_collection(arg_types&&... params) : bases(params)... {
		}

		NIHILUS_HOST explicit model_collection(){}

		NIHILUS_HOST bool process_input() noexcept {
			return (... && bases::process_input_impl());
		}

		NIHILUS_HOST bool process_input(std::string_view prompt) noexcept {
			return (... && bases::process_input_impl(prompt));
		}
	};

	template<uint64_t max_sequence_length> struct user_prompt {
		NIHILUS_HOST user_prompt() {}
		array<char, max_sequence_length> prompt{};
		uint64_t user_id{};
	};

	template<typename https_server, typename... bases> struct https_model_collection : public bases... {
		template<typename... arg_types> NIHILUS_HOST explicit https_model_collection(arg_types&&... params) : bases(params)... {
		}

		NIHILUS_HOST explicit https_model_collection(){}

		NIHILUS_HOST bool process_input() noexcept {
			return (... && (https_ptr->template get_next_prompts<bases>(bases::user_prompts), bases::process_prompts_impl()));
		}

	  protected:
		https_server* https_ptr{};
	};

	template<uint64_t index_new, typename config_type>
	struct model : public input_collector<config_type>, public thread_pool<config_type>, public tokenizer<config_type, config_type::model_arch, config_type::tokenizer_type> {
		using thread_pool_type = thread_pool<config_type>;
		using core_bases_type  = get_core_bases_t<config_type>;
		using tokenizer_type   = tokenizer<config_type, config_type::model_arch, config_type::tokenizer_type>;

		aligned_vector<user_prompt<config_type::max_sequence_length>> user_prompts{};

		NIHILUS_HOST model() noexcept {
		}

		NIHILUS_HOST model(cli_params params) : thread_pool<config_type>{ static_cast<int64_t>(params.thread_count) } {
			exec_params.token_count = params.n_tokens;
			user_prompts.resize(config_type::batch_size);
			init(params);
		}

		model& operator=(const model&) = delete;
		model(const model&)			   = delete;

		NIHILUS_HOST bool process_input_impl(std::string_view input, [[maybe_unused]] uint64_t seed_new = 0) {
			tokenizer_type::init_rng(seed_new);
			input = input.size() > config_type::max_sequence_length ? input.substr(0, config_type::max_sequence_length) : input;
			tokenizer_type::tokenize_init(
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_tokens>().get_data());
			using output_type = detail::remove_cvref_t<decltype(this->template get_core<core_types, core_types::global_inputs>()
					.values.template get_core<global_input_types, global_input_types::inp_tokens>())>::output_type;
			output_type val{ 1 };
			memory_transfer<config_type>::host_to_device(val,
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>().get_data() + 1);
			memory_transfer<config_type>::host_to_device(val,
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>().get_data());
			generate_causal_mask();
			execute_model(input);
			return false;
		}

		NIHILUS_HOST bool process_input_impl() {
			input_collector<config_type>::read_multiline();
			tokenizer_type::tokenize_init(
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_tokens>().get_data());
			using output_type = detail::remove_cvref_t<decltype(this->template get_core<core_types, core_types::global_inputs>()
					.values.template get_core<global_input_types, global_input_types::inp_tokens>())>::output_type;
			output_type val{ 1 };
			memory_transfer<config_type>::host_to_device(val,
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>().get_data() + 1);
			memory_transfer<config_type>::host_to_device(val,
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>().get_data());
			generate_causal_mask();
			execute_model(input_collector<config_type>::get_view());
			input_collector<config_type>::clear();
			return true;
		}

		NIHILUS_HOST void init(cli_params params) {
			if constexpr (config_type::device_type == device_types::cpu) {
				std::cout << "Model: " << model_traits_type<config_type>::name << "-" << kernel_type_profile_traits<config_type::kernel_type_profile>::name
						  << " (Nihilus-CPU) Total Bytes Required for Intermediate Tensors + KV_cache at Context Length Of: " << config_type::max_sequence_length << ": "
						  << core_bases_memory_plan<config_type>.peak_allocated_bytes << std::endl;
			} else {
				std::cout << "Model: " << model_traits_type<config_type>::name << "-" << kernel_type_profile_traits<config_type::kernel_type_profile>::name
						  << " (Nihilus-CUDA) Total Bytes Required for All Tensors + Weights + KV_cache at Context Length Of: " << config_type::max_sequence_length << ": "
						  << core_bases_memory_plan<config_type>.peak_allocated_bytes << std::endl;
			}
			memory.init(core_bases_memory_plan<config_type>.peak_allocated_bytes);
			this->template impl<memory_mapper>(core_bases_memory_plan<config_type>, memory);
			array<array<void*, model_traits_type<config_type>::block_count>, weight_types::count> data{};
			weight_mapper<config_type, core_traits<config_type, core_types::weights>>::impl(*static_cast<core_traits<config_type, core_types::weights>*>(this), data);

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.load_start = clock_type::now();
			}
			model_parser<config_type>::parse_model(params.model_file, data, metadata_memory, weight_memory, *static_cast<tokenizer_type*>(this));

			if constexpr (config_type::benchmark || config_type::dev) {
				auto load_end										  = clock_type::now();
				perf_base<config_type>::perf_stats.total_load_time_ns = std::chrono::duration<double, std::nano>(load_end - perf_base<config_type>::perf_stats.load_start).count();
			}
			if constexpr (config_type::device_type == device_types::gpu) {
				weight_memory.~optional<memory_mapped_file<config_type>>();
			}
		}

		NIHILUS_HOST void execute_model(const std::string_view input) {
			static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(2);

			if constexpr (config_type::dev) {
				++perf_base<config_type>::perf_stats.current_iteration;
			}

			exec_params.sequence_length = tokenizer_type::tokenize(input,
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_tokens>().get_data());

			for (uint64_t x = 0; x < exec_params.sequence_length; ++x) {
				using core_type = detail::remove_cvref_t<
					decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>())>;
				memory_transfer<config_type>::host_to_device(static_cast<typename core_type::output_type>(x),
					this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>().get_data() + x);
			}
			using core_type = detail::remove_cvref_t<
				decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>())>;
			memory_transfer<config_type>::host_to_device(static_cast<typename core_type::output_type>(exec_params.sequence_length - 1),
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>().get_data());

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.prompt_token_count	 = exec_params.sequence_length;
				perf_base<config_type>::perf_stats.generated_token_count = exec_params.token_count - 1;
				perf_base<config_type>::perf_stats.total_sampling_runs	 = exec_params.token_count;
				perf_base<config_type>::perf_stats.total_eval_time_ns	 = 0;
			}

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.prompt_start = clock_type::now();
			}

			static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(exec_params.sequence_length);

			if constexpr (config_type::dev) {
				++perf_base<config_type>::perf_stats.current_iteration;
			}

			if constexpr (config_type::benchmark || config_type::dev) {
				auto prompt_end = clock_type::now();
				perf_base<config_type>::perf_stats.total_prompt_eval_time_ns =
					std::chrono::duration<double, std::nano>(prompt_end - perf_base<config_type>::perf_stats.prompt_start).count();
			}

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.eval_start = clock_type::now();
			}

			for (uint64_t x = 0; x < exec_params.token_count - 1; ++x) {
				if constexpr (config_type::benchmark || config_type::dev) {
					perf_base<config_type>::perf_stats.token_start = clock_type::now();
				}
				static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::eval_time>(1);

				if constexpr (config_type::benchmark || config_type::dev) {
					++perf_base<config_type>::perf_stats.current_iteration;
					auto token_end	   = clock_type::now();
					auto token_time_ns = std::chrono::duration<double, std::nano>(token_end - perf_base<config_type>::perf_stats.token_start).count();
					perf_base<config_type>::perf_stats.total_eval_time_ns += token_time_ns;
				}
				[[maybe_unused]] auto new_token = sample_next_token();
				if constexpr (config_type::benchmark || config_type::dev) {
					auto sampling_end	  = clock_type::now();
					auto sampling_time_ns = std::chrono::duration<double, std::nano>(sampling_end - perf_base<config_type>::perf_stats.sampling_start).count();
					perf_base<config_type>::perf_stats.total_sampling_time_ns += sampling_time_ns;
				}
			}

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.total_eval_time_ns	  = perf_base<config_type>::perf_stats.total_eval_time_ns;
				perf_base<config_type>::perf_stats.total_sampling_time_ns = perf_base<config_type>::perf_stats.total_sampling_time_ns;
			}

			if constexpr (config_type::benchmark || config_type::dev) {
				print_performance_stats();
			}
		}

	  protected:
		optional<memory_mapped_file<config_type>> metadata_memory{};
		optional<memory_mapped_file<config_type>> weight_memory{};
		jitter_generator prompt_eval_time_jitter{};
		memory_buffer<config_type> memory{};
		jitter_generator eval_time_jitter{};
		execution_parameters exec_params{};

		NIHILUS_HOST int32_t sample_next_token() {
			//auto& result_output_tensor = get_core<core_types, core_types::result_output>();
			//double* logits			   = static_cast<double*>(result_output_tensor.get_data());
			//uint64_t vocab_size		   = model_traits_type::vocab_size;
			return {};
		}

		NIHILUS_HOST void print_performance_stats() {
			if constexpr (config_type::benchmark || config_type::dev) {

				double prompt_eval_time_ms = perf_base<config_type>::perf_stats.total_prompt_eval_time_ns * 1e-6;
				double eval_time_ms		   = perf_base<config_type>::perf_stats.total_eval_time_ns * 1e-6;

				prompt_eval_time_jitter += prompt_eval_time_ms;
				double prompt_eval_jitter			= prompt_eval_time_jitter.get_next_jitter_amount();
				double jittered_prompt_eval_time_ms = prompt_eval_time_ms + prompt_eval_jitter;

				eval_time_jitter += eval_time_ms;
				double eval_time_jitter_amount = eval_time_jitter.get_next_jitter_amount();
				double jittered_eval_time_ms   = eval_time_ms + eval_time_jitter_amount;

				double jittered_prompt_tokens_per_sec = static_cast<double>(perf_base<config_type>::perf_stats.prompt_token_count) / (jittered_prompt_eval_time_ms * 1e-3);
				double jittered_eval_tokens_per_sec	  = static_cast<double>(perf_base<config_type>::perf_stats.generated_token_count) / (jittered_eval_time_ms * 1e-3);

				std::cout << "\n=== NIHILUS TIMING ===" << std::endl;
				std::cout << "nihilus_perf_context_print:        load time = " << perf_base<config_type>::perf_stats.total_load_time_ns * 1e-6 << " ms" << std::endl;

				std::cout << "nihilus_perf_context_print: prompt eval time = " << prompt_eval_time_ms << " ms / " << perf_base<config_type>::perf_stats.prompt_token_count
						  << " tokens (" << prompt_eval_time_ms / static_cast<double>(perf_base<config_type>::perf_stats.prompt_token_count) << " ms per token, "
						  << static_cast<double>(perf_base<config_type>::perf_stats.prompt_token_count) / (perf_base<config_type>::perf_stats.total_prompt_eval_time_ns * 1e-9)
						  << " tokens per second)" << std::endl;

				std::cout << "nihilus_perf_context_print: prompt eval time (jittered) = " << jittered_prompt_eval_time_ms << " ms / "
						  << perf_base<config_type>::perf_stats.prompt_token_count << " tokens ("
						  << jittered_prompt_eval_time_ms / static_cast<double>(perf_base<config_type>::perf_stats.prompt_token_count) << " ms per token, "
						  << jittered_prompt_tokens_per_sec << " tokens per second)" << std::endl;

				std::cout << "nihilus_perf_context_print:        eval time = " << eval_time_ms << " ms / " << perf_base<config_type>::perf_stats.generated_token_count
						  << " runs   (" << eval_time_ms / static_cast<double>(perf_base<config_type>::perf_stats.generated_token_count) << " ms per token, "
						  << static_cast<double>(perf_base<config_type>::perf_stats.generated_token_count) / (perf_base<config_type>::perf_stats.total_eval_time_ns * 1e-9)
						  << " tokens per second)" << std::endl;

				std::cout << "nihilus_perf_context_print:        eval time (jittered) = " << jittered_eval_time_ms << " ms / "
						  << perf_base<config_type>::perf_stats.generated_token_count << " runs   ("
						  << jittered_eval_time_ms / static_cast<double>(perf_base<config_type>::perf_stats.generated_token_count) << " ms per token, "
						  << jittered_eval_tokens_per_sec << " tokens per second)" << std::endl;

				/*
				std::cout << "nihilus_perf_sampler_print:    sampling time = " << perf_base<config_type>::perf_stats.total_sampling_time_ns * 1e-6 << " ms / " << perf_base<config_type>::perf_stats.total_sampling_runs << " runs   ("
				<< perf_base<config_type>::perf_stats.total_sampling_time_ns * 1e-6 / static_cast<double>(perf_base<config_type>::perf_stats.total_sampling_runs) << " ms per token, "
				<< static_cast<double>(perf_base<config_type>::perf_stats.total_sampling_runs) / (perf_base<config_type>::perf_stats.total_sampling_time_ns * 1e-9) << " tokens per second)" << std::endl;
				*/
			}
		}

		NIHILUS_HOST void deinit() {
			memory.deinit();
		}

		NIHILUS_HOST ~model() {
			deinit();
		}
	};

	template<const model_config&... configs> struct model_collection_builder {
		template<size_t... Is> static auto build_impl(std::index_sequence<Is...>) {
			return model_collection<model<Is, model_config_type<configs>>...>{};
		}

		static auto build() {
			return build_impl(std::make_index_sequence<sizeof...(configs)>{});
		}

		using type = decltype(build());
	};

	template<const model_config&... configs> using model_collection_type = model_collection_builder<configs...>::type;

	template<const model_config&... configs> struct https_model_collection_builder {
		template<size_t... Is> static auto build_impl(std::index_sequence<Is...>) {
			return https_model_collection<model<Is, model_config_type<configs>>...>{};
		}

		static auto build() {
			return build_impl(std::make_index_sequence<sizeof...(configs)>{});
		}

		using type = decltype(build());
	};

	template<const model_config&... configs> using https_model_collection_type = https_model_collection_builder<configs...>::type;

}
