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
#include <nihilus-incl/infra/nihilus_cathedral.hpp>
#include <nihilus-incl/cpu/thread_pool.hpp>
#include <nihilus-incl/cuda/thread_pool.hpp>
#include <nihilus-incl/common/input_collector.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <nihilus-incl/infra/jitter_generator.hpp>
#include <algorithm>
#include <span>

namespace nihilus {

	template<typename... bases> struct model_collection : public bases... {
		template<typename arg_types> NIHILUS_HOST explicit model_collection(arg_types&& params) : bases(params)... {
		}

		NIHILUS_HOST model_collection() {
		}

		NIHILUS_HOST bool process_input() noexcept {
			return (... && bases::process_input_impl());
		}

		NIHILUS_HOST bool process_input(rt_string prompt) noexcept {
			return (... && bases::process_input_impl(prompt));
		}

		NIHILUS_HOST void wait() {
			std::cout << "[Model Collection] Waiting on " << sizeof...(bases) << " model(s)...\n";
			(bases::wait(), ...);
			std::cout << "[Model Collection] All models have completed\n";
		}

		NIHILUS_HOST void signal_shutdown() {
			std::cout << "[Model Collection] Signaling shutdown to all models...\n";
			(bases::signal_shutdown(), ...);
		}

		NIHILUS_HOST void stop_all() {
			std::cout << "[Model Collection] Stopping all models...\n";
			(bases::stop_all(), ...);
			std::cout << "[Model Collection] All models stopped\n";
		}
	};

	template<uint64_t index_new, typename... types> struct model;

	template<uint64_t index_new, typename config_type> struct model<index_new, config_type>
		: public input_collector<config_type>, public thread_pool<config_type>, public tokenizer<config_type, config_type::model_arch, config_type::tokenizer_type> {
		using thread_pool_type = thread_pool<config_type>;
		using nihilus_cathedral_type  = get_nihilus_cathedral_t<config_type>;
		using tokenizer_type   = tokenizer<config_type, config_type::model_arch, config_type::tokenizer_type>;

		NIHILUS_HOST model() noexcept {
		}

		NIHILUS_HOST model(cli_params params) : thread_pool<config_type>{ static_cast<int64_t>(params.thread_count) } {
			exec_params.token_count = params.n_tokens;
			init(params);
		}

		model& operator=(const model&) = delete;
		model(const model&)			   = delete;

		NIHILUS_HOST bool process_input_impl(rt_string& input, [[maybe_unused]] uint64_t seed_new = 0) {
			tokenizer_type::init_rng(seed_new);
			input = input.size() > config_type::max_sequence_length ? input.substr(0, config_type::max_sequence_length) : input;
			tokenizer_type::tokenizer_init(
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
			tokenizer_type::tokenizer_init(
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
						  << nihilus_cathedral_memory_plan<config_type>.peak_allocated_bytes << std::endl;
			} else {
				std::cout << "Model: " << model_traits_type<config_type>::name << "-" << kernel_type_profile_traits<config_type::kernel_type_profile>::name
						  << " (Nihilus-CUDA) Total Bytes Required for All Tensors + Weights + KV_cache at Context Length Of: " << config_type::max_sequence_length << ": "
						  << nihilus_cathedral_memory_plan<config_type>.peak_allocated_bytes << std::endl;
			}
			memory.init(nihilus_cathedral_memory_plan<config_type>.peak_allocated_bytes);
			this->template impl<memory_mapper>(nihilus_cathedral_memory_plan<config_type>, memory);
			array<array<void*, model_traits_type<config_type>::block_count>, weight_types::count> data{};
			weight_mapper<config_type, core_traits_old<config_type, core_types::weights>>::impl(*static_cast<core_traits_old<config_type, core_types::weights>*>(this), data);

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.load_start = clock_type::now();
			}
			model_parser<config_type>::parse_model(params.model_file, data, metadata_memory, weight_memory, *static_cast<tokenizer_type*>(this));

			if constexpr (config_type::benchmark || config_type::dev) {
				auto load_end										  = clock_type::now();
				perf_base<config_type>::perf_stats.total_load_time_ns = std::chrono::duration<double, std::nano>(load_end - perf_base<config_type>::perf_stats.load_start).count();
			}
			if constexpr (config_type::device_type == device_types::gpu) {
				weight_memory.~optional<file_loader<config_type>>();
			}
		}

		NIHILUS_HOST void execute_model(rt_string_view input) {
			if constexpr (config_type::dev) {
				++perf_base<config_type>::perf_stats.current_iteration;
			}

			exec_params.sequence_length = tokenizer_type::tokenize(input,
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_tokens>().get_data());

			using core_type_inp_pos = detail::remove_cvref_t<
				decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>())>;
			static array<typename core_type_inp_pos::output_type, config_type::max_sequence_length> inp_pos_values{ [] {
				array<typename core_type_inp_pos::output_type, config_type::max_sequence_length> return_values;
				for (uint64_t x = 0; x < config_type::max_sequence_length; ++x) {
					return_values[x] = static_cast<typename core_type_inp_pos::output_type>(x);
				}
				return return_values;
			}() };
			memory_transfer<config_type>::host_to_device(inp_pos_values.data(),
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>().get_data(),
				exec_params.sequence_length);

			using core_type_inp_out_ids = detail::remove_cvref_t<
				decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>())>;
			memory_transfer<config_type>::host_to_device(static_cast<typename core_type_inp_out_ids::output_type>(exec_params.sequence_length - 1),
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

			static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(2, 1);
			static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(exec_params.sequence_length, 1);

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
				static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::eval_time>(1, 1);

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
		optional<file_loader<config_type>> metadata_memory{};
		optional<file_loader<config_type>> weight_memory{};
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

		NIHILUS_HOST void generate_causal_mask() {
			using core_type = detail::remove_cvref_t<
				decltype(this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::kq_mask>())>;
			using output_type = typename core_type::output_type;

			output_type* mask_data =
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::kq_mask>().get_data();
			static constexpr auto dims = core_type::get_array();
			const uint64_t total_dims  = dims[0] * dims[1] * dims[2] * dims[3];
			output_type value{};
			for (uint64_t x = 0; x < total_dims; ++x) {
				if (x == 0 || x == 32 || x == 33) {
					value = 0.0f;
					memory_transfer<config_type>::host_to_device(value, mask_data + x);
				} else {
					value = -std::numeric_limits<output_type>::infinity();
					memory_transfer<config_type>::host_to_device(value, mask_data + x);
				}
			}
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

	template<typename request_type> struct batch_request_metadata {
		uint64_t output_logit_offset{};
		uint64_t input_token_offset{};
		request_type* request_ptr{};
		uint64_t kv_cache_offset{};
		uint64_t batch_index{};
	};

	template<typename config_type, typename request_type>
	NIHILUS_HOST void generate_request_metadata(batch_request_metadata<request_type>& metadata, request_type& request, uint64_t batch_index) {
		using mtt = model_traits_type<config_type>;
		static constexpr uint64_t kv_cache_size_per_request{ config_type::max_sequence_length * mtt::block_count * 2 * mtt::n_embd_kv_gqa };
		metadata.input_token_offset	 = batch_index * config_type::max_sequence_length;
		metadata.request_ptr		 = &request;
		metadata.kv_cache_offset	 = batch_index * kv_cache_size_per_request;
		metadata.output_logit_offset = batch_index * mtt::vocab_size;
		metadata.batch_index		 = batch_index;
	}

	template<uint64_t batch_size, typename request_type> struct batch_request_bucket {
		array<batch_request_metadata<request_type>, batch_size> requests{};
		uint64_t active_request_count{};
		uint64_t max_length{};
	};

	template<uint64_t batch_size, typename request_type> struct bucket_manager {
		array<batch_request_bucket<batch_size, request_type>, batch_size> buckets{};
		uint64_t active_bucket_count{};
	};

	template<uint64_t batch_size, typename request_type> NIHILUS_HOST void create_optimal_buckets(bucket_manager<batch_size, request_type>& manager,
		const array<batch_request_metadata<request_type>, batch_size>& metadata, uint64_t active_count) {
		manager.active_bucket_count = 0;
		for (uint64_t x = 0; x < batch_size; ++x) {
			manager.buckets[x].active_request_count = 0;
			manager.buckets[x].max_length			= 0;
		}
		if (active_count == 0) {
			return;
		}

		rt_array<size_t, batch_size> indices(active_count);
		for (size_t i = 0; i < active_count; ++i)
			indices[i] = i;

		std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
			return metadata[a].request_ptr->prompt_length < metadata[b].request_ptr->prompt_length;
		});

		uint64_t max_len = metadata[indices.back()].request_ptr->prompt_length;

		size_t num_buckets;
		if (max_len < 512 || active_count < 6) {
			num_buckets = detail::min(size_t(2), active_count);
		} else if (max_len < 2048) {
			num_buckets = detail::min(size_t(3) + active_count / 20, active_count);
		} else if (max_len < 8192) {
			num_buckets = detail::min(size_t(5) + active_count / 15, active_count);
		} else {
			num_buckets = detail::min(size_t(8) + active_count / 10, active_count);
		}

		num_buckets = std::clamp(num_buckets, size_t(1), detail::min(size_t(12), active_count));

		rt_array<rt_array<size_t, batch_size>, batch_size> dp(active_count + 1, rt_array<uint64_t, batch_size>(num_buckets + 1, std::numeric_limits<uint64_t>::max()));
		rt_array<rt_array<size_t, batch_size>, batch_size> splits(active_count + 1, rt_array<uint64_t, batch_size>(num_buckets + 1, 0));

		dp[0][0] = 0;

		for (size_t i = 1; i <= active_count; ++i) {
			for (size_t k = 1; k <= detail::min(i, num_buckets); ++k) {
				for (size_t j = k - 1; j < i; ++j) {
					if (dp[j][k - 1] == std::numeric_limits<uint64_t>::max())
						continue;

					uint64_t bucket_max_len = metadata[indices[i - 1]].request_ptr->prompt_length;
					uint64_t bucket_cost	= bucket_max_len * (i - j);
					uint64_t total_cost		= dp[j][k - 1] + bucket_cost;

					if (total_cost < dp[i][k]) {
						dp[i][k]	 = total_cost;
						splits[i][k] = j;
					}
				}
			}
		}

		rt_array<size_t, batch_size> split_points{ num_buckets };
		size_t pos = active_count;
		size_t k   = num_buckets;

		while (pos > 0 && k > 0) {
			split_points.emplace_back(splits[pos][k]);
			pos = splits[pos][k];
			k--;
		}

		std::reverse(split_points.begin(), split_points.end());
		split_points.emplace_back(active_count);

		size_t start				= 0;
		manager.active_bucket_count = 0;

		for (size_t end: split_points) {
			if (end > start) {
				auto& bucket				= manager.buckets[manager.active_bucket_count];
				bucket.active_request_count = 0;
				bucket.max_length			= 0;

				for (size_t i = start; i < end; ++i) {
					bucket.requests[bucket.active_request_count++] = metadata[indices[i]];
					bucket.max_length							   = detail::max(bucket.max_length, metadata[indices[i]].request_ptr->prompt_length);
				}

				manager.active_bucket_count++;
			}
			start = end;
		}

		return;
	}

	template<uint64_t index_new, typename config_type, typename connection_type> struct model<index_new, config_type, connection_type> : public model<index_new, config_type> {
		using base_model_type		 = model<index_new, config_type>;
		using output_queue_type		 = typename connection_type::output_queue_type;
		using input_queue_type		 = typename connection_type::input_queue_type;
		using response_type			 = typename connection_type::response_type;
		using request_type			 = typename connection_type::request_type;
		using tokenizer_type		 = typename base_model_type::tokenizer_type;
		using connection_config_type = typename connection_type::config_type;
		using flag_type				 = typename connection_type::flag_type;

		array<batch_request_metadata<request_type>, config_type::batch_size> metadata{};
		bucket_manager<config_type::batch_size, request_type> manager{};
		array<response_type*, config_type::batch_size> responses{};
		array<request_type*, config_type::batch_size> requests{};
		flag_type threads_running{};
		connection_type connection{};
		std::thread processor_thread{};

		NIHILUS_HOST model() {
		}

		NIHILUS_HOST connection_config_type get_connection_config(const cli_params& params) {
			connection_config_type return_values{};
			return_values.max_sequence_length = decltype(return_values.max_sequence_length){ config_type::max_sequence_length };
			return_values.max_queue_size	  = decltype(return_values.max_queue_size){ config_type::batch_size };
			return_values.port				  = decltype(return_values.port){ static_cast<uint16_t>(params.port + static_cast<uint16_t>(index_new)) };
			return_values.ip				  = decltype(return_values.ip){ params.ip };
			return return_values;
		}

		NIHILUS_HOST model(cli_params params) : base_model_type{ params }, threads_running{ false }, connection{ get_connection_config(params) } {
			processor_thread = std::thread(&model::processor_loop, this);
			threads_running.store(true, std::memory_order_release);
			logger<log_levels::status>::log("[Model] Initialized with connection threads");
		}

		NIHILUS_HOST ~model() {
			stop_all();
		}

		NIHILUS_HOST void wait() {
			logger<log_levels::status>::log("[Model] Main thread waiting for shutdown signal...");

			while (threads_running.load(std::memory_order_relaxed)) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

			logger<log_levels::status>::log("[Model] Shutdown signal received");
		}

		NIHILUS_HOST void stop_all() {
			bool was_running = threads_running.exchange(false, std::memory_order_acq_rel);
			if (!was_running)
				return;

			logger<log_levels::status>::log("[Model] Stopping threads...");

			connection.input_queue.wake_all();
			connection.output_queue.wake_all();

			if (processor_thread.joinable()) {
				processor_thread.join();
			}

			logger<log_levels::status>::log("[Model] Threads stopped");
		}

		NIHILUS_HOST void signal_shutdown() {
			logger<log_levels::status>::log("[Model] Shutdown signal received, stopping threads...");
			threads_running.store(false, std::memory_order_release);
		}

	  private:
		array<execution_parameters, config_type::batch_size> exec_params{};

		NIHILUS_HOST void execute_model_batched(uint64_t max_sequence_length, uint64_t batch_size) {
			static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(2, batch_size);
			static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::prompt_eval_time>(max_sequence_length, batch_size);

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

			for (uint64_t y = 0; y < base_model_type::exec_params.token_count - 1; ++y) {
				if constexpr (config_type::benchmark || config_type::dev) {
					perf_base<config_type>::perf_stats.token_start = clock_type::now();
				}
				static_cast<thread_pool<config_type>*>(this)->template execute_tasks<processing_phases::eval_time>(1, batch_size);

				if constexpr (config_type::benchmark || config_type::dev) {
					++perf_base<config_type>::perf_stats.current_iteration;
					auto token_end	   = clock_type::now();
					auto token_time_ns = std::chrono::duration<double, std::nano>(token_end - perf_base<config_type>::perf_stats.token_start).count();
					perf_base<config_type>::perf_stats.total_eval_time_ns += token_time_ns;
				}
				[[maybe_unused]] auto new_token = base_model_type::sample_next_token();
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
				base_model_type::print_performance_stats();
			}
		}

		NIHILUS_HOST void processor_loop() {
			logger<log_levels::status>::log("[Processor] Thread started");

			while (threads_running.load(std::memory_order_relaxed)) {
				auto input_slot = connection.input_queue.get_read_buffer();

				if (!input_slot) {
					if constexpr (config_type::dev) {
						logger<log_levels::status>::log("[Processor] Queue shutdown");
					}
					break;
				}

				request_type* request;
				uint64_t batch_size{};
				while (input_slot.clear_slot(request)) {
					if (!threads_running.load(std::memory_order_relaxed)) {
						break;
					}
					requests[batch_size] = request;
					generate_request_metadata<config_type, request_type>(metadata[batch_size], *request, batch_size);
					if constexpr (config_type::dev) {
						logger<log_levels::status>::log("[Processor] Processing request " + std::to_string(request->request_id));
					}
					++batch_size;
				}
				create_optimal_buckets<config_type::batch_size, request_type>(manager, metadata, batch_size);
				prep_all_requests();
				input_slot.mark_one_ready();
				if (batch_size > 0) {
					execute_all_requests();
					for (uint64_t x = 0; x < batch_size; ++x) {
						if (!threads_running.load(std::memory_order_relaxed)) {
							break;
						}
						auto output_slot = connection.output_queue.get_write_buffer();
						if (output_slot.add_slot(responses[x])) {
							output_slot.mark_one_ready();
						}
					}
				}
			}

			if constexpr (config_type::dev) {
				logger<log_levels::status>::log("[Processor] Thread stopped");
			}
		}

		NIHILUS_HOST void prep_input_impl(batch_request_metadata<request_type>& request) {
			auto& inp_pos_ref	 = this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_pos>();
			auto& inp_tokens_ref = this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_tokens>();
			auto& inp_out_ids_ref =
				this->template get_core<core_types, core_types::global_inputs>().values.template get_core<global_input_types, global_input_types::inp_out_ids>();
			tokenizer_type::init_rng(request.request_ptr->seed);
			tokenizer_type::tokenizer_init(inp_tokens_ref.get_data() + request.input_token_offset);
			using output_type = detail::remove_cvref_t<decltype(this->template get_core<core_types, core_types::global_inputs>()
					.values.template get_core<global_input_types, global_input_types::inp_tokens>())>::output_type;
			output_type val{ 1 };
			memory_transfer<config_type>::host_to_device(val, inp_pos_ref.get_data() + 1 + request.input_token_offset);
			memory_transfer<config_type>::host_to_device(val, inp_out_ids_ref.get_data() + request.input_token_offset);

			if constexpr (config_type::dev) {
				++perf_base<config_type>::perf_stats.current_iteration;
			}

			exec_params[request.batch_index].sequence_length =
				tokenizer_type::tokenize(request.request_ptr->prompt.data(), request.request_ptr->prompt.size(), inp_tokens_ref.get_data() + request.input_token_offset);

			using core_type_inp_pos = detail::remove_cvref_t<decltype(inp_pos_ref)>;
			static array<typename core_type_inp_pos::output_type, config_type::max_sequence_length> inp_pos_values{ [] {
				array<typename core_type_inp_pos::output_type, config_type::max_sequence_length> return_values{};
				for (uint64_t x = 0; x < config_type::max_sequence_length; ++x) {
					return_values[x] = x;
				}
				return return_values;
			}() };
			memory_transfer<config_type>::host_to_device(inp_pos_values.data(), inp_pos_ref.get_data() + request.input_token_offset,
				exec_params[request.batch_index].sequence_length);

			using core_type_inp_out_ids = detail::remove_cvref_t<decltype(inp_out_ids_ref)>;
			memory_transfer<config_type>::host_to_device(static_cast<typename core_type_inp_out_ids::output_type>(exec_params[request.batch_index].sequence_length - 1),
				inp_out_ids_ref.get_data() + request.input_token_offset + request.input_token_offset);

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.prompt_token_count	 = exec_params[request.batch_index].sequence_length;
				perf_base<config_type>::perf_stats.generated_token_count = exec_params[request.batch_index].token_count - 1;
				perf_base<config_type>::perf_stats.total_sampling_runs	 = exec_params[request.batch_index].token_count;
				perf_base<config_type>::perf_stats.total_eval_time_ns	 = 0;
			}

			if constexpr (config_type::benchmark || config_type::dev) {
				perf_base<config_type>::perf_stats.prompt_start = clock_type::now();
			}
		}

		NIHILUS_HOST void prep_request_data(batch_request_bucket<config_type::batch_size, request_type>& bucket) {
			for (uint64_t i = 0; i < bucket.active_request_count; ++i) {
				if (!threads_running.load(std::memory_order_relaxed)) {
					break;
				}
				prep_input_impl(bucket.requests[i]);
			}
		}

		NIHILUS_HOST void prep_all_requests() {
			for (uint64_t i = 0; i < manager.active_bucket_count; ++i) {
				if (!threads_running.load(std::memory_order_relaxed)) {
					break;
				}
				prep_request_data(manager.buckets[i]);
			}
		}

		NIHILUS_HOST void execute_all_requests() {
			for (uint64_t x = 0; x < manager.active_bucket_count; ++x) {
				if (!threads_running.load(std::memory_order_relaxed)) {
					break;
				}
				execute_model_batched(manager.buckets[x].max_length, manager.buckets[x].active_request_count);
			}
		}
	};

	template<template<bool> typename connection_type, const model_config&... configs> struct server_model_collection_builder {
		template<size_t... Is> static auto build_impl(std::index_sequence<Is...>) {
			return model_collection<model<Is, model_config_type<configs>, connection_type<model_config_type<configs>::dev>>...>{};
		}

		static auto build() {
			return build_impl(std::make_index_sequence<sizeof...(configs)>{});
		}

		using type = decltype(build());
	};

	template<template<bool> typename connection_type, const model_config&... configs> using server_model_collection_type =
		server_model_collection_builder<connection_type, configs...>::type;

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

}
