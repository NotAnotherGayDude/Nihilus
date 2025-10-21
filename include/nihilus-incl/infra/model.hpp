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
		template<typename arg_types> NIHILUS_HOST explicit model_collection(arg_types&& params) : bases(params)... {
		}

		NIHILUS_HOST model_collection() {
		}

		NIHILUS_HOST bool process_input() noexcept {
			return (... && bases::process_input_impl());
		}

		NIHILUS_HOST bool process_input(std::string_view prompt) noexcept {
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
		using core_bases_type  = get_core_bases_t<config_type>;
		using tokenizer_type   = tokenizer<config_type, config_type::model_arch, config_type::tokenizer_type>;

		NIHILUS_HOST model() noexcept {
		}

		NIHILUS_HOST model(cli_params params) : thread_pool<config_type>{ static_cast<int64_t>(params.thread_count) } {
			exec_params.token_count = params.n_tokens;
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

	inline constexpr size_t round_to_power_of_two(size_t value) {
		if (value == 0)
			return 1;
		value--;
		value |= value >> 1;
		value |= value >> 2;
		value |= value >> 4;
		value |= value >> 8;
		value |= value >> 16;
		value |= value >> 32;
		return value + 1;
	}

	template<typename config_type, typename request_type> struct batch_queue {
		using request_ptr = request_type*;

		struct batch {
			array<request_ptr, config_type::batch_size> requests{};
			atomic_flag_wrapper<uint64_t> sequence{};
			atomic_flag_wrapper<uint64_t> count{};

			NIHILUS_HOST void add(request_ptr req) {
				requests[count.fetch_add(1)] = req;
			}

			NIHILUS_HOST void mark_ready(uint64_t seq) {
				sequence.store(seq);
				sequence.notify_one();
			}

			NIHILUS_HOST void wait_until_ready(uint64_t expected_seq) {
				sequence.hybrid_wait(expected_seq);
			}

			NIHILUS_HOST void reset() {
				count.store(0);
			}

			NIHILUS_HOST bool is_full() const {
				return count.load() >= config_type::batch_size;
			}

			NIHILUS_HOST bool is_empty() const {
				return count.load() == 0;
			}
		};

	  private:
		static constexpr auto batch_timeout = std::chrono::milliseconds{ 10 };
		atomic_flag_wrapper<uint64_t> currently_active_batch{};
		atomic_flag_wrapper<uint64_t> batch_start_time_ns{};
		atomic_flag_wrapper<uint64_t> global_sequence{};
		atomic_flag_wrapper<bool> shutdown{};
		array<batch, 2> current_batch{};

		NIHILUS_HOST uint64_t get_current_time_ns() {
			return std::chrono::duration_cast<std::chrono::duration<uint64_t, std::nano>>(clock_type::now().time_since_epoch()).count();
		}

	  public:
		NIHILUS_HOST bool try_add_request(request_ptr req) {
			uint64_t batch_idx = currently_active_batch.load() & 1;
			auto& batch		   = current_batch[batch_idx];

			if (batch.is_empty()) {
				batch_start_time_ns.store(get_current_time_ns());
			}

			if (!batch.is_full()) {
				batch.add(req);

				if (batch.is_full()) {
					uint64_t seq = global_sequence.fetch_add(1);
					batch.mark_ready(seq);
					return true;
				}

				uint64_t start_time	  = batch_start_time_ns.load();
				uint64_t current_time = get_current_time_ns();
				uint64_t elapsed_ns	  = current_time - start_time;

				if (elapsed_ns >= batch_timeout.count() * 1'000'000) {
					uint64_t seq = global_sequence.fetch_add(1);
					batch.mark_ready(seq);
				}

				return true;
			}

			return false;
		}

		NIHILUS_HOST batch* get_batch() {
			uint64_t old_idx	  = currently_active_batch.fetch_add(1);
			uint64_t batch_idx	  = old_idx & 1;
			uint64_t expected_seq = old_idx;

			current_batch[batch_idx].wait_until_ready(expected_seq);

			return &current_batch[batch_idx];
		}

		NIHILUS_HOST void signal_shutdown() {
			shutdown.store(true, std::memory_order_release);
			current_batch[0].sequence.notify_all();
			current_batch[1].sequence.notify_all();
		}
	};

	template<uint64_t index_new, typename config_type, typename connection_type> struct model<index_new, config_type, connection_type> : public model<index_new, config_type> {
		using output_queue_type		 = typename connection_type::output_queue_type;
		using input_queue_type		 = typename connection_type::input_queue_type;
		using response_type			 = typename connection_type::response_type;
		using request_type			 = typename connection_type::request_type;
		using connection_config_type = typename connection_type::config_type;
		using flag_type				 = typename connection_type::flag_type;
		using base_model_type		 = model<index_new, config_type>;

		flag_type threads_running{};
		connection_type connection{};
		std::thread output_writer_thread{};
		std::thread input_reader_thread{};

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
			threads_running.store(true, std::memory_order_release);
			input_reader_thread	 = std::thread(&model::input_reader_loop, this);
			output_writer_thread = std::thread(&model::output_writer_loop, this);

			log<log_levels::status>("[Model] Initialized with connection threads");
		}

		NIHILUS_HOST ~model() {
			stop_all();
		}

		NIHILUS_HOST void wait() {
			log<log_levels::status>("[Model] Main thread waiting for shutdown signal...");

			while (threads_running.load(std::memory_order_relaxed)) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

			log<log_levels::status>("[Model] Shutdown signal received");
		}

		NIHILUS_HOST void stop_all() {
			bool was_running = threads_running.exchange(false, std::memory_order_acq_rel);
			if (!was_running)
				return;

			log<log_levels::status>("[Model] Stopping threads...");

			connection.input_queue.wake_all();
			connection.output_queue.wake_all();

			if (input_reader_thread.joinable()) {
				input_reader_thread.join();
			}

			if (output_writer_thread.joinable()) {
				output_writer_thread.join();
			}

			log<log_levels::status>("[Model] Threads stopped");
		}

		NIHILUS_HOST void signal_shutdown() {
			log<log_levels::status>("[Model] Shutdown signal received, stopping threads...");
			threads_running.store(false, std::memory_order_release);
		}

	  private:
		NIHILUS_HOST void input_reader_loop() {
			log<log_levels::status>("[Input Reader] Thread started");

			while (threads_running.load(std::memory_order_relaxed)) {
				auto input_slot = connection.input_queue.get_read_buffer();
				if (!input_slot) {
					if constexpr (config_type::dev) {
						log<log_levels::status>("[Input Reader] Queue shutdown");
					}
					break;
				}

				if (!threads_running.load(std::memory_order_relaxed)) {
					break;
				}

				request_type* request;
				while (input_slot.clear_slot(request)) {
					if (!threads_running.load(std::memory_order_relaxed)) {
						break;
					}

					if constexpr (config_type::dev) {
						log<log_levels::status>("[Input Reader] Processing request " + std::to_string(request->request_id));
					}

					std::string_view prompt(request->prompt.data(), request->prompt_length);

					auto output_slot = connection.output_queue.get_write_buffer();
					response_type* response;
					if (output_slot.add_slot(response)) {
						process_inference_request(*request, prompt, *response);
						output_slot.mark_one_ready();
					}

					input_slot.mark_one_ready();
				}
			}

			if constexpr (config_type::dev) {
				log<log_levels::status>("[Input Reader] Thread stopped");
			}
		}

		NIHILUS_HOST void output_writer_loop() {
			log<log_levels::status>("[Output Writer] Thread started");

			while (threads_running.load(std::memory_order_relaxed)) {
				auto internal_slot = connection.output_queue.get_write_buffer();
				if (!internal_slot) {
					if constexpr (config_type::dev) {
						log<log_levels::status>("[Output Writer] Internal queue shutdown");
					}
					break;
				}

				if (!threads_running.load(std::memory_order_relaxed)) {
					break;
				}

				response_type* response;
				while (internal_slot.add_slot(response)) {
					if (!threads_running.load(std::memory_order_relaxed)) {
						break;
					}

					if constexpr (config_type::dev) {
						log<log_levels::status>("[Output Writer] Writing response " + std::to_string(response->response_id));
					}
					response->response_id	  = response->response_id;
					response->user_id		  = response->user_id;
					response->response_length = response->response_length;
					std::memcpy(response->response.data(), response->response.data(), response->response_length);
					response->eval_tokens_per_sec		 = response->eval_tokens_per_sec;
					response->prompt_eval_tokens_per_sec = response->prompt_eval_tokens_per_sec;
					response->tokens_generated			 = response->tokens_generated;
					response->prompt_tokens				 = response->prompt_tokens;
					response->total_time_ms				 = response->total_time_ms;

					internal_slot.mark_one_ready();
				}
			}

			log<log_levels::status>("[Output Writer] Thread stopped");
		}

		NIHILUS_HOST void process_inference_request(const request_type& request, std::string_view prompt, response_type& response) {
			base_model_type::process_input_impl(prompt, request.seed);

			response.response_id = request.request_id;
			response.user_id	 = request.user_id;

			const char* placeholder = "Generated response placeholder";
			size_t len				= std::strlen(placeholder);
			std::memcpy(response.response.data(), placeholder, len);
			response.response_length = len;

			if constexpr (config_type::benchmark || config_type::dev) {
				response.eval_tokens_per_sec = static_cast<uint64_t>(
					static_cast<double>(perf_base<config_type>::perf_stats.generated_token_count) / (perf_base<config_type>::perf_stats.total_eval_time_ns * 1e-9));

				response.prompt_eval_tokens_per_sec = static_cast<uint64_t>(
					static_cast<double>(perf_base<config_type>::perf_stats.prompt_token_count) / (perf_base<config_type>::perf_stats.total_prompt_eval_time_ns * 1e-9));

				response.tokens_generated = static_cast<uint32_t>(perf_base<config_type>::perf_stats.generated_token_count);
				response.prompt_tokens	  = static_cast<uint32_t>(perf_base<config_type>::perf_stats.prompt_token_count);
				response.total_time_ms	  = (perf_base<config_type>::perf_stats.total_prompt_eval_time_ns + perf_base<config_type>::perf_stats.total_eval_time_ns) * 1e-6;
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
