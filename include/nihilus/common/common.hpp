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

#include <nihilus/common/config.hpp>
#include <nihilus/cpu/simd/nihilus_cpu_instructions.hpp>
#include <nihilus/common/data_types.hpp>
#include <nihilus/common/concepts.hpp>
#include <nihilus/common/array.hpp>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>

namespace nihilus {

	template<bool exceptions> class file_loader {
	  public:
		explicit file_loader(const std::filesystem::path& filePath) {
			if (!std::filesystem::exists(filePath)) {
				if constexpr (exceptions) {
					throw std::runtime_error("File does not exist: " + filePath.string());
				} else {
					std::cerr << "File does not exist: " + filePath.string() << std::endl;
				}
			}

			std::ifstream file(filePath, std::ios::binary | std::ios::ate);
			if (!file) {
				if constexpr (exceptions) {
					throw std::runtime_error("Failed to open file: " + filePath.string());
				} else {
					std::cerr << "Failed to open file: " + filePath.string() << std::endl;
				}
			}

			const std::streamsize size = file.tellg();
			file.seekg(0, std::ios::beg);
			if (size != -1) {
				contents.resize(static_cast<uint64_t>(size));
				if (!file.read(contents.data(), size)) {
					if constexpr (exceptions) {
						throw std::runtime_error("Failed to read file: " + filePath.string());
					} else {
						std::cerr << "Failed to read file: " + filePath.string() << std::endl;
					}
				}
			}
		}

		operator const std::string&() const noexcept {
			return contents;
		}

		uint64_t size() const noexcept {
			return contents.size();
		}

	  protected:
		std::string contents;
	};

	template<bool exceptions> class file_saver {
	  public:
		file_saver(const std::filesystem::path& path, const void* data, uint64_t size) {
			if (!data || size == 0) {
				if constexpr (exceptions) {
					throw std::runtime_error("Cannot save null or empty data to file: " + path.string());
				} else {
					std::cerr << "Cannot save null or empty data to file: " + path.string() << std::endl;
				}
			}

			std::ofstream file(path, std::ios::binary | std::ios::trunc);
			if (!file) {
				if constexpr (exceptions) {
					throw std::runtime_error("Failed to open file for writing: " + path.string());
				} else {
					std::cerr << "Failed to open file for writing: " + path.string() << std::endl;
				}
			}

			file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
			if (!file) {
				if constexpr (exceptions) {
					throw std::runtime_error("Failed to write data to file: " + path.string());
				} else {
					std::cerr << "Failed to write data to file: " + path.string() << std::endl;
				}
			}
		}
	};

	static constexpr auto spinlock_time{ 500 };

	inline std::mutex mutex{};

	enum class log_level {
		error,
		status,
	};

	template<log_level level> NIHILUS_FORCE_INLINE void log(std::string_view string) {
		std::unique_lock lock{ mutex };
		if constexpr (level == log_level::error) {
			std::cerr << string << std::endl;
		} else {
			std::cout << string << std::endl;
		}
	}

	struct alignas(64) atomic_flag_wrapper {
		NIHILUS_FORCE_INLINE atomic_flag_wrapper() noexcept = default;
		NIHILUS_FORCE_INLINE atomic_flag_wrapper& operator=(const atomic_flag_wrapper&) noexcept {
			return *this;
		}

		NIHILUS_FORCE_INLINE atomic_flag_wrapper(const atomic_flag_wrapper&) noexcept {
		}

		NIHILUS_FORCE_INLINE void clear() {
			flag.store(0, std::memory_order_release);
		}

		NIHILUS_FORCE_INLINE void test_and_set() {
			flag.store(1, std::memory_order_release);
		}

		NIHILUS_FORCE_INLINE void notify_one() {
			flag.notify_one();
		}

		NIHILUS_FORCE_INLINE bool test() {
			return flag.load(std::memory_order_acquire) == 1;
		}

		NIHILUS_FORCE_INLINE void wait(bool value) {
			flag.wait(value, std::memory_order_acquire);
		}

	  protected:
		alignas(64) std::atomic_signed_lock_free flag{};
	};

	struct alignas(64) op_latch {
		NIHILUS_FORCE_INLINE op_latch()							  = default;
		NIHILUS_FORCE_INLINE op_latch& operator=(const op_latch&) = delete;
		NIHILUS_FORCE_INLINE op_latch(const op_latch&)			  = delete;
		alignas(64) std::vector<atomic_flag_wrapper> finish_flags{};
		char padding01[32];
		alignas(64) std::vector<atomic_flag_wrapper> start_flags{};
		char padding02[32];
		alignas(64) std::atomic_signed_lock_free global_counter{};
		char padding03[56];
		alignas(64) size_t thread_count{};

		NIHILUS_FORCE_INLINE void init(size_t thread_count_new) {
			thread_count = thread_count_new;
			start_flags.resize(thread_count);
			finish_flags.resize(thread_count);
			global_counter.store(static_cast<int64_t>(thread_count), std::memory_order_release);
		}

		NIHILUS_FORCE_INLINE void worker_wait(size_t thread_index) {
			while (!start_flags[thread_index].test()) {
				start_flags[thread_index].wait(false);
			}
			start_flags[thread_index].clear();
		}

		NIHILUS_FORCE_INLINE void arrive_and_wait(size_t thread_index) {
			global_counter.fetch_sub(1, std::memory_order_acq_rel);
			global_counter.notify_one();

			while (!finish_flags[thread_index].test()) {
				nihilus_pause();
			}
			finish_flags[thread_index].clear();
		}

		NIHILUS_FORCE_INLINE void count_down() {
			for (size_t x = 0; x < thread_count; ++x) {
				start_flags[x].test_and_set();
				start_flags[x].notify_one();
			}
		}

		NIHILUS_FORCE_INLINE void main_wait() {
			int64_t current_value = global_counter.load(std::memory_order_acquire);
			while (current_value > 0) {
				current_value = global_counter.load(std::memory_order_acquire);
				nihilus_pause();
			}

			global_counter.store(static_cast<int64_t>(thread_count), std::memory_order_release);
			for (size_t x = 0; x < thread_count; ++x) {
				finish_flags[x].test_and_set();
				finish_flags[x].notify_one();
			}
		}
	};

	template<typename value_type>
	concept time_t = is_specialization_v<value_type, std::chrono::duration>;

	template<time_t value_type = std::chrono::nanoseconds> class stop_watch {
	  public:
		using hr_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
		static constexpr bool lock_free{ std::atomic<value_type>::is_always_lock_free };
		using time_type = std::conditional_t<lock_free, value_type, uint64_t>;

		NIHILUS_FORCE_INLINE stop_watch(uint64_t newTime) noexcept {
			total_time_units.store(time_type{ newTime }, std::memory_order_release);
		}

		NIHILUS_FORCE_INLINE stop_watch& operator=(stop_watch&& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_FORCE_INLINE stop_watch(stop_watch&& other) noexcept {
			*this = std::move(other);
		}

		NIHILUS_FORCE_INLINE stop_watch& operator=(const stop_watch& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_FORCE_INLINE stop_watch(const stop_watch& other) noexcept {
			*this = other;
		}

		NIHILUS_FORCE_INLINE bool has_time_elapsed() noexcept {
			return ((get_current_time() - start_time_units.load(std::memory_order_acquire)) >= total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_FORCE_INLINE void add_time() noexcept {
			std::unique_lock lock{ mutex };
			values.emplace_back(total_time_elapsed());
			reset();
		}

		NIHILUS_FORCE_INLINE uint64_t get_average() noexcept {
			std::unique_lock lock{ mutex };
			uint64_t total_time{};
			for (auto& value: values) {
				total_time += get_value_as_uint(value);
			}
			return total_time / ((values.size() > 0) ? values.size() : 1);
		}

		NIHILUS_FORCE_INLINE uint64_t get_count() noexcept {
			return values.size();
		}

		NIHILUS_FORCE_INLINE void reset(time_type newTimeValue = time_type{}) noexcept {
			if NIHILUS_LIKELY (newTimeValue != time_type{}) {
				total_time_units.store(newTimeValue, std::memory_order_release);
			}
			start_time_units.store(get_current_time(), std::memory_order_release);
		}

		NIHILUS_FORCE_INLINE uint64_t get_total_wait_time() const noexcept {
			return get_value_as_uint(total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_FORCE_INLINE time_type total_time_elapsed() noexcept {
			return get_current_time() - start_time_units.load(std::memory_order_acquire);
		}

		NIHILUS_FORCE_INLINE uint64_t total_time_elapsed_uint64() noexcept {
			return get_value_as_uint(get_current_time()) - get_value_as_uint(start_time_units.load(std::memory_order_acquire));
		}

	  protected:
		std::atomic<time_type> total_time_units{};
		std::atomic<time_type> start_time_units{};
		std::vector<time_type> values{};
		std::mutex mutex{};

		NIHILUS_FORCE_INLINE time_type get_current_time() {
			if constexpr (lock_free) {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch());
			} else {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch()).count();
			}
		}

		NIHILUS_FORCE_INLINE uint64_t get_value_as_uint(time_type time) {
			if constexpr (lock_free) {
				return time.count();
			} else {
				return time;
			}
		}
	};

	inline stop_watch<std::chrono::nanoseconds> stop_watch_val_nihilus{ 0 };

	template<auto current_index, auto enum_count> NIHILUS_FORCE_INLINE constexpr std::string_view get_enum_name() {
		std::string_view return_string{ std::source_location::current().function_name() };
		auto new_size	   = std::size("get_enum_name<");
		uint64_t new_index = return_string.find("get_enum_name<") + new_size - 1;
		return_string	   = return_string.substr(new_index, return_string.size() - new_index);
		return_string	   = return_string.substr(0, return_string.find(','));
		return return_string;
	}

	template<auto current_index, auto enum_count> NIHILUS_FORCE_INLINE std::string print_enum_value(auto enum_val) {
		if constexpr (static_cast<uint64_t>(current_index) < static_cast<uint64_t>(enum_count)) {
			if (static_cast<uint64_t>(current_index) == static_cast<uint64_t>(enum_val)) {
				constexpr std::string_view string{ get_enum_name<current_index, enum_count>() };
				return static_cast<std::string>(string);
			} else {
				return print_enum_value<static_cast<decltype(enum_count)>(static_cast<uint64_t>(current_index) + 1), enum_count>(enum_val);
			}
		} else {
			return {};
		}
	};

	enum class data_types : uint64_t {
		f32	 = 0,
		f16	 = 1,
		q8_0 = 8,
		i8	 = 24,
		i16	 = 25,
		i32	 = 26,
		i64	 = 27,
		f64	 = 28,
		count,
	};

	NIHILUS_FORCE_INLINE constexpr const char* get_type_name(data_types type) {
		switch (type) {
			case data_types::f64: {
				return "double";
			}
			case data_types::f32: {
				return "float_32";
			}
			case data_types::f16: {
				return "float_16";
			}
			case data_types::q8_0: {
				return "q8_0";
			}
			case data_types::i64: {
				return "int64_t";
			}
			case data_types::i32: {
				return "int32_t";
			}
			case data_types::i16: {
				return "int16_t";
			}
			case data_types::i8: {
				return "int8_t";
			}
			case data_types::count: {
				return "count";
			}
		}
	}

	enum class kernel_types : uint8_t {
		none,
		add_rms_norm_mul,
		get_rows,
		rms_norm_mul,
		rms_norm,
		mul,
		mul_mat,
		reshape,
		permute,
		transpose,
		view,
		cont,
		copy,
		rope,
		softmax,
		silu,
		add,
		sub,
		count,
	};

	static constexpr array<const char*, kernel_types::count> kernel_names{ { "none", "get_rows", "rms_norm", "mul", "mul_mat", "reshape", "permute", "transpose", "view", "cont",
		"copy", "rope", "softmax", "silu", "add", "sub" } };

	enum class op_types : uint16_t {
		token_embd_weight,
		rope_freqs_weight,
		output_weight,
		output_norm_weight,
		attn_q_weight,
		attn_k_weight,
		attn_v_weight,
		attn_output_weight,
		attn_norm_weight,
		ffn_gate_weight,
		ffn_up_weight,
		ffn_down_weight,
		ffn_norm_weight,
		inp_embd,
		inp_tokens,
		inp_pos,
		inp_out_ids,
		cache_k,
		cache_v,
		kq_mask,
		norm_attn_norm,
		qcur,
		qcur_reshaped,
		qcur_rope,
		kcur,
		kcur_reshaped,
		kcur_rope,
		vcur,
		k_cache_view,
		k_cache_view_copy,
		vcur_transposed,
		v_cache_view,
		v_cache_view_copy,
		v,
		k,
		q,
		kq,
		kq_soft_max,
		kqv,
		kqv_merged,
		kqv_merged_cont,
		kqv_out,
		ffn_inp,
		ffn_inp_norm_out_ffn_norm,
		norm_out,
		ffn_norm,
		ffn_gate,
		ffn_silu,
		ffn_up,
		ffn_gate_par,
		ffn_out,
		l_out,
		attn_residual,
		prev_residual,
		final_norm,
		result_norm,
		result_output,
		count
	};

	static constexpr array<const char*, op_types::count> llama_op_names{ { "inp_embd", "token_embd_weight", "inp_tokens", "inp_pos", "inp_out_ids", "rope_freqs_weight",
		"output_weight", "output_norm_weight", "attn_q_weight", "attn_k_weight", "attn_v_weight", "attn_output_weight", "attn_norm_weight", "ffn_gate_weight", "ffn_up_weight",
		"ffn_down_weight", "ffn_norm_weight", "cache_k", "cache_v", "kq_mask", "norm", "attn_norm", "qcur", "qcur_reshaped", "qcur_rope", "kcur", "kcur_reshaped", "kcur_rope",
		"vcur", "k_cache_view", "k_cache_view_copy", "vcur_transposed", "v_cache_view", "v_cache_view_copy", "v", "k", "q", "kq", "kq_soft_max", "kqv", "kqv_merged",
		"kqv_merged_cont", "kqv_out", "ffn_inp", "norm_out", "ffn_norm", "ffn_gate", "ffn_silu", "ffn_up", "ffn_gate_par", "ffn_out", "l_out", "attn_residual", "prev_residual",
		"final_norm", "result_norm", "result_output" } };

	template<integral_or_enum value_type> constexpr kernel_types get_kernel_type_from_llama_op(value_type op) {
		switch (static_cast<op_types>(op)) {
			case op_types::inp_tokens:
			case op_types::inp_pos:
			case op_types::inp_out_ids:
			case op_types::token_embd_weight:
			case op_types::rope_freqs_weight:
			case op_types::output_weight:
			case op_types::output_norm_weight:
			case op_types::attn_q_weight:
			case op_types::attn_k_weight:
			case op_types::attn_v_weight:
			case op_types::attn_output_weight:
			case op_types::attn_norm_weight:
			case op_types::ffn_gate_weight:
			case op_types::ffn_up_weight:
			case op_types::ffn_down_weight:
			case op_types::ffn_norm_weight:
			case op_types::cache_k:
			case op_types::cache_v:
			case op_types::kq_mask:
				return kernel_types::none;
			case op_types::inp_embd:
			case op_types::attn_residual:
			case op_types::prev_residual:
				return kernel_types::get_rows;
			case op_types::norm_out:
			case op_types::ffn_norm:
			case op_types::final_norm:
				return kernel_types::rms_norm;
			case op_types::ffn_gate_par:
			case op_types::result_norm:
				return kernel_types::mul;
			case op_types::qcur:
			case op_types::kcur:
			case op_types::vcur:
			case op_types::kq:
			case op_types::kqv:
			case op_types::kqv_out:
			case op_types::ffn_gate:
			case op_types::ffn_up:
			case op_types::ffn_out:
			case op_types::result_output:
				return kernel_types::mul_mat;
			case op_types::qcur_reshaped:
			case op_types::kcur_reshaped:
				return kernel_types::reshape;
			case op_types::q:
			case op_types::kqv_merged:
				return kernel_types::permute;
			case op_types::vcur_transposed:
				return kernel_types::transpose;
			case op_types::k_cache_view:
			case op_types::v_cache_view:
			case op_types::v:
			case op_types::k:
				return kernel_types::view;
			case op_types::kqv_merged_cont:
				return kernel_types::cont;
			case op_types::k_cache_view_copy:
			case op_types::v_cache_view_copy:
				return kernel_types::copy;
			case op_types::qcur_rope:
			case op_types::kcur_rope:
				return kernel_types::rope;
			case op_types::kq_soft_max:
				return kernel_types::softmax;
			case op_types::ffn_silu:
				return kernel_types::silu;
			case op_types::ffn_inp:
			case op_types::l_out:
				return kernel_types::add;
			case op_types::norm_attn_norm:
				return kernel_types::rms_norm_mul;
			case op_types::ffn_inp_norm_out_ffn_norm:
				return kernel_types::add_rms_norm_mul;
			case op_types::count:
			default:
				return kernel_types::none;
		}
	}

	enum class device_types {
		cpu,
		gpu,
		numa,
	};

	enum class model_arches {
		llama,
		count,
	};

	enum class kernel_type_profiles : uint64_t {
		fp16_mha,
		fp16_moe,
		bf16_mha,
		bf16_gqa,
		q4_mha,
		q4_gqa,
		q4_moe,
		q8_mha,
		q8_gqa,
		q8_moe,
		mixed_fp16_fp32,
		mixed_bf16_fp32,
		count,
	};

	enum class norm_types : uint64_t {
		rms_standard,
		rms_parallel,
		rms_grouped,
		layer_norm_standard,
		layer_norm_no_bias,
		rms_norm_welford,
		adaptive_norm,
		count,
	};

	enum class kv_cache_strategies : uint64_t {
		contiguous,
		paged,
		compressed,
		streaming,
		hierarchical,
		count,
	};

	enum class rope_scaling_types : uint64_t {
		none,
		linear,
		dynamic,
		yarn,
		longrope,
		count,
	};

	enum class model_generations : uint64_t {
		v1_v2,
		v3,
		count,
	};

	enum class model_sizes {
		llama_1B,
		llama_3B,
		llama_7B,
		llama_8B,
		llama_11B,
		llama_13B,
		llama_70B,
		llama_90B,
		llama_405B,
		count,
	};

	static constexpr int32_t token_null{ -1 };

	enum class vocab_types {
		none,
		spm,
		bpe,
		wpm,
		ugm,
		rwkv,
	};

	enum class vocab_pre_types {
		default_pre,
		llama3,
		deepseek_llm,
		deepseek_coder,
		falcon,
		mpt,
		starcoder,
		gpt2,
		refact,
		command_r,
		stablelm2,
		qwen2,
		olmo,
		dbrx,
		smaug,
		poro,
		chatglm3,
		chatglm4,
		viking,
		jais,
		tekken,
		smollm,
		codeshell,
		bloom,
		gpt3_finnish,
		exaone,
		chameleon,
		minerva,
		deepseek3_llm,
	};

	enum class rope_types {
		none_rope = -1,
		norm,
		neox,
		mrope,
		vision,
	};

	enum class token_types {
		undefined_token,
		normal,
		unknown,
		control,
		user_defined,
		unused,
		byte,
	};

	enum class tokens {
		undefined	 = 0,
		unknown		 = 1 << 0,
		unused		 = 1 << 1,
		normal		 = 1 << 2,
		control		 = 1 << 3,
		user_defined = 1 << 4,
		byte		 = 1 << 5,
		normalized	 = 1 << 6,
		lstrip		 = 1 << 7,
		rstrip		 = 1 << 8,
		single_word	 = 1 << 9,
	};

	enum class model_format { gguf = 1 };

	template<auto> struct harbinger;

	struct model_config {
		model_generations model_generation{};
		model_sizes model_size{};
		kernel_type_profiles kernel_profile{};
		model_arches arch{};
		bool exceptions{};
		kv_cache_strategies cache_strategy{};
		bool use_gradient_checkpointing{};
		rope_scaling_types rope_scaling{};
		vocab_pre_types vocab_pre_type{};
		uint64_t kv_cache_block_size{};
		bool use_rotary_embeddings{};
		bool use_flash_attention{};
		norm_types rms_norm_type{};
		vocab_types vocab_type{};
		model_format format{};
		float norm_epsilon{};
		bool benchmark{};
	};

	struct cli_params {
		uint64_t thread_count{ std::thread::hardware_concurrency() };
		bool no_conversation{ false };
		uint64_t batch_size{ 512 };
		uint64_t n_predict{ 128 };
		std::string model_file{};
		uint64_t n_tokens{ 0 };
		std::string prompt{};
		uint64_t seed{ 0 };
	};

	struct impl_indices {
		uint64_t cpu_index{};
		uint64_t gpu_index{};
	};

	struct execution_parameters {
		const int32_t* input_tokens{};
		uint64_t kv_cache_seq_len{};
		uint64_t position_offset{};
		uint64_t sequence_length{};
		uint64_t max_new_tokens{};
		uint64_t thread_count{};
		uint64_t token_count{};
		uint64_t random_seed{};
		int32_t eos_token_id{};
		uint64_t sequence_id{};
		bool clear_kv_cache{};
		uint64_t batch_size{};
		float temperature{};
		bool is_prefill{};
		bool use_cache{};
		int32_t top_k{};
		float top_p{};
	};

	template<model_config config_new> struct config_holder {
		static constexpr model_config config{ config_new };
	};
}