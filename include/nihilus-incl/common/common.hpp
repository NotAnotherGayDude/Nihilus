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

#include <nihilus-incl/common/allocator.hpp>
#include <nihilus-incl/common/exception.hpp>
#include <nihilus-incl/cpu/simd/nihilus_cpu_instructions.hpp>
#include <nihilus-incl/common/data_types.hpp>
#include <nihilus-incl/common/concepts.hpp>
#include <nihilus-incl/common/array.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>

namespace nihilus {

	template<typename value_type> struct vector : public std::vector<value_type> {};

	template<typename value_type> struct aligned_vector : public std::vector<value_type, allocator<value_type, 64>> {};

	static constexpr array<bool, 256> alpha_table{ [] {
		array<bool, 256> return_values{};

		for (int32_t i = 'A'; i <= 'Z'; ++i) {
			return_values[static_cast<uint64_t>(i)] = true;
		}

		for (int32_t i = 'a'; i <= 'z'; ++i) {
			return_values[static_cast<uint64_t>(i)] = true;
		}

		return return_values;
	}() };

	template<typename value_type> NIHILUS_INLINE static constexpr bool is_alpha(value_type c) noexcept {
		return alpha_table[static_cast<uint8_t>(c)];
	}

	static constexpr array<bool, 256> space_table{ [] {
		array<bool, 256> return_values{};
		return_values[static_cast<uint64_t>('\r')] = true;
		return_values[static_cast<uint64_t>('\n')] = true;
		return_values[static_cast<uint64_t>(' ')]  = true;
		return_values[static_cast<uint64_t>('\t')] = true;
		return_values[static_cast<uint64_t>('\v')] = true;
		return_values[static_cast<uint64_t>('\f')] = true;
		return return_values;
	}() };

	template<typename value_type> NIHILUS_INLINE static constexpr bool is_space(value_type c) noexcept {
		return space_table[static_cast<uint8_t>(c)];
	}

	template<typename value_type> NIHILUS_INLINE static constexpr bool is_digit(value_type c) noexcept {
		return static_cast<uint8_t>(c - '0') < 10;
	}

	template<typename value_type> struct parse_core;

#if defined(NIHILUS_DEV) || defined(NIHILUS_BENCHMARK)
	static constexpr auto spinlock_time{ 172000 };
#endif

	struct alignas(64) atomic_flag_wrapper {
		NIHILUS_ALIGN(64) static constexpr uint32_t spin_cycles { 500000000 };
		using value_type										= typename std::atomic_signed_lock_free::value_type;
		NIHILUS_INLINE constexpr atomic_flag_wrapper() noexcept = default;
		NIHILUS_INLINE constexpr atomic_flag_wrapper& operator=(const atomic_flag_wrapper&) noexcept {
			return *this;
		}

		NIHILUS_INLINE constexpr atomic_flag_wrapper(const atomic_flag_wrapper&) noexcept {
		}

		NIHILUS_INLINE void store(int64_t value_new) {
			flag.store(static_cast<value_type>(value_new), std::memory_order_release);
		}

		NIHILUS_INLINE int64_t load() {
			return flag.load(std::memory_order_acquire);
		}

		NIHILUS_INLINE void clear() {
			flag.store(0, std::memory_order_release);
		}

		NIHILUS_INLINE void test_and_set() {
			flag.store(1, std::memory_order_release);
		}

		NIHILUS_INLINE void notify_one() {
			flag.notify_one();
		}

		NIHILUS_INLINE void notify_all() {
			flag.notify_all();
		}

		NIHILUS_INLINE int64_t fetch_add(int64_t value) {
			return flag.fetch_add(static_cast<value_type>(value), std::memory_order_seq_cst);
		}

		NIHILUS_INLINE int64_t fetch_sub(int64_t value) {
			return flag.fetch_sub(static_cast<value_type>(value), std::memory_order_seq_cst);
		}

		NIHILUS_INLINE bool test() {
			return flag.load(std::memory_order_acquire) == 1;
		}

		NIHILUS_INLINE void wait(int64_t value) {
			flag.wait(static_cast<value_type>(value), std::memory_order_acquire);
		}

		NIHILUS_INLINE void hybrid_wait(int64_t expected_value) {
			for (uint32_t i = 0; i < spin_cycles; ++i) {
				if (flag.load(std::memory_order_acquire) == expected_value) {
					return;
				}
				nihilus_pause();
			}
			while (flag.load(std::memory_order_acquire) != static_cast<value_type>(expected_value)) {
				flag.wait(static_cast<value_type>(expected_value + 1), std::memory_order_acquire);
			}
		}

	  protected:
		alignas(64) std::atomic_signed_lock_free flag{};
	};

	template<bool all_blocking> struct linked_latch;

	struct linked_latch_base {
		alignas(64) int64_t thread_count{};
		atomic_flag_wrapper completed_dependencies{};
		atomic_flag_wrapper completed_workers{};
		atomic_flag_wrapper in_process_workers{};
		alignas(64) int64_t dependent_count{};
		alignas(64) int64_t depended_count{};
		alignas(64) array<linked_latch_base*, 4> dependents{};

		NIHILUS_INLINE void init(int64_t thread_count_new) {
			thread_count = thread_count_new;
		}

		NIHILUS_INLINE void reset(int64_t global_input_count) {
			in_process_workers.store(0);
			completed_workers.store(0);
			completed_dependencies.store(global_input_count);
		}

	  protected:
		constexpr ~linked_latch_base() {};
	};

	template<> struct linked_latch<false> : public linked_latch_base {
		static constexpr bool blocking{ false };
		NIHILUS_INLINE constexpr linked_latch() = default;

		NIHILUS_INLINE bool is_ready(int64_t& thread_index) {
			if (completed_dependencies.load() == depended_count) {
				thread_index		= in_process_workers.fetch_add(1);
				return thread_index < thread_count;
			} else {
				completed_dependencies.hybrid_wait(depended_count);
				thread_index		= in_process_workers.fetch_add(1);
				return thread_index < thread_count;
			}
		}

		NIHILUS_INLINE void complete_work() {
			if (completed_workers.fetch_add(1) == thread_count - 1) {
				for (int64_t x = 0; x < dependent_count; ++x) {
					auto comp_deps = dependents[x]->completed_dependencies.fetch_add(1) + 1;
					if (comp_deps == dependents[x]->depended_count) {
						dependents[x]->completed_dependencies.notify_all();
					}
				}
			}
		}

		NIHILUS_INLINE constexpr ~linked_latch() noexcept {};
	};

	template<> struct linked_latch<true> : public linked_latch_base {
		static constexpr bool blocking{ true };
		NIHILUS_INLINE constexpr linked_latch() = default;

		NIHILUS_INLINE int64_t is_ready(int64_t& thread_index) {
			if (completed_dependencies.load() >= depended_count) {
				thread_index = in_process_workers.fetch_add(1);
				return thread_count;
			}

			completed_dependencies.hybrid_wait(depended_count);
			thread_index = in_process_workers.fetch_add(1);
			return thread_count;
		}

		NIHILUS_INLINE void complete_work() {
			if (completed_workers.fetch_add(1) == thread_count - 1) {
				for (int64_t x = 0; x < dependent_count; ++x) {
					auto comp_deps = dependents[x]->completed_dependencies.fetch_add(1) + 1;
					if (comp_deps == dependents[x]->depended_count) {
						dependents[x]->completed_dependencies.notify_all();
					}
				}
				completed_workers.notify_all();
			} else {
				completed_workers.hybrid_wait(thread_count);
			}
		}

		NIHILUS_INLINE constexpr ~linked_latch() noexcept {};
	};

	template<typename value_type> using latch_type = std::remove_cvref_t<decltype(std::remove_cvref_t<value_type>::latch[0])>;

	template<typename value_type>
	concept has_latch = requires(std::remove_cvref_t<value_type> value) { value.latch; };

	template<typename value_type> static constexpr bool blocking = latch_type<value_type>::blocking;

	template<typename value_type>
	concept all_blocking = has_latch<value_type> && blocking<value_type>;

	struct alignas(64) op_latch {
		NIHILUS_INLINE op_latch() = default;
		atomic_flag_wrapper global_flag{};
		alignas(64) int64_t thread_count{};

		NIHILUS_INLINE void init(int64_t thread_count_new) {
			thread_count = thread_count_new;
			global_flag.store(0);
		}

		NIHILUS_INLINE int64_t arrive_and_wait_get_thread(int64_t& thread_index) {
			thread_index = global_flag.fetch_add(1);
			bool wait{ (thread_index < thread_count - 1) };

			if (wait) {
				global_flag.hybrid_wait(thread_count);
			} else {
				global_flag.notify_all();
				global_flag.store(0);
			}
			return thread_count;
		}

		NIHILUS_INLINE void arrive_and_wait() {
			auto thread_index = global_flag.fetch_add(1);
			bool wait{ (thread_index < thread_count - 1) };

			if (wait) {
				global_flag.hybrid_wait(thread_count);
			} else {
				global_flag.notify_all();
				global_flag.store(0);
			}
			return;
		}
	};

	struct alignas(64) main_gate_latch {
		NIHILUS_INLINE main_gate_latch()								  = default;
		NIHILUS_INLINE main_gate_latch& operator=(const main_gate_latch&) = delete;
		NIHILUS_INLINE main_gate_latch(const main_gate_latch&)			  = delete;
		vector<atomic_flag_wrapper> finish_flags{};
		vector<atomic_flag_wrapper> start_flags{};
		atomic_flag_wrapper global_counter{};
		alignas(64) int64_t thread_count{};

		NIHILUS_INLINE void init(int64_t thread_count_new) {
			thread_count = thread_count_new;
			start_flags.resize(static_cast<uint64_t>(thread_count));
			finish_flags.resize(static_cast<uint64_t>(thread_count));
			global_counter.store(thread_count);
		}

		NIHILUS_INLINE void worker_wait(uint64_t thread_index) {
			start_flags[thread_index].wait(false);
			start_flags[thread_index].clear();
		}

		NIHILUS_INLINE void arrive_and_wait(uint64_t thread_index) {
			global_counter.fetch_sub(1);
			global_counter.notify_one();

			while (!finish_flags[thread_index].test()) {
				nihilus_pause();
			}
			finish_flags[thread_index].clear();
		}

		NIHILUS_INLINE void count_down() {
			for (int64_t x = 0; x < thread_count; ++x) {
				start_flags[static_cast<uint64_t>(x)].test_and_set();
				start_flags[static_cast<uint64_t>(x)].notify_one();
			}
		}

		NIHILUS_INLINE void main_wait() {
			int64_t current_value = global_counter.load();
			while (current_value > 0) {
				global_counter.wait(current_value + 1);
				current_value = global_counter.load();
			}

			global_counter.store(static_cast<int64_t>(thread_count));
			for (int64_t x = 0; x < thread_count; ++x) {
				finish_flags[static_cast<uint64_t>(x)].test_and_set();
				finish_flags[static_cast<uint64_t>(x)].notify_one();
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

		NIHILUS_INLINE stop_watch(uint64_t newTime) noexcept {
			total_time_units.store(time_type{ newTime }, std::memory_order_release);
		}

		NIHILUS_INLINE stop_watch& operator=(stop_watch&& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_INLINE stop_watch(stop_watch&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE stop_watch& operator=(const stop_watch& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_INLINE stop_watch(const stop_watch& other) noexcept {
			*this = other;
		}

		NIHILUS_INLINE bool has_time_elapsed() noexcept {
			return ((get_current_time() - start_time_units.load(std::memory_order_acquire)) >= total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_INLINE void add_time() noexcept {
			std::unique_lock<std::mutex> lock{ mutex };
			values.emplace_back(total_time_elapsed());
			reset();
		}

		NIHILUS_INLINE uint64_t get_average() noexcept {
			std::unique_lock<std::mutex> lock{ mutex };
			uint64_t total_time{};
			for (auto& value: values) {
				total_time += get_value_as_uint(value);
			}
			return total_time / ((values.size() > 0) ? values.size() : 1);
		}

		NIHILUS_INLINE uint64_t get_count() noexcept {
			return values.size();
		}

		NIHILUS_INLINE void reset(time_type newTimeValue = time_type{}) noexcept {
			if NIHILUS_LIKELY (newTimeValue != time_type{}) {
				total_time_units.store(newTimeValue, std::memory_order_release);
			}
			start_time_units.store(get_current_time(), std::memory_order_release);
		}

		NIHILUS_INLINE uint64_t get_total_wait_time() const noexcept {
			return get_value_as_uint(total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_INLINE time_type total_time_elapsed() noexcept {
			return get_current_time() - start_time_units.load(std::memory_order_acquire);
		}

		NIHILUS_INLINE uint64_t total_time_elapsed_uint64() noexcept {
			return get_value_as_uint(get_current_time()) - get_value_as_uint(start_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_INLINE ~stop_watch() {
		}

	  protected:
		std::atomic<time_type> total_time_units{};
		std::atomic<time_type> start_time_units{};
		vector<time_type> values{};
		std::mutex mutex{};

		NIHILUS_INLINE time_type get_current_time() {
			if constexpr (lock_free) {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch());
			} else {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch()).count();
			}
		}

		NIHILUS_INLINE uint64_t get_value_as_uint(time_type time) {
			if constexpr (lock_free) {
				return time.count();
			} else {
				return time;
			}
		}
	};

	template<auto current_index, auto enum_count> NIHILUS_INLINE constexpr std::string_view get_enum_name() {
		std::string_view return_string{ std::source_location::current().function_name() };
		auto new_size	   = std::size("get_enum_name<");
		uint64_t new_index = return_string.find("get_enum_name<") + new_size - 1;
		return_string	   = return_string.substr(new_index, return_string.size() - new_index);
		return_string	   = return_string.substr(0, return_string.find(','));
		return return_string;
	}

	template<auto current_index, auto enum_count> NIHILUS_INLINE std::string print_enum_value(auto enum_val) {
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

	template<typename enum_type>
		requires(std::is_same_v<data_types, enum_type>)
	NIHILUS_INLINE constexpr const char* get_type_name(enum_type type) {
		switch (type) {
			case enum_type::f64: {
				return "double";
			}
			case enum_type::f32: {
				return "float_32";
			}
			case enum_type::f16: {
				return "float_16";
			}
			case enum_type::q8_0: {
				return "q8_0";
			}
			case enum_type::i64: {
				return "int64_t";
			}
			case enum_type::i32: {
				return "int32_t";
			}
			case enum_type::i16: {
				return "int16_t";
			}
			case enum_type::i8: {
				return "int8_t";
			}
			default: {
				return "count";
			}
		}
	}

	enum class kernel_types : uint8_t {
		none,
		embedding_lookup,
		fused_qkv_rope,
		fused_attention,
		fused_attn_out,
		fused_swiglu,
		fused_final_proj,
		count,
	};

	template<typename enum_type>
		requires(std::is_same_v<kernel_types, enum_type>)
	NIHILUS_INLINE std::ostream& operator<<(std::ostream& os, enum_type type) {
		switch (type) {
			default: {
				os << "count";
				return os;
			}
		}
	}

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
		inp_tokens,
		inp_pos,
		inp_out_ids,
		cache_k,
		cache_v,
		kq_mask,
		embedding_lookup,
		attention_preprocessing,
		attention_computation,
		attention_output,
		swiglu_transformation,
		final_projection,
		count
	};

	template<typename enum_type>
		requires(std::is_same_v<op_types, enum_type>)
	NIHILUS_INLINE std::ostream& operator<<(std::ostream& os, enum_type type) {
		switch (type) {
			case enum_type::token_embd_weight:
				return os << "token_embd_weight";
			case enum_type::rope_freqs_weight:
				return os << "rope_freqs_weight";
			case enum_type::output_weight:
				return os << "output_weight";
			case enum_type::output_norm_weight:
				return os << "output_norm_weight";
			case enum_type::attn_q_weight:
				return os << "attn_q_weight";
			case enum_type::attn_k_weight:
				return os << "attn_k_weight";
			case enum_type::attn_v_weight:
				return os << "attn_v_weight";
			case enum_type::attn_output_weight:
				return os << "attn_output_weight";
			case enum_type::attn_norm_weight:
				return os << "attn_norm_weight";
			case enum_type::ffn_gate_weight:
				return os << "ffn_gate_weight";
			case enum_type::ffn_up_weight:
				return os << "ffn_up_weight";
			case enum_type::ffn_down_weight:
				return os << "ffn_down_weight";
			case enum_type::ffn_norm_weight:
				return os << "ffn_norm_weight";
			case enum_type::inp_tokens:
				return os << "inp_tokens";
			case enum_type::inp_pos:
				return os << "inp_pos";
			case enum_type::inp_out_ids:
				return os << "inp_out_ids";
			case enum_type::cache_k:
				return os << "cache_k";
			case enum_type::cache_v:
				return os << "cache_v";
			case enum_type::kq_mask:
				return os << "kq_mask";
			case enum_type::embedding_lookup:
				return os << "embedding_lookup";
			case enum_type::attention_preprocessing:
				return os << "attention_preprocessing";
			case enum_type::attention_computation:
				return os << "attention_computation";
			case enum_type::attention_output:
				return os << "attention_output";
			case enum_type::swiglu_transformation:
				return os << "swiglu_transformation";
			case enum_type::final_projection:
				return os << "final_projection";
			case enum_type::count:
			default:
				return os << "count";
		}
	}

	template<enum_types enum_type> constexpr kernel_types get_kernel_type_from_llm_op(enum_type op) {
		switch (op) {
			case enum_type::inp_tokens:
			case enum_type::inp_pos:
			case enum_type::inp_out_ids:
			case enum_type::token_embd_weight:
			case enum_type::rope_freqs_weight:
			case enum_type::output_weight:
			case enum_type::output_norm_weight:
			case enum_type::attn_q_weight:
			case enum_type::attn_k_weight:
			case enum_type::attn_v_weight:
			case enum_type::attn_output_weight:
			case enum_type::attn_norm_weight:
			case enum_type::ffn_gate_weight:
			case enum_type::ffn_up_weight:
			case enum_type::ffn_down_weight:
			case enum_type::ffn_norm_weight:
			case enum_type::cache_k:
			case enum_type::cache_v:
			case enum_type::kq_mask:
				return kernel_types::none;
			case enum_type::inp_embd:
			case enum_type::embedding_lookup:
				return kernel_types::embedding_lookup;
			case enum_type::attention_preprocessing:
				return kernel_types::fused_qkv_rope;
			case enum_type::attention_computation:
				return kernel_types::fused_attention;
			case enum_type::attention_output:
				return kernel_types::fused_attn_out;
			case enum_type::swiglu_transformation:
				return kernel_types::fused_swiglu;
			case enum_type::final_projection:
				return kernel_types::fused_final_proj;
			case enum_type::count:
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
		deci,
		falcon,
		baichuan,
		grok,
		gpt2,
		gptj,
		gptneox,
		mpt,
		starcoder,
		refact,
		bert,
		nomic_bert,
		jina_bert_v2,
		bloom,
		stablelm,
		qwen,
		qwen2,
		qwen2moe,
		qwen2vl,
		phi2,
		phi3,
		phimoe,
		plamo,
		codeshell,
		orion,
		internlm2,
		minicpm,
		minicpm3,
		gemma,
		gemma2,
		starcoder2,
		mamba,
		xverse,
		command_r,
		cohere2,
		dbrx,
		olmo,
		olmo2,
		olmoe,
		openelm,
		arctic,
		deepseek,
		deepseek2,
		chatglm,
		bitnet,
		t5,
		t5encoder,
		jais,
		nemotron,
		exaone,
		rwkv6,
		rwkv6qwen2,
		granite,
		granite_moe,
		chameleon,
		wavtokenizer_dec,
		unknown,
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
		llm_unknown,
		llm_14M,
		llm_17M,
		llm_22M,
		llm_33M,
		llm_60M,
		llm_70M,
		llm_80M,
		llm_109M,
		llm_137M,
		llm_160M,
		llm_220M,
		llm_250M,
		llm_270M,
		llm_335M,
		llm_410M,
		llm_450M,
		llm_770M,
		llm_780M,
		llm_0_5B,
		llm_1B,
		llm_1_3B,
		llm_1_4B,
		llm_1_5B,
		llm_1_6B,
		llm_2B,
		llm_2_8B,
		llm_3B,
		llm_4B,
		llm_6B,
		llm_6_9B,
		llm_7B,
		llm_8B,
		llm_9B,
		llm_11B,
		llm_12B,
		llm_13B,
		llm_14B,
		llm_15B,
		llm_16B,
		llm_20B,
		llm_30B,
		llm_32B,
		llm_34B,
		llm_35B,
		llm_40B,
		llm_65B,
		llm_70B,
		llm_236B,
		llm_314B,
		llm_671B,
		llm_SMALL,
		llm_MEDIUM,
		llm_LARGE,
		llm_XL,
		llm_A1_7B,
		llm_A2_7B,
		llm_8x7B,
		llm_8x22B,
		llm_16x12B,
		llm_16x3_8B,
		llm_10B_128x3_66B,
		llm_57B_A14B,
		llm_27B,
		count,
	};

	enum class tokenizer_types {
		none,
		spm,
		bpe,
		wpm,
		ugm,
		rwkv,
	};

	enum class tokenizer_pre_types {
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

	enum class model_format {
		nh_void,
		gguf,
	};

	struct model_config {
		model_generations model_generation{};
		model_sizes model_size{};
		kernel_type_profiles kernel_profile{};
		model_arches arch{};
		bool exceptions{};
		std::istream* input_stream{};
		uint64_t max_thread_count{};
		uint64_t cpu_arch_index{};
		uint64_t default_max_context_length{};
		kv_cache_strategies cache_strategy{};
		bool use_gradient_checkpointing{};
		rope_scaling_types rope_scaling{};
		tokenizer_pre_types tokenizer_pre_type{};
		uint64_t kv_cache_block_size{};
		bool use_rotary_embeddings{};
		bool use_flash_attention{};
		norm_types rms_norm_type{};
		tokenizer_types tokenizer_type{};
		model_format format{};
		float norm_epsilon{};
		bool benchmark{};
		bool dev{};
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

	struct execution_parameters {
		const int32_t* input_tokens{};
		uint64_t kv_cache_seq_len{};
		uint64_t position_offset{};
		uint64_t sequence_length{};
		uint64_t max_new_tokens{};
		std::string_view prompt{};
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

	template<model_config config> class file_loader {
	  public:
		explicit file_loader(const std::filesystem::path& filePath) {
			if (!std::filesystem::exists(filePath)) {
				static constexpr auto location = std::source_location::current();
				//nihilus_exception<config, "file_loader - Path does not exist", location>::impl(filePath.string());
			}

			std::ifstream file(filePath, std::ios::binary | std::ios::ate);
			if (!file) {
				static constexpr auto location = std::source_location::current();
				//nihilus_exception<config, "file_loader - Failed to open file", location>::impl();
			}

			const std::streamsize size = file.tellg();
			file.seekg(0, std::ios::beg);
			if (size != -1) {
				contents.resize(static_cast<uint64_t>(size));
				if (!file.read(contents.data(), size)) {
					static constexpr auto location = std::source_location::current();
					//nihilus_exception<config, "file_loader - Failed to read file", location>::impl();
				}
			}
		}

		operator const std::string_view() const noexcept {
			return contents;
		}

		uint64_t size() const noexcept {
			return contents.size();
		}

	  protected:
		std::string contents;
	};

	template<model_config config> class file_saver {
	  public:
		file_saver(const std::filesystem::path& path, const void* data, uint64_t size) {
			if (!data || size == 0) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_saver - Cannot save null or empty data to file: ", location>::impl(path.string());
			}

			std::ofstream file(path, std::ios::binary | std::ios::trunc);
			if (!file) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_saver - Cannot save null or empty data to file: ", location>::impl(path.string());
			}

			file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
			if (!file) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_saver - Cannot save null or empty data to file: ", location>::impl(path.string());
			}
		}
	};
}
