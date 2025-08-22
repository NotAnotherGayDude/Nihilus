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
#include <nihilus-incl/common/vector.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>

namespace nihilus {

	template<typename value_type>
	concept time_t = is_specialization_v<value_type, std::chrono::duration>;

	template<time_t value_type = std::chrono::nanoseconds> class stop_watch {
	  public:
		using hr_clock = std::conditional_t<clock_type::is_steady, clock_type, std::chrono::steady_clock>;
		static constexpr static_aligned_const lock_free{ std::atomic<value_type>::is_always_lock_free };
		using time_type = std::conditional_t<lock_free, value_type, uint64_t>;

		NIHILUS_INLINE constexpr stop_watch() noexcept {
		}

		NIHILUS_INLINE stop_watch(uint64_t newTime) noexcept {
			total_time_units.store(newTime);
		}

		NIHILUS_INLINE constexpr stop_watch& operator=(stop_watch&& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_INLINE constexpr stop_watch(stop_watch&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE constexpr stop_watch& operator=(const stop_watch& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_INLINE constexpr stop_watch(const stop_watch& other) noexcept {
			*this = other;
		}

		NIHILUS_INLINE bool has_time_elapsed() noexcept {
			return ((get_current_time() - start_time_units.load(std::memory_order_acquire)) >= total_time_units.load(std::memory_order_acquire));
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

		NIHILUS_INLINE constexpr ~stop_watch() {
		}

	  protected:
		std::atomic<time_type> total_time_units{};
		std::atomic<time_type> start_time_units{};

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

	struct alignas(1024 * 8)
		jiff_the_jiping_jippalon_the_grand_jiper_of_the_jappaloneans_the_grand_chopper_of_jammals_onions_the_grand_jammer_of_jammals_vaccuum_the_grand_erector_of_jammals_pillar {};

	static constexpr jiff_the_jiping_jippalon_the_grand_jiper_of_the_jappaloneans_the_grand_chopper_of_jammals_onions_the_grand_jammer_of_jammals_vaccuum_the_grand_erector_of_jammals_pillar
		jiff{};

	template<auto value_new> struct make_static {
		static constexpr auto value{ value_new };
	};

	template<typename value_type> NIHILUS_INLINE static constexpr bool is_alpha(value_type c) noexcept {
		alignas(64) static constexpr const static_aligned_const<bool>* __restrict alpha_table{ [] constexpr {
			alignas(64) constexpr array<static_aligned_const<bool>, 256> return_values{ [] {
				array<static_aligned_const<bool>, 256> return_values{};
				for (int32_t i = 'A'; i <= 'Z'; ++i) {
					return_values[static_cast<uint64_t>(i)] = static_aligned_const<bool>{ true };
				}

				for (int32_t i = 'a'; i <= 'z'; ++i) {
					return_values[static_cast<uint64_t>(i)] = static_aligned_const<bool>{ true };
				}
				return return_values;
			}() };
			return make_static<return_values>::value.data();
		}() };
		return alpha_table[static_cast<uint8_t>(c)];
	}

	template<typename value_type> NIHILUS_INLINE static constexpr bool is_space(value_type c) noexcept {
		alignas(64) static constexpr const static_aligned_const<bool>* __restrict space_table{ [] {
			alignas(64) constexpr array<static_aligned_const<bool>, 256> return_values{ [] {
				array<static_aligned_const<bool>, 256> return_values{};
				return_values[static_cast<uint64_t>('\r')] = static_aligned_const<bool>{ true };
				return_values[static_cast<uint64_t>('\n')] = static_aligned_const<bool>{ true };
				return_values[static_cast<uint64_t>(' ')]  = static_aligned_const<bool>{ true };
				return_values[static_cast<uint64_t>('\t')] = static_aligned_const<bool>{ true };
				return_values[static_cast<uint64_t>('\v')] = static_aligned_const<bool>{ true };
				return_values[static_cast<uint64_t>('\f')] = static_aligned_const<bool>{ true };
				return return_values;
			}() };
			return make_static<return_values>::value.data();
		}() };
		return space_table[static_cast<uint8_t>(c)];
	}

	template<typename value_type> NIHILUS_INLINE static constexpr bool is_digit(value_type c) noexcept {
		return static_cast<uint8_t>(c - '0') < 10;
	}

	template<integral_or_enum_types value_type_new> struct alignas(64) atomic_flag_wrapper {
		static constexpr static_aligned_const spin_cycles{ 500000ull };
		using value_type										= typename std::atomic_signed_lock_free::value_type;
		NIHILUS_INLINE constexpr atomic_flag_wrapper() noexcept = default;
		NIHILUS_INLINE constexpr atomic_flag_wrapper& operator=(const atomic_flag_wrapper&) noexcept {
			return *this;
		}

		NIHILUS_INLINE constexpr atomic_flag_wrapper(const atomic_flag_wrapper&) noexcept {
		}

		NIHILUS_INLINE void store(value_type_new value_new) {
			flag.store(static_cast<value_type>(value_new), std::memory_order_release);
		}

		NIHILUS_INLINE value_type_new load() {
			return static_cast<value_type_new>(flag.load(std::memory_order_acquire));
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

		NIHILUS_INLINE value_type_new fetch_add(value_type value) {
			return static_cast<value_type_new>(flag.fetch_add(value, std::memory_order_seq_cst));
		}

		NIHILUS_INLINE value_type_new fetch_sub(value_type value) {
			return static_cast<value_type_new>(flag.fetch_sub(value, std::memory_order_seq_cst));
		}

		NIHILUS_INLINE bool test() {
			return flag.load(std::memory_order_acquire) == 1;
		}

		NIHILUS_INLINE void hybrid_wait(value_type expected) {
			for (uint32_t i = 0; i < spin_cycles; ++i) {
				if (flag.load(std::memory_order_acquire) == expected) {
					return;
				}
				nihilus_pause();
			}
			for (value_type v = flag.load(std::memory_order_acquire); v != expected; v = flag.load(std::memory_order_acquire)) {
				flag.wait(v, std::memory_order_acquire);
			}
		}

		NIHILUS_INLINE void wait() {
			value_type current_value_01{ flag.load(std::memory_order_acquire) };
			while (current_value_01 != 0) {
				current_value_01 = flag.load(std::memory_order_acquire);
				nihilus_pause();
			}
		}

	  protected:
		alignas(64) std::atomic_signed_lock_free flag{};
	};

	struct alignas(64) main_gate_latch {
		NIHILUS_INLINE main_gate_latch()								  = default;
		NIHILUS_INLINE main_gate_latch& operator=(const main_gate_latch&) = delete;
		NIHILUS_INLINE main_gate_latch(const main_gate_latch&)			  = delete;
		aligned_vector<atomic_flag_wrapper<int64_t>> flags{};
		atomic_flag_wrapper<int64_t> global_counter{};
		using value_type = std::atomic_signed_lock_free::value_type;
		alignas(64) value_type thread_count{};

		NIHILUS_INLINE void init(value_type thread_count_new) {
			thread_count = thread_count_new;
			flags.resize(thread_count);
			global_counter.store(thread_count);
		}

		NIHILUS_INLINE void worker_wait(uint64_t thread_index) {
			flags[thread_index].hybrid_wait(1);
			flags[thread_index].clear();
		}

		NIHILUS_INLINE void arrive(uint64_t thread_index) {
			global_counter.fetch_sub(1);
		}

		NIHILUS_INLINE void count_down() {
			for (uint64_t x = 0; x < static_cast<uint64_t>(thread_count); ++x) {
				flags[x].test_and_set();
				flags[x].notify_one();
			}
		}

		NIHILUS_INLINE void main_wait() {
			global_counter.wait();
			global_counter.store(thread_count);
		}
	};

	template<printable_enum_types auto current_index> consteval std::string_view get_enum_name() {
#if defined(NIHILUS_COMPILER_MSVC)
		alignas(64) constexpr char pretty_function_tail[]{ ">(void)" };
#else
		alignas(64) constexpr char pretty_function_tail[]{ "]" };
#endif
		std::string_view str = std::source_location::current().function_name();
#if defined(NIHILUS_COMPILER_GNUCXX)
		str			   = str.substr(str.find("=") + 2);
		uint64_t end   = str.find(';');
		str			   = str.substr(0, end);
		uint64_t start = str.find_last_of(':') + 1;
		return str.substr(start);
#else
		str			   = str.substr(str.find("=") + 2);
		uint64_t start = str.find_last_of(':') + 1;
		uint64_t end   = str.find(pretty_function_tail);
		return str.substr(start, end - start);
#endif
	}

	template<printable_enum_types current_type, size_t... I> consteval auto get_enum_names_impl(std::index_sequence<I...>) {
		return array<std::string_view, current_type::count>{ get_enum_name<static_cast<current_type>(I)>()... };
	}

	template<printable_enum_types current_type> consteval auto get_enum_names() {
		return get_enum_names_impl<current_type>(std::make_index_sequence<static_cast<size_t>(current_type::count)>{});
	}

	template<printable_enum_types enum_type> struct names {
		static constexpr auto data{ get_enum_names<enum_type>() };
	};

	template<printable_enum_types enum_type> NIHILUS_INLINE std::string_view get_name(enum_type type) {
		if (static_cast<uint64_t>(type) < names<enum_type>::data.size()) {
			return names<enum_type>::data[type];
		} else {
			return "Unknown Type.";
		}
	}

	template<printable_enum_types enum_type> NIHILUS_INLINE std::ostream& operator<<(std::ostream& os, enum_type type) {
		os << get_name(type);
		return os;
	}

	enum class data_types : uint64_t {
		f32	  = 0,
		f16	  = 1,
		q8_0  = 8,
		i8	  = 24,
		i16	  = 25,
		i32	  = 26,
		i64	  = 27,
		f64	  = 28,
		bf16  = 30,
		count = 39,
	};

	enum class core_types : uint8_t {
		weights,
		global_inputs,
		token_embeddings,
		mega_qkv_prep_and_cache_publish,
		mega_attention_apply,
		mega_ffn,
		final_norm_and_sampling,
		count,
	};

	enum class kernel_types : uint8_t {
		none,
		get_rows,
		rms_norm,
		mul,
		mul_mat,
		reshape,
		transpose,
		permute,
		view,
		rope,
		softmax,
		silu,
		copy,
		cont,
		add,
		sub,
		top_k_filter,
		top_p_filter,
		repetition_penalty,
		presence_penalty,
		temperature_scale,
		frequency_penalty,
		vocab_mask,
		sample_logits,
		count,
	};

	enum class composite_kernel_types : uint8_t {
		none,
		view,
		get_rows,
		mega_qkv_prep_and_cache,
		mega_attention_apply,
		mega_ffn,
		final_norm_and_sampling,
		count,
	};

	enum class weight_types : uint8_t {
		attn_q,
		attn_k,
		attn_v,
		attn_output,
		attn_norm,
		ffn_gate,
		ffn_up,
		ffn_down,
		ffn_norm,
		token_embd,
		rope_freqs,
		output_norm,
		output,
		count,
	};

	enum class global_input_types : uint8_t {
		inp_tokens,
		inp_pos,
		cache_k,
		cache_v,
		kq_mask,
		inp_out_ids,
		temperature,
		top_k,
		top_p,
		repetition_penalty,
		presence_penalty,
		frequency_penalty,
		rep_window,
		token_history,
		rng_state,
		logits_bias,
		allowed_vocab_mask,
		count,
	};

	enum class token_embedding_types : uint8_t {
		get_rows,
		count,
	};

	enum class mega_qkv_prep_and_cache_publish_types : uint8_t {
		q_out,
		count,
	};

	enum class mega_attention_apply_types {
		ffn_inp,
		count,
	};

	enum class mega_ffn_types {
		l_out,
		count,
	};

	enum class final_norm_and_sampling_types {
		result_token_id,
		count,
	};

	enum class global_output_types : uint8_t {
		result_output_composite,
		count,
	};

	enum class rope_and_cache_types : uint8_t {
		rope_q_permute_type,
		rope_k_copy_type,
		k_rope_view_type,
		v_rope_view_type,
		count,
	};

	enum class attention_scores_types : uint8_t {
		kq_scores_type,
		count,
	};

	enum class attention_weighted_values_types : uint8_t {
		attention_output_type,
		count,
	};

	enum class attention_output_projection_types : uint8_t {
		attn_output_type,
		count,
	};

	enum class ffn_parallel_projection_types : uint8_t {
		ffn_gate_type,
		ffn_up_type,
		count,
	};

	enum class ffn_down_projection_types : uint8_t {
		ffn_down_type,
		count,
	};

	enum class device_types : uint8_t {
		cpu,
		gpu,
		numa,
	};

	enum class model_arches : uint8_t {
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

	enum class kernel_type_profiles : uint8_t {
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

	enum class norm_types : uint8_t {
		rms_standard,
		rms_parallel,
		rms_grouped,
		layer_norm_standard,
		layer_norm_no_bias,
		rms_norm_welford,
		adaptive_norm,
		count,
	};

	enum class kv_cache_strategies : uint8_t {
		contiguous,
		paged,
		compressed,
		streaming,
		hierarchical,
		count,
	};

	enum class rope_scaling_types : uint8_t {
		none,
		linear,
		dynamic,
		yarn,
		longrope,
		count,
	};

	enum class model_generations : uint8_t {
		v1_v2,
		v3,
		v3_1,
		v3_2,
		count,
	};

	enum class model_sizes : uint8_t {
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
		llm_405B,
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

	enum class tokenizer_types : uint8_t {
		none,
		spm,
		bpe,
		wpm,
		ugm,
		rwkv,
		count,
	};

	enum class tokenizer_pre_types : uint8_t {
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
		count,
	};

	enum class rope_types : int8_t {
		none_rope = -1,
		norm,
		neox,
		mrope,
		vision,
		count,
	};

	enum class token_types : uint8_t {
		undefined_token,
		normal,
		unknown,
		control,
		user_defined,
		unused,
		byte,
		count,
	};

	enum class tokens : uint16_t {
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
		count,
	};

	enum class model_format {
		nh_void,
		gguf,
		count,
	};

	struct model_config {
		model_generations model_generation{};
		model_sizes model_size{};
		kernel_type_profiles kernel_profile{};
		model_arches arch{};
		bool exceptions{};
		std::istream* input_stream{};
		uint64_t default_max_sequence_length{};
		uint64_t default_batch_size{};
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
		static constexpr static_aligned_const config{ config_new };
	};

	template<model_config config> class file_loader;

	template<model_config config>
		requires(config.exceptions)
	class file_loader<config> {
	  public:
		explicit file_loader(const std::filesystem::path& filePath) {
			if (!std::filesystem::exists(filePath)) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_loader - Path does not exist", location>::impl(filePath.string());
			}

			std::ifstream file(filePath, std::ios::binary | std::ios::ate);
			if (!file) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_loader - Failed to open file", location>::impl();
			}

			const std::streamsize size = file.tellg();
			file.seekg(0, std::ios::beg);
			if (size != -1) {
				contents.resize(static_cast<uint64_t>(size));
				if (!file.read(contents.data(), size)) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "file_loader - Failed to read file", location>::impl();
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

	template<model_config config>
		requires(!config.exceptions)
	class file_loader<config> {
	  public:
		explicit file_loader(const std::filesystem::path& filePath) {
			if (!std::filesystem::exists(filePath)) {
				log<log_levels::error>("file_loader - Path does not exist");
			}

			std::ifstream file(filePath, std::ios::binary | std::ios::ate);
			if (!file) {
				log<log_levels::error>("file_loader - Failed to open file");
			}

			const std::streamsize size = file.tellg();
			file.seekg(0, std::ios::beg);
			if (size != -1) {
				contents.resize(static_cast<uint64_t>(size));
				if (!file.read(contents.data(), size)) {
					log<log_levels::error>("file_loader - Failed to read file");
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