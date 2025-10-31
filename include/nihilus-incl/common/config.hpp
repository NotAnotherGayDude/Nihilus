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

#include <source_location>
#include <cstring>
#include <cstdint>
#include <utility>
#include <chrono>
#include <atomic>

#if NIHILUS_PLATFORM_WINDOWS
	#ifndef PATH_MAX
		#define PATH_MAX MAX_PATH
	#endif
	#include <io.h>
	#include <Windows.h>
#else
	#include <sys/mman.h>
	#include <sys/stat.h>
	#include <fcntl.h>
	#include <unistd.h>
	#if NIHILUS_PLATFORM_LINUX
		#include <sys/resource.h>
	#elif NIHILUS_PLATFORM_MAC
		#include <mach/mach.h>
		#include <TargetConditionals.h>
	#endif
#endif

#if NIHILUS_COMPILER_CUDA
	#define NIHILUS_ALIGN(x) __align__(x)
	#include <cuda_fp16.h>
	#include <cuda_bf16.h>
#else
	#define NIHILUS_ALIGN(x) alignas(x)
#endif

#if !defined(NIHILUS_LIKELY)
	#define NIHILUS_LIKELY(...) (__VA_ARGS__) [[likely]]
#endif

#if !defined(NIHILUS_UNLIKELY)
	#define NIHILUS_UNLIKELY(...) (__VA_ARGS__) [[unlikely]]
#endif

#if !defined(NIHILUS_ELSE_UNLIKELY)
	#define NIHILUS_ELSE_UNLIKELY(...) __VA_ARGS__ [[unlikely]]
#endif

#if NIHILUS_ARCH_X64
	#include <immintrin.h>
#elif NIHILUS_ARCH_ARM64
	#include <arm_sve.h>
	#include <arm_neon.h>
#else
	#error "Unsupported architecture"
#endif

namespace nihilus {

	using clock_type = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	static constexpr bool false_type_v = std::false_type::value;

	template<auto enum_error, typename... types> struct error_printer_impl;

	template<bool value, auto enum_error, typename... value_to_test> struct static_assert_printer {
		static constexpr bool impl{ [] {
			if constexpr (!value) {
				error_printer_impl<enum_error, value_to_test...>::failure_value;
				return false;
			} else {
				return true;
			}
		}() };
	};

	template<auto enum_error, auto... values> struct error_printer_impl_val;

	template<bool value, auto enum_error, auto... values> struct static_assert_printer_val {
		static constexpr bool impl{ [] {
			if constexpr (!value) {
				error_printer_impl_val<enum_error, values...>::failure_value;
				return false;
			} else {
				return true;
			}
		}() };
	};

	template<uint64_t byte_count, typename value_type01, typename value_type02> NIHILUS_HOST_DEVICE void constexpr_memcpy(value_type02* dst, const value_type01* src) {
		std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), byte_count);
	}

	template<typename value_type01, typename value_type02> NIHILUS_HOST_DEVICE void memcpy_wrapper(value_type02* dst, const value_type01* src, uint64_t byte_count) {
		std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), byte_count);
	}

	template<typename value_type> struct NIHILUS_ALIGN(64) static_aligned_const {
		NIHILUS_ALIGN(64) value_type value {};

		NIHILUS_HOST_DEVICE constexpr operator const value_type&() const& {
			return value;
		}

		NIHILUS_HOST_DEVICE constexpr operator value_type&() & {
			return value;
		}

		NIHILUS_HOST_DEVICE constexpr operator value_type&&() && {
			return std::move(value);
		}

		NIHILUS_HOST_DEVICE constexpr const value_type& operator*() const {
			return value;
		}

		NIHILUS_HOST_DEVICE constexpr value_type& operator*() {
			return value;
		}

		NIHILUS_HOST_DEVICE constexpr bool operator==(const static_aligned_const& other) const {
			return value == other.value;
		}

		NIHILUS_HOST_DEVICE constexpr bool operator!=(const static_aligned_const& other) const {
			return value != other.value;
		}

		NIHILUS_HOST_DEVICE constexpr bool operator<(const static_aligned_const& other) const {
			return value < other.value;
		}

		NIHILUS_HOST_DEVICE constexpr bool operator>(const static_aligned_const& other) const {
			return value > other.value;
		}
	};

	template<typename value_type> static_aligned_const(value_type) -> static_aligned_const<value_type>;

	enum class thread_strategy_types : uint8_t {
		none,
		global_input,
		global_output,
		per_block,
	};

	enum class allocation_strategy_types : uint8_t {
		none,
		mmap,
		remap,
		alloc,
	};

	enum class data_strategy_types : uint8_t {
		none,
		global,
		per_block,
	};

	enum class data_types : uint64_t {
		f32		= 0,
		f16		= 1,
		q4_0	= 2,
		q4_1	= 3,
		q5_0	= 6,
		q5_1	= 7,
		q8_0	= 8,
		q8_1	= 9,
		q2_k	= 10,
		q3_k	= 11,
		q4_k	= 12,
		q5_k	= 13,
		q6_k	= 14,
		q8_k	= 15,
		iq2_xxs = 16,
		iq2_xs	= 17,
		iq3_xxs = 18,
		iq1_s	= 19,
		iq4_nl	= 20,
		iq3_s	= 21,
		iq2_s	= 22,
		iq4_xs	= 23,
		i8		= 24,
		i16		= 25,
		i32		= 26,
		i64		= 27,
		f64		= 28,
		iq1_m	= 29,
		bf16	= 30,
		tq1_0	= 34,
		tq2_0	= 35,
		count	= 39,
	};

	enum class core_types : uint8_t {
		weights,
		global_inputs,
		token_embeddings,
		attn_prep_and_score,
		attn_out_and_ffn,
		global_output_and_sampling,
		count,
	};

	enum class kernel_types : uint8_t {
		weights,
		global_inputs,
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
		count,
	};

	enum class tensor_types {
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
		token_embeddings_get_rows,
		norm_rms_norm,
		attn_norm_mul,
		q_mul_mat,
		q_reshape,
		q_rope,
		k_mul_mat,
		k_reshape,
		k_rope,
		v_mul_mat,
		k_cache_view,
		k_cache_cpy,
		v_transpose,
		v_cache_view,
		v_cache_cpy,
		v_view,
		k_view,
		q_permute,
		kq_mul_mat,
		kq_soft_max,
		kqv_mul_mat,
		kqv_permute,
		kqv_cont,
		attn_output_mul_mat,
		ffn_inp_add,
		ffn_norm_rms_norm,
		ffn_norm_mul,
		ffn_gate_mul_mat,
		ffn_gate_silu,
		ffn_up_mul_mat,
		ffn_gate_par_mul,
		ffn_down_mul_mat,
		layer_out_add,
		l_out_get_rows,
		extracted_add,
		output_norm_rms_norm,
		output_norm_mul,
		output_projection_mul_mat,
		apply_repetition_penalty,
		apply_presence_penalty,
		apply_frequency_penalty,
		apply_logits_bias,
		apply_vocab_mask,
		apply_temperature,
		compute_softmax,
		apply_top_k_filter,
		apply_top_p_filter,
		sample_token,
		count
	};

	enum class mega_fused_kernel_types {
		token_embeddings,
		attn_prep_and_score,
		attn_out_and_ffn,
		ffn,
		global_output_and_sampling,
	};

	enum class user_input_types {
		direct_string,
		cin,
		managed,
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

	enum class rms_norm_types : uint8_t {
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

	enum class model_formats {
		nh_void,
		gguf,
		count,
	};

}
