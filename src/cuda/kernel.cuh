#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstddef>
#include "test.hpp"
template<typename value_type> constexpr value_type&& forward(value_type& arg) noexcept {
	return static_cast<value_type&&>(arg);
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

enum class kernel_profiles : uint8_t {
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
#include <iostream>
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
enum class tokenizer_types : uint8_t {
	none,
	spm,
	bpe,
	wpm,
	ugm,
	rwkv,
	count,
};

enum class model_formats {
	nh_void,
	gguf,
	count,
};

enum class processing_phases {
	prompt_eval_time,
	eval_time,
};


struct model_config {
	model_generations model_generation{};
	model_sizes model_size{};
	kernel_profiles kernel_profile{};
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
	device_types device_type{};
	model_formats format{};
	float norm_epsilon{};
	bool benchmark{};
	bool dev{};
};


enum class model_format {
	nh_void,
	gguf,
	count,
};
template<const model_config& config_new> struct model_config_holder {
	static constexpr const model_config& config{ config_new };
};

template<const model_config& config> __global__ void token_embeddings_prompt_eval_time_optimized_test() {};

template<const model_config&> struct kernel_dispatcher_impl;

template<const model_config& config> struct kernel_dispatcher_impl {
	__forceinline__ static cudaError_t impl() {
		std::cout << "WERE HERE THIS IS TI!" << std::endl;
		using model_config_type = model_config_holder<config>;
		token_embeddings_prompt_eval_time_optimized_test<config><<<dim3{}, dim3{}>>>();
		return {};
	}
};