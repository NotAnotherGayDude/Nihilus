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

#include <nihilus-incl/infra/core_bases.hpp>
#include <nihilus-incl/common/kernel_traits.hpp>
#include <nihilus-incl/common/kernel_type_profile_traits.hpp>
#include <nihilus-incl/infra/model_traits.hpp>
#include <nihilus-incl/common/type_traits.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/common/tuple.hpp>

namespace nihilus {

	template<const model_config& config_new, data_strategy_types data_strategy_type, typename output_type> struct data_mixin;

	template<const model_config& config_new, typename output_type_new> struct data_mixin<config_new, data_strategy_types::global, output_type_new> {
		using output_type = output_type_new;
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };

		template<typename output_type_newer = output_type> NIHILUS_HOST output_type_newer* get_data() {
			return static_cast<output_type_newer*>(data);
		}

		NIHILUS_HOST void** get_data_ptr() {
			return &data;
		}

		NIHILUS_HOST void set_data(void* data_new) {
			data = data_new;
		}

	  protected:
		void* data{};
	};

	template<const model_config& config_new, typename output_type_new> struct data_mixin<config_new, data_strategy_types::per_block, output_type_new> {
		using output_type = output_type_new;
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };

		template<typename output_type_newer = output_type> NIHILUS_HOST output_type_newer* get_data(uint64_t index) {
			return static_cast<output_type_newer*>(data[index]);
		}

		NIHILUS_HOST void** get_data_ptr(uint64_t index) {
			return &data[index];
		}

		NIHILUS_HOST void set_data(void* data_new, uint64_t index) {
			data[index] = data_new;
		}

	  protected:
		array<void*, model_traits_type<config_new>::block_count> data{};
	};

	template<const model_config& config_new, core_types core_type> struct sync_base {};

	template<const model_config& config_new, core_types core_type>
		requires(config_new.device_type == device_types::cpu && core_type != core_types::token_embeddings && core_type != core_types::final_norm_and_sampling)
	struct sync_base<config_new, core_type> {
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_new>::block_count> current_chunk_prompt_eval{};
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_new>::block_count> current_chunk_eval{};
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_new>::block_count> latch_prompt_eval{};
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_new>::block_count> latch_eval{};
	};

	template<const model_config& config_new, core_types core_type>
		requires(config_new.device_type == device_types::cpu && (core_type == core_types::token_embeddings || core_type == core_types::final_norm_and_sampling))
	struct sync_base<config_new, core_type> {
		atomic_flag_wrapper<int64_t> current_chunk_prompt_eval{};
		atomic_flag_wrapper<int64_t> current_chunk_eval{};
		atomic_flag_wrapper<int64_t> latch_prompt_eval{};
		atomic_flag_wrapper<int64_t> latch_eval{};
	};

	template<integral_or_enum_types auto index, typename derived_type_new> struct core_elem_base {
		using derived_type = derived_type_new;
		mutable uint64_t total_required_bytes_rt{};
		NIHILUS_HOST constexpr decltype(auto) operator[](tag<index>) & noexcept {
			return *static_cast<derived_type*>(this);
		}
	};

	template<uint64_t required_bytes, uint64_t block_count, data_strategy_types data_strategy_type> constexpr uint64_t get_total_required_bytes{ [] {
		if constexpr (data_strategy_type == data_strategy_types::per_block) {
			return required_bytes * block_count;
		} else {
			return required_bytes;
		}
	}() };

	template<device_types device_type> constexpr allocation_strategy_types allocation_strategy_type{ [] {
		if constexpr (device_type == device_types::gpu) {
			return allocation_strategy_types::alloc;
		} else {
			return allocation_strategy_types::mmap;
		}
	}() };

	template<const model_config& config_new, core_types core_type_new, enum_types auto enum_value_new, composite_kernel_types kernel_type, data_strategy_types data_strategy_type,
		allocation_strategy_types allocation_strategy_type, typename composite_kernel_type_new, typename... input_composite_kernel_types_new>
	struct op_traits : public data_mixin<config_new, data_strategy_type, typename composite_kernel_type_new::output_type>,
					   public composite_kernel_type_new,
					   public core_elem_base<enum_value_new,
						   op_traits<config_new, core_type_new, enum_value_new, kernel_type, data_strategy_type, allocation_strategy_type, composite_kernel_type_new,
							   input_composite_kernel_types_new...>> {
		using output_type = composite_kernel_type_new::output_type;
		using dims_type	  = composite_kernel_type_new::dims_type;
		using enum_type	  = decltype(enum_value_new);
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::get_array())),
			model_traits_type<config_new>::block_count, data_strategy_type> };
		static constexpr auto enum_value{ enum_value_new };
		static constexpr core_types core_type{ core_type_new };
	};

	template<const model_config& config_new> struct core_traits<config_new, core_types::weights>
		: public core_elem_base<core_types::weights, core_traits<config_new, core_types::weights>> {
		static constexpr core_types core_type{ core_types::weights };
		static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
		using kernel_profile_type = kernel_type_profile_traits<config_new.kernel_type_profile>;
		using weight_type		  = typename kernel_profile_type::weight_type;
		using norm_type			  = typename kernel_profile_type::norm_type;

		using attn_q_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::embedding_length, 1, 1>, kernel_types::weights, weight_type>;

		using attn_k_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::n_embd_kv_gqa, 1, 1>, kernel_types::weights, weight_type>;

		using attn_v_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::n_embd_kv_gqa, 1, 1>, kernel_types::weights, weight_type>;

		using attn_output_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::embedding_length, 1, 1>, kernel_types::weights, weight_type>;

		using attn_norm_weight_kernel_traits =
			kernel_traits<config_new, core_trait_dims<model_traits_type<config_new>::embedding_length, 1, 1, 1>, kernel_types::weights, norm_type>;

		using ffn_gate_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1>, kernel_types::weights, weight_type>;

		using ffn_up_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1>, kernel_types::weights, weight_type>;

		using ffn_down_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::embedding_length, 1, 1>, kernel_types::weights, weight_type>;

		using ffn_norm_weight_kernel_traits =
			kernel_traits<config_new, core_trait_dims<model_traits_type<config_new>::embedding_length, 1, 1, 1>, kernel_types::weights, norm_type>;

		using token_embd_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::vocab_size, 1, 1>, kernel_types::weights, weight_type>;

		using rope_freqs_weight_kernel_traits =
			kernel_traits<config_new, core_trait_dims<model_traits_type<config_new>::rope_dimension_count / 2, 1, 1, 1>, kernel_types::weights, norm_type>;

		using output_norm_weight_kernel_traits =
			kernel_traits<config_new, core_trait_dims<model_traits_type<config_new>::embedding_length, 1, 1, 1>, kernel_types::weights, norm_type>;

		using output_weight_kernel_traits = kernel_traits<config_new,
			core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::vocab_size, 1, 1>, kernel_types::weights, weight_type>;

		using attn_q_weight_type	  = op_traits<config_new, core_types::weights, weight_types::attn_q, composite_kernel_types::none, data_strategy_types::per_block,
				 allocation_strategy_type<config_new.device_type>, attn_q_weight_kernel_traits>;
		using attn_k_weight_type	  = op_traits<config_new, core_types::weights, weight_types::attn_k, composite_kernel_types::none, data_strategy_types::per_block,
				 allocation_strategy_type<config_new.device_type>, attn_k_weight_kernel_traits>;
		using attn_v_weight_type	  = op_traits<config_new, core_types::weights, weight_types::attn_v, composite_kernel_types::none, data_strategy_types::per_block,
				 allocation_strategy_type<config_new.device_type>, attn_v_weight_kernel_traits>;
		using attn_output_weight_type = op_traits<config_new, core_types::weights, weight_types::attn_output, composite_kernel_types::none, data_strategy_types::per_block,
			allocation_strategy_type<config_new.device_type>, attn_output_weight_kernel_traits>;
		using attn_norm_weight_type	  = op_traits<config_new, core_types::weights, weight_types::attn_norm, composite_kernel_types::none, data_strategy_types::per_block,
			  allocation_strategy_type<config_new.device_type>, attn_norm_weight_kernel_traits>;
		using ffn_gate_weight_type	  = op_traits<config_new, core_types::weights, weight_types::ffn_gate, composite_kernel_types::none, data_strategy_types::per_block,
			   allocation_strategy_type<config_new.device_type>, ffn_gate_weight_kernel_traits>;
		using ffn_up_weight_type	  = op_traits<config_new, core_types::weights, weight_types::ffn_up, composite_kernel_types::none, data_strategy_types::per_block,
				 allocation_strategy_type<config_new.device_type>, ffn_up_weight_kernel_traits>;
		using ffn_down_weight_type	  = op_traits<config_new, core_types::weights, weight_types::ffn_down, composite_kernel_types::none, data_strategy_types::per_block,
			   allocation_strategy_type<config_new.device_type>, ffn_down_weight_kernel_traits>;
		using ffn_norm_weight_type	  = op_traits<config_new, core_types::weights, weight_types::ffn_norm, composite_kernel_types::none, data_strategy_types::per_block,
			   allocation_strategy_type<config_new.device_type>, ffn_norm_weight_kernel_traits>;
		using token_embd_weight_type  = op_traits<config_new, core_types::weights, weight_types::token_embd, composite_kernel_types::none, data_strategy_types::global,
			 allocation_strategy_type<config_new.device_type>, token_embd_weight_kernel_traits>;
		using rope_freqs_weight_type  = op_traits<config_new, core_types::weights, weight_types::rope_freqs, composite_kernel_types::none, data_strategy_types::global,
			 allocation_strategy_type<config_new.device_type>, rope_freqs_weight_kernel_traits>;
		using output_norm_weight_type = op_traits<config_new, core_types::weights, weight_types::output_norm, composite_kernel_types::none, data_strategy_types::global,
			allocation_strategy_type<config_new.device_type>, output_norm_weight_kernel_traits>;
		using output_weight_type	  = op_traits<config_new, core_types::weights, weight_types::output, composite_kernel_types::none, data_strategy_types::global,
				 allocation_strategy_type<config_new.device_type>, output_weight_kernel_traits>;

		using composite_ops =
			get_core_base_t<config_new, attn_q_weight_type, attn_k_weight_type, attn_v_weight_type, attn_output_weight_type, attn_norm_weight_type, ffn_gate_weight_type,
				ffn_up_weight_type, ffn_down_weight_type, ffn_norm_weight_type, token_embd_weight_type, rope_freqs_weight_type, output_norm_weight_type, output_weight_type>;
		composite_ops values{};

		static constexpr uint64_t total_required_bytes{ attn_q_weight_type::total_required_bytes + attn_k_weight_type::total_required_bytes +
			attn_v_weight_type::total_required_bytes + attn_output_weight_type::total_required_bytes + attn_norm_weight_type::total_required_bytes +
			ffn_gate_weight_type::total_required_bytes + ffn_up_weight_type::total_required_bytes + ffn_down_weight_type::total_required_bytes +
			ffn_norm_weight_type::total_required_bytes + token_embd_weight_type::total_required_bytes + rope_freqs_weight_type::total_required_bytes +
			output_norm_weight_type::total_required_bytes + output_weight_type::total_required_bytes };

		static constexpr bool has_total_required_bytes{ config_new.device_type == device_types::gpu };
	};

	template<const model_config& config_new> struct core_traits<config_new, core_types::global_inputs>
		: public core_elem_base<core_types::global_inputs, core_traits<config_new, core_types::global_inputs>> {
		static constexpr core_types core_type{ core_types::global_inputs };
		static constexpr uint64_t depth{ 0 };
		using enum_type			  = global_input_types;
		using mtt				  = model_traits_type<config_new>;
		using kernel_profile_type = kernel_type_profile_traits<config_new.kernel_type_profile>;
		using weight_type		  = typename kernel_profile_type::weight_type;
		using norm_type			  = typename kernel_profile_type::norm_type;

		using inp_tokens_kernel_traits =
			kernel_traits<config_new, core_trait_dims<config_new.default_max_sequence_length, 1, 1, 1, 0>, kernel_types::weights, typename kernel_profile_type::input_token_type>;
		using inp_pos_kernel_traits =
			kernel_traits<config_new, core_trait_dims<config_new.default_max_sequence_length, 1, 1, 1, 0>, kernel_types::weights, typename kernel_profile_type::position_type>;
		using cache_k_kernel_traits =
			kernel_traits<config_new, core_trait_dims<mtt::block_count, config_new.default_max_sequence_length, mtt::attention_head_count_kv, mtt::rope_dimension_count, 0>,
				kernel_types::weights, typename kernel_profile_type::kv_cache_type>;
		using cache_v_kernel_traits =
			kernel_traits<config_new, core_trait_dims<mtt::block_count, config_new.default_max_sequence_length, mtt::attention_head_count_kv, mtt::rope_dimension_count, 0>,
				kernel_types::weights, typename kernel_profile_type::kv_cache_type>;
		using kq_mask_kernel_traits =
			kernel_traits<config_new, core_trait_dims<mtt::block_count, mtt::block_count, 1, 1>, kernel_types::weights, typename kernel_profile_type::mask_type>;
		using inp_out_ids_kernel_traits =
			kernel_traits<config_new, core_trait_dims<config_new.default_max_sequence_length, 1, 1, 1, 0>, kernel_types::weights, typename kernel_profile_type::output_token_type>;

		using temperature_kernel_traits		   = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1>, kernel_types::weights, float>;
		using top_k_kernel_traits			   = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1>, kernel_types::weights, int32_t>;
		using top_p_kernel_traits			   = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1>, kernel_types::weights, float>;
		using repetition_penalty_kernel_traits = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1>, kernel_types::weights, float>;
		using presence_penalty_kernel_traits   = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1>, kernel_types::weights, float>;
		using frequency_penalty_kernel_traits  = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1>, kernel_types::weights, float>;
		using rep_window_kernel_traits		   = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1>, kernel_types::weights, int32_t>;
		using token_history_kernel_traits	   = kernel_traits<config_new, core_trait_dims<config_new.default_max_sequence_length, 1, 1, 1, 0>, kernel_types::weights, int32_t>;
		using rng_state_kernel_traits		   = kernel_traits<config_new, core_trait_dims<2, 1, 1, 1>, kernel_types::weights, uint64_t>;

		using logits_bias_kernel_traits		   = kernel_traits<config_new, core_trait_dims<mtt::vocab_size, 1, 1, 1>, kernel_types::weights, float>;
		using allowed_vocab_mask_kernel_traits = kernel_traits<config_new, core_trait_dims<mtt::vocab_size, 1, 1, 1>, kernel_types::weights, uint8_t>;

		using cache_k_type	   = op_traits<config_new, core_types::global_inputs, global_input_types::cache_k, composite_kernel_types::none, data_strategy_types::global,
				allocation_strategy_types::alloc, cache_k_kernel_traits>;
		using cache_v_type	   = op_traits<config_new, core_types::global_inputs, global_input_types::cache_v, composite_kernel_types::none, data_strategy_types::global,
				allocation_strategy_types::alloc, cache_v_kernel_traits>;
		using inp_tokens_type  = op_traits<config_new, core_types::global_inputs, global_input_types::inp_tokens, composite_kernel_types::none, data_strategy_types::global,
			 allocation_strategy_types::alloc, inp_tokens_kernel_traits>;
		using inp_pos_type	   = op_traits<config_new, core_types::global_inputs, global_input_types::inp_pos, composite_kernel_types::none, data_strategy_types::global,
				allocation_strategy_types::alloc, inp_pos_kernel_traits>;
		using kq_mask_type	   = op_traits<config_new, core_types::global_inputs, global_input_types::kq_mask, composite_kernel_types::none, data_strategy_types::global,
				allocation_strategy_types::alloc, kq_mask_kernel_traits>;
		using inp_out_ids_type = op_traits<config_new, core_types::global_inputs, global_input_types::inp_out_ids, composite_kernel_types::none, data_strategy_types::global,
			allocation_strategy_types::alloc, inp_out_ids_kernel_traits>;

		using temperature_type		  = op_traits<config_new, core_types::global_inputs, global_input_types::temperature, composite_kernel_types::none, data_strategy_types::global,
				   allocation_strategy_types::alloc, temperature_kernel_traits>;
		using top_k_type			  = op_traits<config_new, core_types::global_inputs, global_input_types::top_k, composite_kernel_types::none, data_strategy_types::global,
						 allocation_strategy_types::alloc, top_k_kernel_traits>;
		using top_p_type			  = op_traits<config_new, core_types::global_inputs, global_input_types::top_p, composite_kernel_types::none, data_strategy_types::global,
						 allocation_strategy_types::alloc, top_p_kernel_traits>;
		using repetition_penalty_type = op_traits<config_new, core_types::global_inputs, global_input_types::repetition_penalty, composite_kernel_types::none,
			data_strategy_types::global, allocation_strategy_types::alloc, repetition_penalty_kernel_traits>;
		using presence_penalty_type	  = op_traits<config_new, core_types::global_inputs, global_input_types::presence_penalty, composite_kernel_types::none,
			  data_strategy_types::global, allocation_strategy_types::alloc, presence_penalty_kernel_traits>;
		using frequency_penalty_type  = op_traits<config_new, core_types::global_inputs, global_input_types::frequency_penalty, composite_kernel_types::none,
			 data_strategy_types::global, allocation_strategy_types::alloc, frequency_penalty_kernel_traits>;
		using rep_window_type		  = op_traits<config_new, core_types::global_inputs, global_input_types::rep_window, composite_kernel_types::none, data_strategy_types::global,
					allocation_strategy_types::alloc, rep_window_kernel_traits>;
		using token_history_type = op_traits<config_new, core_types::global_inputs, global_input_types::token_history, composite_kernel_types::none, data_strategy_types::global,
			allocation_strategy_types::alloc, token_history_kernel_traits>;
		using rng_state_type	 = op_traits<config_new, core_types::global_inputs, global_input_types::rng_state, composite_kernel_types::none, data_strategy_types::global,
				allocation_strategy_types::alloc, rng_state_kernel_traits>;

		using logits_bias_type		  = op_traits<config_new, core_types::global_inputs, global_input_types::logits_bias, composite_kernel_types::none, data_strategy_types::global,
				   allocation_strategy_types::alloc, logits_bias_kernel_traits>;
		using allowed_vocab_mask_type = op_traits<config_new, core_types::global_inputs, global_input_types::allowed_vocab_mask, composite_kernel_types::none,
			data_strategy_types::global, allocation_strategy_types::alloc, allowed_vocab_mask_kernel_traits>;

		using composite_ops = get_core_base_t<config_new, inp_tokens_type, inp_pos_type, cache_k_type, cache_v_type, kq_mask_type, inp_out_ids_type, temperature_type, top_k_type,
			top_p_type, repetition_penalty_type, presence_penalty_type, frequency_penalty_type, rep_window_type, token_history_type, rng_state_type, logits_bias_type,
			allowed_vocab_mask_type>;

		composite_ops values{};

		static constexpr uint64_t total_required_bytes{ inp_tokens_type::total_required_bytes + inp_pos_type::total_required_bytes + cache_k_type::total_required_bytes +
			cache_v_type::total_required_bytes + kq_mask_type::total_required_bytes + inp_out_ids_type::total_required_bytes + temperature_type::total_required_bytes +
			top_k_type::total_required_bytes + top_p_type::total_required_bytes + repetition_penalty_type::total_required_bytes + presence_penalty_type::total_required_bytes +
			frequency_penalty_type::total_required_bytes + rep_window_type::total_required_bytes + token_history_type::total_required_bytes +
			rng_state_type::total_required_bytes };
		static constexpr bool has_total_required_bytes{ true };
	};

	template<const model_config& config_new> struct core_traits<config_new, core_types::token_embeddings>
		: public core_elem_base<core_types::token_embeddings, core_traits<config_new, core_types::token_embeddings>>, public sync_base<config_new, core_types::token_embeddings> {
		static constexpr core_types core_type{ core_types::token_embeddings };
		static constexpr const model_config& config{ config_new };
		static constexpr uint64_t depth{ core_traits<config_new, static_cast<core_types>(static_cast<uint64_t>(core_types::token_embeddings) - 1)>::depth + 1 };
		using enum_type			  = token_embeddings_types;
		using mtt				  = model_traits_type<config_new>;
		using kernel_profile_type = kernel_type_profile_traits<config_new.kernel_type_profile>;
		using weight_type		  = typename kernel_profile_type::weight_type;
		using norm_type			  = typename kernel_profile_type::norm_type;
		using compute_t			  = typename kernel_profile_type::compute_type;
		using kv_store_t		  = typename kernel_profile_type::kv_cache_type;
		using index_type		  = typename kernel_profile_type::index_type;
		using output_type		  = typename kernel_type_profile_traits<config_new.kernel_type_profile>::compute_type;

		using input_01_type = typename core_traits<config_new, core_types::weights>::token_embd_weight_type;
		using input_02_type = typename core_traits<config_new, core_types::global_inputs>::inp_tokens_type;

		using input_embedding_kernel_traits = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::get_rows, input_01_type, input_02_type>, kernel_types::get_rows,
			output_type, input_01_type, input_02_type>;

		using token_embeddings_type = op_traits<config_new, core_types::token_embeddings, token_embeddings_types::get_rows, composite_kernel_types::get_rows,
			data_strategy_types::global, allocation_strategy_types::alloc, input_embedding_kernel_traits>;

		using composite_ops = get_core_base_t<config_new, token_embeddings_type>;

		composite_ops values{};

		struct kernel_data_ptrs {
		  protected:
			NIHILUS_ALIGN(16) const void* weight_data;
			NIHILUS_ALIGN(16) const void* token_data;
			NIHILUS_ALIGN(16) void* output_data;

		  public:

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_weight_data() {
				return static_cast<const value_type*>(weight_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_token_data() {
				return static_cast<const value_type*>(token_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_output_data() {
				return static_cast<value_type*>(output_data);
			}

			NIHILUS_HOST void set_ptrs(void* output_data_new, const void* weight_data_new, const void* token_ids_new) {
				output_data = output_data_new;
				weight_data = weight_data_new;
				token_data	= token_ids_new;
			}
		};

		kernel_data_ptrs data_ptrs{};

		static constexpr uint64_t total_required_bytes{ token_embeddings_type::total_required_bytes };
		static constexpr bool has_total_required_bytes{ true };
	};

	template<const model_config& config_new> struct core_traits<config_new, core_types::mega_qkv_prep_and_cache_publish>
		: public core_elem_base<core_types::mega_qkv_prep_and_cache_publish, core_traits<config_new, core_types::mega_qkv_prep_and_cache_publish>>,
		  public sync_base<config_new, core_types::mega_qkv_prep_and_cache_publish> {
		static constexpr core_types core_type{ core_types::mega_qkv_prep_and_cache_publish };
		static constexpr const model_config& config{ config_new };
		static constexpr uint64_t depth{ core_traits<config_new, static_cast<core_types>(static_cast<uint64_t>(core_types::mega_qkv_prep_and_cache_publish) - 1)>::depth + 1 };
		using enum_type			  = mega_qkv_prep_and_cache_publish_types;
		using mtt				  = model_traits_type<config_new>;
		using kernel_profile_type = kernel_type_profile_traits<config_new.kernel_type_profile>;
		using weight_type		  = typename kernel_profile_type::weight_type;
		using norm_type			  = typename kernel_profile_type::norm_type;
		using compute_t			  = typename kernel_profile_type::compute_type;
		using kv_store_t		  = typename kernel_profile_type::kv_cache_type;

		using inp_embd_type	   = typename core_traits<config_new, core_types::token_embeddings>::token_embeddings_type;
		using attn_norm_w_type = typename core_traits<config_new, core_types::weights>::attn_norm_weight_type;
		using attn_q_w_type	   = typename core_traits<config_new, core_types::weights>::attn_q_weight_type;
		using attn_k_w_type	   = typename core_traits<config_new, core_types::weights>::attn_k_weight_type;
		using attn_v_w_type	   = typename core_traits<config_new, core_types::weights>::attn_v_weight_type;
		using inp_pos_type	   = typename core_traits<config_new, core_types::global_inputs>::inp_pos_type;
		using rope_freqs_type  = typename core_traits<config_new, core_types::weights>::rope_freqs_weight_type;
		using cache_k_type	   = typename core_traits<config_new, core_types::global_inputs>::cache_k_type;
		using cache_v_type	   = typename core_traits<config_new, core_types::global_inputs>::cache_v_type;

		static_assert(mtt::n_embd_kv_gqa == mtt::rope_dimension_count * mtt::attention_head_count_kv, "n_embd_kv_gqa must equal rope_dim * kv_heads for GQA.");

		using rms_norm_trait = kernel_traits<config_new, get_new_dims_new_1_t<kernel_types::rms_norm, inp_embd_type>, kernel_types::rms_norm, compute_t, inp_embd_type>;

		using mul_trait =
			kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul, rms_norm_trait, attn_norm_w_type>, kernel_types::mul, compute_t, rms_norm_trait, attn_norm_w_type>;

		using q_mul_mat_trait =
			kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, attn_q_w_type, mul_trait>, kernel_types::mul_mat, compute_t, attn_q_w_type, mul_trait>;

		using q_reshape_trait = kernel_traits<config_new, core_trait_dims<mtt::rope_dimension_count, mtt::attention_head_count, config_new.default_max_sequence_length, 1, 2>,
			kernel_types::reshape, compute_t, q_mul_mat_trait>;

		using q_rope_trait = kernel_traits<config_new, get_new_dims_new_3_t<kernel_types::rope, q_reshape_trait, inp_pos_type, rope_freqs_type>, kernel_types::rope, compute_t,
			q_reshape_trait, inp_pos_type, rope_freqs_type>;

		using k_mul_mat_trait =
			kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, attn_k_w_type, mul_trait>, kernel_types::mul_mat, compute_t, attn_k_w_type, mul_trait>;

		using k_reshape_trait = kernel_traits<config_new, core_trait_dims<mtt::rope_dimension_count, mtt::attention_head_count_kv, config_new.default_max_sequence_length, 1, 2>,
			kernel_types::reshape, compute_t, k_mul_mat_trait>;

		using k_rope_trait = kernel_traits<config_new, get_new_dims_new_3_t<kernel_types::rope, k_reshape_trait, inp_pos_type, rope_freqs_type>, kernel_types::rope, compute_t,
			k_reshape_trait, inp_pos_type, rope_freqs_type>;

		using v_mul_mat_trait =
			kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, attn_v_w_type, mul_trait>, kernel_types::mul_mat, compute_t, attn_v_w_type, mul_trait>;

		using v_transpose_trait =
			kernel_traits<config_new, core_trait_dims<config_new.default_max_sequence_length, mtt::n_embd_kv_gqa, 1, 1, 0>, kernel_types::transpose, compute_t, v_mul_mat_trait>;

		using k_cache_window_view_trait = kernel_traits<config_new,
			core_trait_dims<mtt::rope_dimension_count, config_new.default_max_sequence_length, mtt::attention_head_count_kv, 1, 1>, kernel_types::view, kv_store_t, cache_k_type>;

		using v_cache_window_view_trait = kernel_traits<config_new,
			core_trait_dims<config_new.default_max_sequence_length, mtt::rope_dimension_count, mtt::attention_head_count_kv, 1, 0>, kernel_types::view, kv_store_t, cache_v_type>;

		using k_cache_store_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::copy, k_rope_trait, k_cache_window_view_trait>, kernel_types::copy, kv_store_t,
			k_rope_trait, k_cache_window_view_trait>;

		using v_cache_store_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::copy, v_transpose_trait, v_cache_window_view_trait>, kernel_types::copy,
			kv_store_t, v_transpose_trait, v_cache_window_view_trait>;

		using mega_qkv_composite_traits = composite_kernel_traits<config_new, composite_kernel_types::mega_qkv_prep_and_cache, compute_t, rms_norm_trait, mul_trait,
			k_mul_mat_trait, k_reshape_trait, k_rope_trait, k_cache_window_view_trait, k_cache_store_trait, v_mul_mat_trait, v_transpose_trait, v_cache_window_view_trait,
			v_cache_store_trait, q_mul_mat_trait, q_reshape_trait, q_rope_trait>;

		using q_out_type = op_traits<config_new, core_types::mega_qkv_prep_and_cache_publish, mega_qkv_prep_and_cache_publish_types::q_out,
			composite_kernel_types::mega_qkv_prep_and_cache, data_strategy_types::global, allocation_strategy_types::alloc, mega_qkv_composite_traits, rms_norm_trait, mul_trait,
			k_mul_mat_trait, k_reshape_trait, k_rope_trait, k_cache_window_view_trait, k_cache_store_trait, v_mul_mat_trait, v_transpose_trait, v_cache_window_view_trait,
			v_cache_store_trait, q_mul_mat_trait, q_reshape_trait, q_rope_trait>;

		using composite_ops = get_core_base_t<config_new, q_out_type>;

		composite_ops values{};

		struct kernel_data_ptrs {
		  protected:
			NIHILUS_ALIGN(16) const void* inp_embd_data;
			NIHILUS_ALIGN(16) const void* attn_norm_w_data;
			NIHILUS_ALIGN(16) const void* attn_q_w_data;
			NIHILUS_ALIGN(16) const void* attn_k_w_data;
			NIHILUS_ALIGN(16) const void* attn_v_w_data;
			NIHILUS_ALIGN(16) const void* inp_pos_data;
			NIHILUS_ALIGN(16) const void* rope_freqs_data;
			NIHILUS_ALIGN(16) void* cache_k_data;
			NIHILUS_ALIGN(16) void* cache_v_data;
			NIHILUS_ALIGN(16) void* q_output_data;

		  public:
			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_inp_embd_data() {
				return static_cast<const value_type*>(inp_embd_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_attn_norm_w_data() {
				return static_cast<const value_type*>(attn_norm_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_attn_q_w_data() {
				return static_cast<const value_type*>(attn_q_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_attn_k_w_data() {
				return static_cast<const value_type*>(attn_k_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_attn_v_w_data() {
				return static_cast<const value_type*>(attn_v_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_inp_pos_data() {
				return static_cast<const value_type*>(inp_pos_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_rope_freqs_data() {
				return static_cast<const value_type*>(rope_freqs_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_cache_k_data() {
				return static_cast<value_type*>(cache_k_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_cache_v_data() {
				return static_cast<value_type*>(cache_v_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_q_output_data() {
				return static_cast<value_type*>(q_output_data);
			}

			NIHILUS_HOST void set_ptrs(void* q_output_data_new, const void* inp_embd_data_new, const void* attn_norm_w_data_new, const void* attn_q_w_data_new,
				const void* attn_k_w_data_new, const void* attn_v_w_data_new, const void* inp_pos_data_new, const void* rope_freqs_data_new, void* cache_k_data_new,
				void* cache_v_data_new) {
				q_output_data	 = q_output_data_new;
				inp_embd_data	 = inp_embd_data_new;
				attn_norm_w_data = attn_norm_w_data_new;
				attn_q_w_data	 = attn_q_w_data_new;
				attn_k_w_data	 = attn_k_w_data_new;
				attn_v_w_data	 = attn_v_w_data_new;
				inp_pos_data	 = inp_pos_data_new;
				rope_freqs_data	 = rope_freqs_data_new;
				cache_k_data	 = cache_k_data_new;
				cache_v_data	 = cache_v_data_new;
			}
		};

		kernel_data_ptrs data_ptrs{};

		static constexpr uint64_t total_required_bytes{ q_out_type::total_required_bytes };
		static constexpr bool has_total_required_bytes{ true };
	};

	template<const model_config& config_new> struct core_traits<config_new, core_types::mega_attention_apply>
		: public core_elem_base<core_types::mega_attention_apply, core_traits<config_new, core_types::mega_attention_apply>>,
		  public sync_base<config_new, core_types::mega_attention_apply> {
		static constexpr core_types core_type{ core_types::mega_attention_apply };
		static constexpr const model_config& config{ config_new };
		static constexpr uint64_t depth{ core_traits<config_new, static_cast<core_types>(static_cast<uint64_t>(core_types::mega_attention_apply) - 1)>::depth + 1 };
		using enum_type			  = mega_attention_apply_types;
		using mtt				  = model_traits_type<config_new>;
		using kernel_profile_type = kernel_type_profile_traits<config_new.kernel_type_profile>;
		using weight_type		  = typename kernel_profile_type::weight_type;
		using norm_type			  = typename kernel_profile_type::norm_type;
		using compute_t			  = typename kernel_profile_type::compute_type;
		using kv_store_t		  = typename kernel_profile_type::kv_cache_type;

		using q_rope_input_dims	 = core_trait_dims<mtt::rope_dimension_count, config_new.default_max_sequence_length, mtt::attention_head_count, 1, 1>;
		using cache_k_type		 = typename core_traits<config_new, core_types::global_inputs>::cache_k_type;
		using cache_v_type		 = typename core_traits<config_new, core_types::global_inputs>::cache_v_type;
		using kq_mask_type		 = typename core_traits<config_new, core_types::global_inputs>::kq_mask_type;
		using attn_output_w_type = typename core_traits<config_new, core_types::weights>::attn_output_weight_type;
		using inp_embd_type		 = typename core_traits<config_new, core_types::token_embeddings>::token_embeddings_type;

		using k_cache_read_view_trait = kernel_traits<config_new,
			core_trait_dims<mtt::rope_dimension_count, config_new.default_max_sequence_length, mtt::attention_head_count_kv, 1, 1>, kernel_types::view, kv_store_t, cache_k_type>;
		using v_cache_read_view_trait = kernel_traits<config_new,
			core_trait_dims<config_new.default_max_sequence_length, mtt::rope_dimension_count, mtt::attention_head_count_kv, 1, 0>, kernel_types::view, kv_store_t, cache_v_type>;

		using q_input_trait = kernel_traits<config_new, q_rope_input_dims, kernel_types::weights, compute_t>;

		using kq_mul_mat_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, k_cache_read_view_trait, q_input_trait>, kernel_types::mul_mat, compute_t,
			k_cache_read_view_trait, q_input_trait>;

		using softmax_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::softmax, kq_mul_mat_trait, kq_mask_type>, kernel_types::softmax, compute_t,
			kq_mul_mat_trait, kq_mask_type>;

		using kqv_mul_mat_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, v_cache_read_view_trait, softmax_trait>, kernel_types::mul_mat, compute_t,
			v_cache_read_view_trait, softmax_trait>;

		using merge_permute_trait = kernel_traits<config_new, core_trait_dims<mtt::rope_dimension_count, mtt::attention_head_count, config_new.default_max_sequence_length, 1, 2>,
			kernel_types::permute, compute_t, kqv_mul_mat_trait>;

		using cont_trait =
			kernel_traits<config_new, core_trait_dims<mtt::embedding_length, config_new.default_max_sequence_length, 1, 1, 1>, kernel_types::cont, compute_t, merge_permute_trait>;

		using attn_out_mul_mat_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, attn_output_w_type, cont_trait>, kernel_types::mul_mat, compute_t,
			attn_output_w_type, cont_trait>;

		using residual_add_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::add, attn_out_mul_mat_trait, inp_embd_type>, kernel_types::add, compute_t,
			attn_out_mul_mat_trait, inp_embd_type>;

		using mega_attention_composite_traits = composite_kernel_traits<config_new, composite_kernel_types::mega_attention_apply, compute_t, k_cache_read_view_trait,
			v_cache_read_view_trait, kq_mul_mat_trait, softmax_trait, kqv_mul_mat_trait, merge_permute_trait, cont_trait, attn_out_mul_mat_trait, residual_add_trait>;
		using ffn_inp_type = op_traits<config_new, core_types::mega_attention_apply, mega_attention_apply_types::ffn_inp, composite_kernel_types::mega_attention_apply,
			data_strategy_types::global, allocation_strategy_types::alloc, mega_attention_composite_traits>;

		using composite_ops = get_core_base_t<config_new, ffn_inp_type>;
		composite_ops values{};

		struct kernel_data_ptrs {
		  protected:
			NIHILUS_ALIGN(16) const void* q_input_data;
			NIHILUS_ALIGN(16) const void* cache_k_data;
			NIHILUS_ALIGN(16) const void* cache_v_data;
			NIHILUS_ALIGN(16) const void* kq_mask_data;
			NIHILUS_ALIGN(16) const void* attn_output_w_data;
			NIHILUS_ALIGN(16) const void* inp_embd_data;
			NIHILUS_ALIGN(16) void* ffn_inp_data;

		  public:
			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_q_input_data() {
				return static_cast<const value_type*>(q_input_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_cache_k_data() {
				return static_cast<const value_type*>(cache_k_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_cache_v_data() {
				return static_cast<const value_type*>(cache_v_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_kq_mask_data() {
				return static_cast<const value_type*>(kq_mask_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_attn_output_w_data() {
				return static_cast<const value_type*>(attn_output_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_inp_embd_data() {
				return static_cast<const value_type*>(inp_embd_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_ffn_inp_data() {
				return static_cast<value_type*>(ffn_inp_data);
			}

			NIHILUS_HOST void set_ptrs(void* ffn_inp_data_new, const void* q_input_data_new, const void* cache_k_data_new, const void* cache_v_data_new,
				const void* kq_mask_data_new, const void* attn_output_w_data_new, const void* inp_embd_data_new) {
				ffn_inp_data	   = ffn_inp_data_new;
				q_input_data	   = q_input_data_new;
				cache_k_data	   = cache_k_data_new;
				cache_v_data	   = cache_v_data_new;
				kq_mask_data	   = kq_mask_data_new;
				attn_output_w_data = attn_output_w_data_new;
				inp_embd_data	   = inp_embd_data_new;
			}
		};

		kernel_data_ptrs data_ptrs{};

		static constexpr uint64_t total_required_bytes{ ffn_inp_type::total_required_bytes };
		static constexpr bool has_total_required_bytes{ true };
	};

	template<const model_config& config_new> struct core_traits<config_new, core_types::mega_ffn>
		: public core_elem_base<core_types::mega_ffn, core_traits<config_new, core_types::mega_ffn>>, public sync_base<config_new, core_types::mega_ffn> {
		static constexpr core_types core_type{ core_types::mega_ffn };
		static constexpr const model_config& config{ config_new };
		static constexpr uint64_t depth{ core_traits<config_new, static_cast<core_types>(static_cast<uint64_t>(core_types::mega_ffn) - 1)>::depth + 1 };
		using enum_type			  = mega_ffn_types;
		using mtt				  = model_traits_type<config_new>;
		using kernel_profile_type = kernel_type_profile_traits<config_new.kernel_type_profile>;
		using compute_t			  = typename kernel_profile_type::compute_type;

		using ffn_inp_input_dims = core_trait_dims<mtt::embedding_length, config_new.default_max_sequence_length, 1, 1, 0>;
		using ffn_norm_w_type	 = typename core_traits<config_new, core_types::weights>::ffn_norm_weight_type;
		using ffn_gate_w_type	 = typename core_traits<config_new, core_types::weights>::ffn_gate_weight_type;
		using ffn_up_w_type		 = typename core_traits<config_new, core_types::weights>::ffn_up_weight_type;
		using ffn_down_w_type	 = typename core_traits<config_new, core_types::weights>::ffn_down_weight_type;

		using ffn_inp_trait = kernel_traits<config_new, ffn_inp_input_dims, kernel_types::weights, compute_t>;

		using ffn_rms_norm_trait = kernel_traits<config_new, get_new_dims_new_1_t<kernel_types::rms_norm, ffn_inp_trait>, kernel_types::rms_norm, compute_t, ffn_inp_trait>;

		using ffn_norm_mul_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul, ffn_rms_norm_trait, ffn_norm_w_type>, kernel_types::mul, compute_t,
			ffn_rms_norm_trait, ffn_norm_w_type>;

		using ffn_gate_mul_mat_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, ffn_gate_w_type, ffn_norm_mul_trait>, kernel_types::mul_mat, compute_t,
			ffn_gate_w_type, ffn_norm_mul_trait>;

		using ffn_up_mul_mat_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, ffn_up_w_type, ffn_norm_mul_trait>, kernel_types::mul_mat, compute_t,
			ffn_up_w_type, ffn_norm_mul_trait>;

		using ffn_silu_trait = kernel_traits<config_new, get_new_dims_new_1_t<kernel_types::silu, ffn_gate_mul_mat_trait>, kernel_types::silu, compute_t, ffn_gate_mul_mat_trait>;

		using ffn_gate_par_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul, ffn_silu_trait, ffn_up_mul_mat_trait>, kernel_types::mul, compute_t,
			ffn_silu_trait, ffn_up_mul_mat_trait>;

		using ffn_down_mul_mat_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, ffn_down_w_type, ffn_gate_par_trait>, kernel_types::mul_mat, compute_t,
			ffn_down_w_type, ffn_gate_par_trait>;

		using ffn_residual_add_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::add, ffn_down_mul_mat_trait, ffn_inp_trait>, kernel_types::add, compute_t,
			ffn_down_mul_mat_trait, ffn_inp_trait>;

		using mega_ffn_composite_traits = composite_kernel_traits<config_new, composite_kernel_types::mega_ffn, compute_t, ffn_rms_norm_trait, ffn_norm_mul_trait,
			ffn_gate_mul_mat_trait, ffn_up_mul_mat_trait, ffn_silu_trait, ffn_gate_par_trait, ffn_down_mul_mat_trait, ffn_residual_add_trait>;

		using l_out_type = op_traits<config_new, core_types::mega_ffn, mega_ffn_types::l_out, composite_kernel_types::mega_ffn, data_strategy_types::global,
			allocation_strategy_types::alloc, mega_ffn_composite_traits>;

		using composite_ops = get_core_base_t<config_new, l_out_type>;
		composite_ops values{};

		struct kernel_data_ptrs {
		  protected:
			NIHILUS_ALIGN(16) const void* ffn_inp_data;
			NIHILUS_ALIGN(16) const void* ffn_norm_w_data;
			NIHILUS_ALIGN(16) const void* ffn_gate_w_data;
			NIHILUS_ALIGN(16) const void* ffn_up_w_data;
			NIHILUS_ALIGN(16) const void* ffn_down_w_data;
			NIHILUS_ALIGN(16) void* l_out_data;

		  public:
			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_ffn_inp_data() {
				return static_cast<const value_type*>(ffn_inp_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_ffn_norm_w_data() {
				return static_cast<const value_type*>(ffn_norm_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_ffn_gate_w_data() {
				return static_cast<const value_type*>(ffn_gate_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_ffn_up_w_data() {
				return static_cast<const value_type*>(ffn_up_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_ffn_down_w_data() {
				return static_cast<const value_type*>(ffn_down_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_l_out_data() {
				return static_cast<value_type*>(l_out_data);
			}

			NIHILUS_HOST void set_ptrs(void* l_out_data_new, const void* ffn_inp_data_new, const void* ffn_norm_w_data_new, const void* ffn_gate_w_data_new,
				const void* ffn_up_w_data_new, const void* ffn_down_w_data_new) {
				l_out_data		= l_out_data_new;
				ffn_inp_data	= ffn_inp_data_new;
				ffn_norm_w_data = ffn_norm_w_data_new;
				ffn_gate_w_data = ffn_gate_w_data_new;
				ffn_up_w_data	= ffn_up_w_data_new;
				ffn_down_w_data = ffn_down_w_data_new;
			}
		};

		kernel_data_ptrs data_ptrs{};

		static constexpr uint64_t total_required_bytes{ l_out_type::total_required_bytes };
		static constexpr bool has_total_required_bytes{ true };
	};

	template<const model_config& config_new> struct core_traits<config_new, core_types::final_norm_and_sampling>
		: public core_elem_base<core_types::final_norm_and_sampling, core_traits<config_new, core_types::final_norm_and_sampling>>,
		  public sync_base<config_new, core_types::final_norm_and_sampling> {
		static constexpr core_types core_type{ core_types::final_norm_and_sampling };
		static constexpr const model_config& config{ config_new };
		static constexpr uint64_t depth{ core_traits<config_new, static_cast<core_types>(static_cast<uint64_t>(core_types::final_norm_and_sampling) - 1)>::depth + 1 };
		using enum_type			  = final_norm_and_sampling_types;
		using mtt				  = model_traits_type<config_new>;
		using kernel_profile_type = kernel_type_profile_traits<config_new.kernel_type_profile>;
		using compute_t			  = typename kernel_profile_type::compute_type;

		using l_out_input_dims	 = core_trait_dims<mtt::embedding_length, config_new.default_max_sequence_length, 1, 1, 0>;
		using output_norm_w_type = typename core_traits<config_new, core_types::weights>::output_norm_weight_type;
		using output_w_type		 = typename core_traits<config_new, core_types::weights>::output_weight_type;

		using temperature_type		  = typename core_traits<config_new, core_types::global_inputs>::temperature_type;
		using top_k_type			  = typename core_traits<config_new, core_types::global_inputs>::top_k_type;
		using top_p_type			  = typename core_traits<config_new, core_types::global_inputs>::top_p_type;
		using repetition_penalty_type = typename core_traits<config_new, core_types::global_inputs>::repetition_penalty_type;
		using presence_penalty_type	  = typename core_traits<config_new, core_types::global_inputs>::presence_penalty_type;
		using frequency_penalty_type  = typename core_traits<config_new, core_types::global_inputs>::frequency_penalty_type;
		using rep_window_type		  = typename core_traits<config_new, core_types::global_inputs>::rep_window_type;
		using token_history_type	  = typename core_traits<config_new, core_types::global_inputs>::token_history_type;
		using rng_state_type		  = typename core_traits<config_new, core_types::global_inputs>::rng_state_type;
		using logits_bias_type		  = typename core_traits<config_new, core_types::global_inputs>::logits_bias_type;
		using allowed_vocab_mask_type = typename core_traits<config_new, core_types::global_inputs>::allowed_vocab_mask_type;

		using l_out_trait = kernel_traits<config_new, l_out_input_dims, kernel_types::weights, compute_t>;

		using last_token_view_trait = kernel_traits<config_new, core_trait_dims<mtt::embedding_length, 1, 1, 1, 0>, kernel_types::view, compute_t, l_out_trait>;

		using final_rms_norm_trait =
			kernel_traits<config_new, get_new_dims_new_1_t<kernel_types::rms_norm, last_token_view_trait>, kernel_types::rms_norm, compute_t, last_token_view_trait>;

		using final_norm_mul_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul, final_rms_norm_trait, output_norm_w_type>, kernel_types::mul, compute_t,
			final_rms_norm_trait, output_norm_w_type>;

		using logits_mul_mat_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::mul_mat, output_w_type, final_norm_mul_trait>, kernel_types::mul_mat, compute_t,
			output_w_type, final_norm_mul_trait>;

		using logits_bias_add_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::add, logits_mul_mat_trait, logits_bias_type>, kernel_types::add, compute_t,
			logits_mul_mat_trait, logits_bias_type>;

		using rep_penalty_trait =
			kernel_traits<config_new, get_new_dims_new_3_t<kernel_types::repetition_penalty, logits_bias_add_trait, token_history_type, repetition_penalty_type>,
				kernel_types::repetition_penalty, compute_t, logits_bias_add_trait, token_history_type, repetition_penalty_type>;

		using presence_penalty_trait = kernel_traits<config_new, get_new_dims_new_3_t<kernel_types::presence_penalty, rep_penalty_trait, token_history_type, presence_penalty_type>,
			kernel_types::presence_penalty, compute_t, rep_penalty_trait, token_history_type, presence_penalty_type>;

		using frequency_penalty_trait =
			kernel_traits<config_new, get_new_dims_new_3_t<kernel_types::frequency_penalty, presence_penalty_trait, token_history_type, frequency_penalty_type>,
				kernel_types::frequency_penalty, compute_t, presence_penalty_trait, token_history_type, frequency_penalty_type>;

		using vocab_mask_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::vocab_mask, frequency_penalty_trait, allowed_vocab_mask_type>,
			kernel_types::vocab_mask, compute_t, frequency_penalty_trait, allowed_vocab_mask_type>;

		using temperature_scale_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::temperature_scale, vocab_mask_trait, temperature_type>,
			kernel_types::temperature_scale, compute_t, vocab_mask_trait, temperature_type>;

		using top_k_filter_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::top_k_filter, temperature_scale_trait, top_k_type>, kernel_types::top_k_filter,
			compute_t, temperature_scale_trait, top_k_type>;

		using top_p_filter_trait = kernel_traits<config_new, get_new_dims_new_2_t<kernel_types::top_p_filter, top_k_filter_trait, top_p_type>, kernel_types::top_p_filter,
			compute_t, top_k_filter_trait, top_p_type>;

		using sample_trait = kernel_traits<config_new, core_trait_dims<1, 1, 1, 1, 0>, kernel_types::sample_logits, int32_t, top_p_filter_trait, rng_state_type>;

		using mega_final_composite_traits = composite_kernel_traits<config_new, composite_kernel_types::final_norm_and_sampling, compute_t, last_token_view_trait,
			final_rms_norm_trait, final_norm_mul_trait, logits_mul_mat_trait, logits_bias_add_trait, rep_penalty_trait, presence_penalty_trait, frequency_penalty_trait,
			vocab_mask_trait, temperature_scale_trait, top_k_filter_trait, top_p_filter_trait, sample_trait>;

		using result_token_id_type = op_traits<config_new, core_types::final_norm_and_sampling, final_norm_and_sampling_types::result_token_id,
			composite_kernel_types::final_norm_and_sampling, data_strategy_types::global, allocation_strategy_types::alloc, mega_final_composite_traits>;

		using composite_ops = get_core_base_t<config_new, result_token_id_type>;
		composite_ops values{};

	struct kernel_data_ptrs {
		  protected:
			NIHILUS_ALIGN(16) const void* l_out_data;
			NIHILUS_ALIGN(16) const void* output_norm_w_data;
			NIHILUS_ALIGN(16) const void* output_w_data;
			NIHILUS_ALIGN(16) const void* temperature_data;
			NIHILUS_ALIGN(16) const void* top_k_data;
			NIHILUS_ALIGN(16) const void* top_p_data;
			NIHILUS_ALIGN(16) const void* repetition_penalty_data;
			NIHILUS_ALIGN(16) const void* presence_penalty_data;
			NIHILUS_ALIGN(16) const void* frequency_penalty_data;
			NIHILUS_ALIGN(16) const void* rep_window_data;
			NIHILUS_ALIGN(16) const void* token_history_data;
			NIHILUS_ALIGN(16) const void* logits_bias_data;
			NIHILUS_ALIGN(16) const void* allowed_vocab_mask_data;
			NIHILUS_ALIGN(16) void* result_token_id_data;
			NIHILUS_ALIGN(16) void* rng_state_data;

		  public:
			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_l_out_data() {
				return static_cast<const value_type*>(l_out_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_output_norm_w_data() {
				return static_cast<const value_type*>(output_norm_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_output_w_data() {
				return static_cast<const value_type*>(output_w_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_temperature_data() {
				return static_cast<const value_type*>(temperature_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_top_k_data() {
				return static_cast<const value_type*>(top_k_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_top_p_data() {
				return static_cast<const value_type*>(top_p_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_repetition_penalty_data() {
				return static_cast<const value_type*>(repetition_penalty_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_presence_penalty_data() {
				return static_cast<const value_type*>(presence_penalty_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_frequency_penalty_data() {
				return static_cast<const value_type*>(frequency_penalty_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_rep_window_data() {
				return static_cast<const value_type*>(rep_window_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_token_history_data() {
				return static_cast<const value_type*>(token_history_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_logits_bias_data() {
				return static_cast<const value_type*>(logits_bias_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE const value_type* get_allowed_vocab_mask_data() {
				return static_cast<const value_type*>(allowed_vocab_mask_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_result_token_id_data() {
				return static_cast<value_type*>(result_token_id_data);
			}

			template<typename value_type> NIHILUS_HOST_DEVICE value_type* get_rng_state_data() {
				return static_cast<value_type*>(rng_state_data);
			}

			NIHILUS_HOST void set_ptrs(void* result_token_id_data_new, const void* l_out_data_new, const void* output_norm_w_data_new, const void* output_w_data_new,
				const void* temperature_data_new, const void* top_k_data_new, const void* top_p_data_new, const void* repetition_penalty_data_new,
				const void* presence_penalty_data_new, const void* frequency_penalty_data_new, const void* rep_window_data_new, const void* token_history_data_new,
				void* rng_state_data_new, const void* logits_bias_data_new, const void* allowed_vocab_mask_data_new) {
				result_token_id_data	= result_token_id_data_new;
				l_out_data				= l_out_data_new;
				output_norm_w_data		= output_norm_w_data_new;
				output_w_data			= output_w_data_new;
				temperature_data		= temperature_data_new;
				top_k_data				= top_k_data_new;
				top_p_data				= top_p_data_new;
				repetition_penalty_data = repetition_penalty_data_new;
				presence_penalty_data	= presence_penalty_data_new;
				frequency_penalty_data	= frequency_penalty_data_new;
				rep_window_data			= rep_window_data_new;
				token_history_data		= token_history_data_new;
				rng_state_data			= rng_state_data_new;
				logits_bias_data		= logits_bias_data_new;
				allowed_vocab_mask_data = allowed_vocab_mask_data_new;
			}
		};

		kernel_data_ptrs data_ptrs{};

		static constexpr uint64_t total_required_bytes{ result_token_id_type::total_required_bytes };
		static constexpr bool has_total_required_bytes{ true };
	};

	template<const model_config& config_new, auto kernel_type> struct get_adjacent_value {
		using derived_type	   = core_traits<config_new, kernel_type>;
		using thread_pool_type = thread_pool<config_new>;
		using core_bases_type  = typename thread_pool<config_new>::core_bases_type;
		NIHILUS_HOST static auto& impl(auto& parse_core) {
			return *static_cast<derived_type*>(static_cast<core_bases_type*>(&parse_core));
		}
	};


}
