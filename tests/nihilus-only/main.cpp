#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nihilus>

#if !defined(LLAMA_MODEL_SIZE)
static constexpr nihilus::model_sizes model_size{ nihilus::model_sizes::llm_8B };
#else
static constexpr nihilus::model_sizes model_size{ LLAMA_MODEL_SIZE };
#endif

namespace nihilus {

	struct core_base {
		array<uint64_t, 4> strides{ 0, 0, 0, 0 };
		array<uint64_t, 4> dims{ 1, 1, 1, 1 };
		data_types data_type{};
		uint64_t data_size{};
		op_types op_type{};
		uint64_t depth{};
		uint64_t id{};
	};

	struct core : public core_base {
		array<int64_t, 3> input_ids{ -1, -1, -1 };
		kernel_types kernel_type{};
	};

	struct core_context {
		array<core_base, 512> core_bases{};
		array<core, 512> cores{};
		uint64_t current_id{};
		NIHILUS_INLINE constexpr const core_base* get_core(op_types op_type) const {
			for (auto iter = core_bases.end()-1; iter != core_bases.begin(); --iter) {
				if (iter->op_type == op_type) {
					return &*iter;
				}
			}
			for (auto iter = cores.end()-1; iter != cores.begin(); --iter) {
				if (iter->op_type == op_type) {
					return &*iter;
				}
			}
			return nullptr;
		}
		NIHILUS_INLINE constexpr core_base* get_core(op_types op_type) {
			for (auto iter = core_bases.end() - 1; iter != core_bases.begin(); --iter) {
				if (iter->op_type == op_type) {
					return &*iter;
				}
			}
			for (auto iter = cores.end() - 1; iter != cores.begin(); --iter) {
				if (iter->op_type == op_type) {
					return &*iter;
				}
			}
			return nullptr;
		}
	};

	static constexpr auto create_core_base_impl(core_context& context, data_types type, op_types op_type, std::initializer_list<uint64_t> dims) {
		core_base& result{ context.core_bases[context.current_id] };
		result.id	   = context.current_id;
		result.op_type = op_type;
		++context.current_id;

		const auto type_traits = get_type_traits(type);

		if (dims.size() > 0) {
			result.data_size = type_traits.row_size(dims.begin()[0]);
		}

		for (uint64_t i = 1; i < dims.size(); i++) {
			result.data_size *= dims.begin()[i];
		}

		for (uint64_t i = 0; i < dims.size(); i++) {
			result.dims[i] = dims.begin()[i];
		}

		result.strides[0] = type_traits.type_size;
		result.strides[1] = result.strides[0] * (result.dims[0] / type_traits.block_size);
		for (uint64_t i = 2; i < 4; i++) {
			result.strides[i] = result.strides[i - 1] * result.dims[i - 1];
		}
	};

	static constexpr auto create_core_impl(core_context& context, data_types type, op_types op_type, std::initializer_list<uint64_t> dims, std::initializer_list<uint64_t> ids) {
		core& result{ context.cores[context.current_id] };
		result.id	   = context.current_id;
		result.op_type = op_type;
		++context.current_id;

		const auto type_traits = get_type_traits(type);
		for (uint64_t i = 1; i < ids.size(); i++) {
			result.input_ids[i] *= ids.begin()[i];
		}

		if (dims.size() > 0) {
			result.data_size = type_traits.row_size(dims.begin()[0]);
		}

		for (uint64_t i = 1; i < dims.size(); i++) {
			result.data_size *= dims.begin()[i];
		}

		for (uint64_t i = 0; i < dims.size(); i++) {
			result.dims[i] = dims.begin()[i];
		}

		result.strides[0] = type_traits.type_size;
		result.strides[1] = result.strides[0] * (result.dims[0] / type_traits.block_size);
		for (uint64_t i = 2; i < 4; i++) {
			result.strides[i] = result.strides[i - 1] * result.dims[i - 1];
		}
	};

	template<kernel_types> struct create_core;

	template<> struct create_core<kernel_types::get_rows> {
		static constexpr void impl(core_context& ctx, op_types op_type, core_base* a, core_base* b) {
			enum data_types type = data_types::f32;
			if (a->data_type == data_types::i32) {
				type = a->data_type;
			}
			create_core_impl(ctx, type, op_type, { a->dims[0], b->dims[0], b->dims[1], b->dims[2] }, { a->id, b->id });

			ctx.cores[ctx.current_id].kernel_type = kernel_types::get_rows;
			ctx.cores[ctx.current_id].depth		  = detail::max(a->depth, b->depth) + 1;
			return;
		}
	};

	template<model_config config> struct build_weights {
		static constexpr auto impl(core_context& context) {

			using model_traits_type					   = model_traits_type<config>;
			constexpr uint32_t vocab_size			   = model_traits_type::vocab_size;
			constexpr uint32_t embedding_length		   = model_traits_type::embedding_length;
			constexpr uint32_t block_count			   = model_traits_type::block_count;
			constexpr uint32_t attention_head_count	   = model_traits_type::attention_head_count;
			constexpr uint32_t attention_head_count_kv = model_traits_type::attention_head_count_kv;
			constexpr uint32_t head_dim				   = model_traits_type::head_dim;
			constexpr uint32_t rope_dimension_count	   = model_traits_type::rope_dimension_count;
			constexpr uint32_t intermediate_size	   = model_traits_type::intermediate_size;

			nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::token_embd_weight,
				{ static_cast<uint64_t>(embedding_length), vocab_size });
			nihilus::create_core_base_impl(context, nihilus::data_types::f32, op_types::output_norm_weight, { static_cast<uint64_t>(embedding_length) });
			nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::output_weight, { static_cast<uint64_t>(embedding_length), vocab_size });
			nihilus::create_core_base_impl(context, nihilus::data_types::f32, op_types::rope_freqs_weight, { rope_dimension_count / 2 });

			for (uint32_t layer_idx = 0; layer_idx < block_count; ++layer_idx) {
				nihilus::create_core_base_impl(context, nihilus::data_types::f32, op_types::attn_norm_weight, { static_cast<uint64_t>(embedding_length) });
				nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::attn_q_weight,
					{ static_cast<uint64_t>(embedding_length), static_cast<uint64_t>(head_dim * attention_head_count) });
				nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::attn_k_weight,
					{ static_cast<uint64_t>(embedding_length), static_cast<uint64_t>(head_dim * attention_head_count_kv) });
				nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::attn_v_weight,
					{ static_cast<uint64_t>(embedding_length), static_cast<uint64_t>(head_dim * attention_head_count_kv) });
				nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::attn_output_weight,
					{ static_cast<uint64_t>(head_dim * attention_head_count), static_cast<uint64_t>(embedding_length) });
				nihilus::create_core_base_impl(context, nihilus::data_types::f32, op_types::ffn_norm_weight, { static_cast<uint64_t>(embedding_length) });
				nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::ffn_gate_weight,
					{ static_cast<uint64_t>(embedding_length), static_cast<uint64_t>(intermediate_size) });
				nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::ffn_up_weight,
					{ static_cast<uint64_t>(embedding_length), static_cast<uint64_t>(intermediate_size) });
				nihilus::create_core_base_impl(context, nihilus::data_types::q8_0, op_types::ffn_down_weight,
					{ static_cast<uint64_t>(intermediate_size), static_cast<uint64_t>(embedding_length) });
			}

			return context;
		}
	};

	template<model_config config> struct build_model {
		static constexpr auto impl() {
			using model_traits_type					   = model_traits_type<config>;
			//constexpr uint32_t vocab_size			   = model_traits_type::vocab_size;
			//constexpr uint32_t embedding_length		   = model_traits_type::embedding_length;
			//constexpr uint32_t block_count			   = model_traits_type::block_count;
			//constexpr uint32_t feed_forward_length	   = model_traits_type::feed_forward_length;
			//constexpr uint32_t attention_head_count	   = model_traits_type::attention_head_count;
			//constexpr uint32_t attention_head_count_kv = model_traits_type::attention_head_count_kv;
			//constexpr uint32_t head_dim				   = model_traits_type::head_dim;
			////constexpr uint32_t rope_dimension_count	   = model_traits_type::rope_dimension_count;
			//constexpr uint32_t intermediate_size	   = model_traits_type::intermediate_size;
			constexpr int32_t context_length		   = -1;
			core_context context{};
			build_weights<config>::impl(context);
			nihilus::create_core_impl(context, nihilus::data_types::i32, op_types::inp_tokens, { static_cast<uint64_t>(context_length) }, {});
			create_core<kernel_types::get_rows>::impl(context, op_types::inp_embd, context.get_core(op_types::token_embd_weight), context.get_core(op_types::inp_tokens));
			return context;
		}
	};	

};

template<size_t index_new> struct test_struct_base {
	static constexpr size_t index{ index_new };
};

struct test_struct : public test_struct_base<1> {
	using base_t = test_struct_base<0>;
	size_t index{ base_t::index };
};

int32_t main(int32_t argc, char** argv) {
	try {
		test_struct test{};
		std::cout << "CURRENT INDEX: " << test.index << std::endl;
		test.index = 0;
		static constexpr auto model_config =
			nihilus::generate_model_config(nihilus::model_generations::v3, model_size, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		nihilus::cli_params cli_args_final{ nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv) };
		auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args_final) };
		while (model_new->process_input(cli_args_final.prompt)) {
		}
	} catch (const std::exception& error) {
		std::cout << "Error: " << error.what() << std::endl;
	}
	return 0;
}