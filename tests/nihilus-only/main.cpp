#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//#include <BnchSwt/BenchmarkSuite.hpp>
#include <nihilus/index.hpp>

using namespace nihilus;
/*
template<typename EnumType> NIHILUS_FORCE_INLINE static consteval auto get_enum_count() {
	if constexpr (std::is_same_v<EnumType, model_sizes>) {
		return static_cast<uint64_t>(model_sizes::count);
	} else if constexpr (std::is_same_v<EnumType, model_arches>) {
		return static_cast<uint64_t>(model_arches::count);
	}
}

template<uint64_t N> using make_index_sequence_t = std::make_index_sequence<N>;

using model_sizes_indices  = make_index_sequence_t<static_cast<uint64_t>(model_sizes::count)>;
using model_arches_indices = make_index_sequence_t<static_cast<uint64_t>(model_arches::count)>;

template<uint64_t size_idx, uint64_t arch_idx> NIHILUS_FORCE_INLINE static consteval auto make_single_config() {
	return generate_model_config(model_generations::v1_v2,
		static_cast<model_sizes>(size_idx),
		kernel_type_profiles::fp16_mha,
		static_cast<model_arches>(arch_idx),
		false,
		kv_cache_strategies::paged,
		false,
		rope_scaling_types::linear,
		vocab_pre_types::llama3,
		16,
		true,
		true,
		norm_types::rms_standard,
		vocab_types::bpe,
		model_format::gguf,
		1e-6f,
		false
	);
}

template<uint64_t size_idx, size_t... arch_indices> NIHILUS_FORCE_INLINE static consteval auto make_configs_for_size(std::index_sequence<arch_indices...>) {
	return nihilus::make_tuple(make_single_config<size_idx, arch_indices>()...);
}

template<size_t... size_indices> NIHILUS_FORCE_INLINE static consteval auto generate_all_combinations_fold(std::index_sequence<size_indices...>) {
	return nihilus::tuple_cat(make_configs_for_size<size_indices>(model_arches_indices{})...);
}

NIHILUS_FORCE_INLINE static consteval auto get_all_model_configurations() {
	return generate_all_combinations_fold(model_sizes_indices{});
}

template<size_t... Indices> NIHILUS_FORCE_INLINE static consteval auto generate_cartesian_product(std::index_sequence<Indices...>) {
	constexpr auto arch_count = get_enum_count<model_arches>();

	return nihilus::make_tuple(make_single_config<Indices / arch_count,
		Indices % arch_count
		>()...);
}

NIHILUS_FORCE_INLINE static consteval auto get_all_model_configurations_single_fold() {
	constexpr auto size_count		  = get_enum_count<model_sizes>();
	constexpr auto arch_count		  = get_enum_count<model_arches>();
	constexpr auto total_combinations = size_count * arch_count;

	return generate_cartesian_product(std::make_index_sequence<total_combinations>{});
}

inline static constexpr auto all_model_configurations = get_all_model_configurations_single_fold();

using all_model_configs_tuple_type = decltype(all_model_configurations);

inline static constexpr auto total_model_combinations = nihilus::tuple_size_v<all_model_configs_tuple_type>;

template<uint64_t index> NIHILUS_FORCE_INLINE static consteval auto get_model_config() {
	static_assert(index < total_model_combinations, "Index out of range for model configurations");
	return nihilus::get<index>(all_model_configurations);
}

template<model_sizes size, model_arches arch> NIHILUS_FORCE_INLINE static consteval auto find_model_config() {
	constexpr auto arch_count		 = get_enum_count<model_arches>();
	constexpr auto size_index		 = static_cast<uint64_t>(size);
	constexpr auto arch_index		 = static_cast<uint64_t>(arch);
	constexpr auto combination_index = size_index * arch_count + arch_index;

	return get_model_config<combination_index>();
}
*/
int main(int argc, char** argv) {
	try {
		nihilus::stop_watch stop_watch_val{ 0 };
		static constexpr auto model_config = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa,
			nihilus::model_arches::llama, false);
		nihilus::cli_params cli_args_final;
		cli_args_final = { nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv) };
		auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args_final) };
		while (model_new->process_input(cli_args_final.prompt)) {
		}
		return 0;
		//cli_args_final.n_tokens;
	} catch (const std::exception& error) {
		std::cout << "Error: " << error.what() << std::endl;
	}
	return 0;
}