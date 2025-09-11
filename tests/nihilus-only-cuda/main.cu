#include <nihilus>

int main(int32_t argc, char** argv) {
	constexpr auto model_config		   = nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa,
			   nihilus::model_arches::llama, nihilus::device_types::gpu, true, false, 65536);
	const nihilus::cli_params cli_args = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
	auto model_new_01{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args) };
	while (model_new_01->process_input()) {
	}
	return 0;
}