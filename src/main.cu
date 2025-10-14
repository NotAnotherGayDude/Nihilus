#include <nihilus>

static constexpr auto model_config_00 = nihilus::generate_model_config(nihilus::device_types::gpu, nihilus::model_generations::v3_1, nihilus::kernel_type_profiles::q8_gqa,
	nihilus::model_arches::llama, nihilus::default_max_sequence_length_types{ 1024 }, nihilus::benchmark_types::enabled, nihilus::model_sizes::llm_8B);
static constexpr auto model_config_01 =
	nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_405B, nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama,
		nihilus::device_types::gpu, nihilus::exception_types::disabled, nihilus::default_max_sequence_length_types{ 131072 }, nihilus::benchmark_types::enabled);

int32_t main(int64_t argc, char** argv) {
	const nihilus::cli_params cli_args = nihilus::harbinger<model_config_00>::parse_cli_arguments(argc, argv);
	nihilus::model_collection_type<model_config_00, model_config_01> collection{ cli_args, cli_args };
	while (collection.process_input()) {
	}
	return 0;
}