#include <nihilus>

static constexpr auto model_config_00 = nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa,
	nihilus::model_arches::llama, nihilus::device_types::gpu, nihilus::batched_processing_type::enabled, nihilus::exception_type::enabled,
	nihilus::default_max_sequence_length_type{ 1024 }, nihilus::benchmark_type::enabled);

int32_t main(int64_t argc, char** argv) {
	const nihilus::cli_params cli_args = nihilus::harbinger::parse_cli_arguments(argc, argv);
	nihilus::model_collection_type<model_config_00> collection{ cli_args };
	while (collection.process_input()) {
	}
	return 0;
}
