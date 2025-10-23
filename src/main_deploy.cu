#include <nihilus>
#include <nihilus-network/nihilus-network.hpp>

static constexpr auto model_config_00 = nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa,
	nihilus::model_arches::llama, nihilus::device_types::gpu, nihilus::batched_processing_type::enabled, nihilus::exception_type::enabled,
	nihilus::default_max_sequence_length_type{ 1024 }, nihilus::benchmark_type::enabled);
static constexpr auto model_config_01 =
	nihilus::generate_model_config(nihilus::model_sizes::llm_405B, nihilus::kernel_type_profiles::q8_gqa, nihilus::device_types::gpu, nihilus::model_generations::v3_1,
		nihilus::exception_type::enabled, nihilus::batched_processing_type::enabled, nihilus::default_max_sequence_length_type{ 131072 }, nihilus::model_arches::llama);
int32_t main(int32_t argc, char** argv) {
	nihilus::cli_params cli_args{ nihilus::harbinger::parse_cli_arguments(argc, argv) };
	nihilus::server_model_collection_type<nihilus_network::nihilus_grpc_connection, model_config_00, model_config_01> collection{ cli_args };
	collection.wait();
	return 0;
}
