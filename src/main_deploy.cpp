#include <nihilus>
#include <nihilus-network/nihilus-network.hpp>
using namespace nihilus;

int32_t main(int32_t argc, char** argv) {
	static constexpr auto model_config_00 =
		nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama,
			nihilus::device_types::cpu, nihilus::exception_type::enabled, nihilus::default_max_sequence_length_types{ 1024 }, nihilus::benchmark_type::enabled);
	static constexpr auto model_config_01 = nihilus::generate_model_config(nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::device_types::cpu,
		nihilus::model_generations::v3_1, nihilus::exception_type::enabled, nihilus::default_max_sequence_length_types{ 131072 }, nihilus::model_arches::llama);
	cli_params cli_args_01{ harbinger::parse_cli_arguments(argc, argv) };
	cli_params cli_args_02{ harbinger::parse_cli_arguments(argc, argv) };
	nihilus::server_model_collection_type<nihilus_network::nihilus_grpc_connection, model_config_00, model_config_01> collection{ cli_args_01, cli_args_02 };
	collection.wait();
	return 0;
}
