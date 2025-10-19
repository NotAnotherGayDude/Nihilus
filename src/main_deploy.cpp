#include <nihilus>
#include <nihilus-network/nihilus-network.hpp>
using namespace nihilus;

int32_t main(int32_t argc, char** argv) {
	static constexpr auto model_config_00 =
		nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama,
			nihilus::device_types::cpu, nihilus::exception_types::enabled, nihilus::default_max_sequence_length_types{ 1024 }, nihilus::benchmark_types::enabled);
	cli_params_config<nihilus_network::nihilus_grpc_connection_config> cli_args_01{ .config = nihilus_network::nihilus_grpc_connection_config{ .ip = "127.0.0.1", .port = 51000 },
		.params																			 = harbinger<model_config_00>::parse_cli_arguments(argc, argv) };
	cli_params_config<nihilus_network::nihilus_grpc_connection_config> cli_args_02{ .config = nihilus_network::nihilus_grpc_connection_config{ .ip = "127.0.0.1", .port = 51001 },
		.params																			 = harbinger<model_config_00>::parse_cli_arguments(argc, argv) };
	nihilus::server_model_collection_type<nihilus_network::nihilus_grpc_connection, model_config_00, model_config_00> collection{ cli_args_01, cli_args_02 };
	collection.wait();
	return 0;
}
