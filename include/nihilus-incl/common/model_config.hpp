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

#include <nihilus-incl/common/common.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>

namespace nihilus {

	struct model_config {
		model_generations model_generation{};
		model_sizes model_size{};
		kernel_type_profiles kernel_type_profile{};
		model_arches model_arch{};
		bool exceptions{};
		uint64_t default_max_sequence_length{};
		uint64_t default_batch_size{};
		kv_cache_strategies kv_cache_strategy{};
		user_input_types user_input_type{};
		rope_scaling_types rope_scaling_type{};
		tokenizer_pre_types tokenizer_pre_type{};
		uint64_t kv_cache_block_size{};
		bool use_rotary_embeddings{};
		rms_norm_types rms_norm_type{};
		tokenizer_types tokenizer_type{};
		device_types device_type{};
		model_formats model_format{};
		float norm_epsilon{};
		bool benchmark{};
		bool dev{};

		NIHILUS_HOST consteval auto update_model_generation(model_generations value) const {
			model_config return_value{ *this };
			return_value.model_generation = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_model_size(model_sizes value) const {
			model_config return_value{ *this };
			return_value.model_size = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_kernel_type_profile(kernel_type_profiles value) const {
			model_config return_value{ *this };
			return_value.kernel_type_profile = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_model_arch(model_arches value) const {
			model_config return_value{ *this };
			return_value.model_arch = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_exceptions(bool value) const {
			model_config return_value{ *this };
			return_value.exceptions = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_default_max_sequence_length(uint64_t value) const {
			model_config return_value{ *this };
			return_value.default_max_sequence_length = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_default_batch_size(uint64_t value) const {
			model_config return_value{ *this };
			return_value.default_batch_size = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_kv_cache_strategy(kv_cache_strategies value) const {
			model_config return_value{ *this };
			return_value.kv_cache_strategy = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_user_input_type(user_input_types value) const {
			model_config return_value{ *this };
			return_value.user_input_type = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_rope_scaling_type(rope_scaling_types value) const {
			model_config return_value{ *this };
			return_value.rope_scaling_type = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_tokenizer_pre_type(tokenizer_pre_types value) const {
			model_config return_value{ *this };
			return_value.tokenizer_pre_type = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_kv_cache_block_size(uint64_t value) const {
			model_config return_value{ *this };
			return_value.kv_cache_block_size = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_use_rotary_embeddings(bool value) const {
			model_config return_value{ *this };
			return_value.use_rotary_embeddings = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_rms_norm_type(rms_norm_types value) const {
			model_config return_value{ *this };
			return_value.rms_norm_type = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_tokenizer_type(tokenizer_types value) const {
			model_config return_value{ *this };
			return_value.tokenizer_type = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_device_type(device_types value) const {
			model_config return_value{ *this };
			return_value.device_type = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_model_format(model_formats value) const {
			model_config return_value{ *this };
			return_value.model_format = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_norm_epsilon(float value) const {
			model_config return_value{ *this };
			return_value.norm_epsilon = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_benchmark(bool value) const {
			model_config return_value{ *this };
			return_value.benchmark = value;
			return return_value;
		}

		NIHILUS_HOST consteval auto update_dev(bool value) const {
			model_config return_value{ *this };
			return_value.dev = value;
			return return_value;
		}
	};

}
