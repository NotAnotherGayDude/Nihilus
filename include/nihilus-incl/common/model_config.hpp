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

	enum class model_config_types {
		model_generation,
		model_size,
		kernel_type_profile,
		model_arch,
		exceptions,
		default_max_sequence_length,
		default_batch_size,
		kv_cache_strategy,
		user_input_type,
		rope_scaling_type,
		tokenizer_pre_type,
		kv_cache_block_size,
		use_rotary_embeddings,
		rms_norm_type,
		tokenizer_type,
		device_type,
		model_format,
		norm_epsilon,
		benchmark,
		dev
	};

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
	};

	template<const model_config& config> struct model_config_type {
		static constexpr model_generations model_generation							   = config.model_generation;
		static constexpr model_sizes model_size										   = config.model_size;
		static constexpr kernel_type_profiles kernel_type_profile					   = config.kernel_type_profile;
		static constexpr model_arches model_arch									   = config.model_arch;
		static constexpr bool exceptions								   = config.exceptions;
		static constexpr uint64_t default_max_sequence_length = config.default_max_sequence_length;
		static constexpr uint64_t default_batch_size				   = config.default_batch_size;
		static constexpr kv_cache_strategies kv_cache_strategy						   = config.kv_cache_strategy;
		static constexpr rope_scaling_types rope_scaling_type						   = config.rope_scaling_type;
		static constexpr tokenizer_pre_types tokenizer_pre_type						   = config.tokenizer_pre_type;
		static constexpr uint64_t kv_cache_block_size				   = config.kv_cache_block_size;
		static constexpr bool use_rotary_embeddings				   = config.use_rotary_embeddings;
		static constexpr rms_norm_types rms_norm_type								   = config.rms_norm_type;
		static constexpr tokenizer_types tokenizer_type								   = config.tokenizer_type;
		static constexpr device_types device_type									   = config.device_type;
		static constexpr model_formats format										   = config.model_format;
		static constexpr float norm_epsilon											   = config.norm_epsilon;
		static constexpr bool benchmark									   = config.benchmark;
		static constexpr bool dev												   = config.dev;

		NIHILUS_HOST static constexpr model_config get_config() {
			return config;
		}
	};

	template<typename model_config> static constexpr model_config global_config{ model_config::get_config() };

}
