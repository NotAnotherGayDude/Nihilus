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

#include <nihilus/common/common.hpp>
#include <iterator>

namespace nihilus {

	template<model_arches> struct hyper_parameters;

	template<> struct hyper_parameters<model_arches::llama> {
		uint64_t current_sequence_length{};
		uint64_t kv_cache_size_per_layer{};
		uint64_t batch_size{};
		uint64_t rope_dims{};
		double rope_freqs{};
	};

}
