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

#include <nihilus-incl/cpu/memory_mapped_file.hpp>
#include <nihilus-incl/common/parse_entity.hpp>
#include <nihilus-incl/common/optional.hpp>
#include <nihilus-incl/infra/tokenizer.hpp>
#include <nihilus-incl/infra/core_traits.hpp>

namespace nihilus {

	template<typename config_type> struct file_loader : memory_mapped_file<config_type> {
		NIHILUS_HOST file_loader() noexcept {
		}
		NIHILUS_HOST file_loader(std::string_view path, uint64_t file_offset_new = 0) : memory_mapped_file<config_type>{ path, file_offset_new } {
		}
	};

}
