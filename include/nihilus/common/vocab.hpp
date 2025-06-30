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

#include <nihilus/common/parse_entity.hpp>
#include <nihilus/common/common.hpp>
#include <unordered_map>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>
#include <map>
#include <set>

namespace nihilus {

	template<model_arches arch, vocab_types type, vocab_pre_types pre> struct vocab_traits;

	template<> struct vocab_traits<model_arches::llama, vocab_types::bpe, vocab_pre_types::llama3> {
		static constexpr vocab_pre_types pre_type		 = vocab_pre_types::llama3;
		static constexpr int32_t max_token_len			 = 256;
		static constexpr token special_bos_id			 = 128000;
		static constexpr token special_eos_id			 = 128009;
		static constexpr token special_eot_id			 = 128009;
		static constexpr token special_eom_id			 = 128008;
		static constexpr token special_unk_id			 = -1;
		static constexpr token special_sep_id			 = -1;
		static constexpr token special_pad_id			 = -1;
		static constexpr token special_mask_id			 = -1;
		static constexpr token linefeed_id				 = 13;
		static constexpr token special_fim_pre_id		 = -1;
		static constexpr token special_fim_suf_id		 = -1;
		static constexpr token special_fim_mid_id		 = -1;
		static constexpr token special_fim_pad_id		 = -1;
		static constexpr token special_fim_rep_id		 = -1;
		static constexpr token special_fim_sep_id		 = -1;
		static constexpr bool add_space_prefix			 = false;
		static constexpr bool add_bos					 = true;
		static constexpr bool add_eos					 = false;
		static constexpr bool ignore_merges				 = false;
		static constexpr bool clean_spaces				 = true;
		static constexpr bool remove_extra_whitespaces	 = false;
		static constexpr bool escape_whitespaces		 = true;
		static constexpr bool treat_whitespace_as_suffix = false;
	};
}