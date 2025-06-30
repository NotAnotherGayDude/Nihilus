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
		static constexpr token special_bos_id			 = 11;
		static constexpr token special_eos_id			 = 11;
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

	struct token_data {
		std::string text;
		float score;
		tokens att;
	};

	struct pair_hash {
		NIHILUS_FORCE_INLINE size_t operator()(const std::pair<std::string_view, std::string_view>& p) const {
			return std::hash<std::string_view>{}(p.first) ^ (std::hash<std::string_view>{}(p.second) << 1);
		}
	};

	template<model_config config, vocab_types type, typename derivd_type> struct vocab : public vocab_traits<config.arch, type, config.vocab_pre_type> {
		std::unordered_map<std::string_view, token> token_to_id;
		std::vector<token_data> id_to_token;

		std::vector<token> cache_special_tokens;
		std::vector<std::string> cache_token_to_piece;

		std::unordered_map<std::pair<std::string_view, std::string_view>, int32_t, pair_hash> bpe_ranks;

		std::set<token> special_eog_ids;

		std::vector<uint8_t> precompiled_charsmap;

		enum vocab_types get_type() const {
			return type;
		}

		NIHILUS_FORCE_INLINE consteval std::string_view type_name() const {
			if constexpr (type == vocab_types::none) {
				return "no vocab";
			} else if constexpr (type == vocab_types::spm) {
				return "SPM";
			} else if constexpr (type == vocab_types::bpe) {
				return "BPE";
			} else if constexpr (type == vocab_types::wpm) {
				return "WPM";
			} else if constexpr (type == vocab_types::ugm) {
				return "UGM";
			} else if constexpr (type == vocab_types::rwkv) {
				return "RWKV";
			} else {
				return "unknown";
			}
		}

		bool is_normal(token id) const {
			return static_cast<size_t>(id_to_token[id].att) & static_cast<size_t>(tokens::normal);
		}

		bool is_unknown(token id) const {
			return static_cast<size_t>(id_to_token[id].att) & static_cast<size_t>(tokens::unknown);
		}

		bool is_control(token id) const {
			return static_cast<size_t>(id_to_token[id].att) & static_cast<size_t>(tokens::control);
		}

		bool is_byte(token id) const {
			return static_cast<size_t>(id_to_token[id].att) & static_cast<size_t>(tokens::byte);
		}

		bool is_user_defined(token id) const {
			return static_cast<size_t>(id_to_token[id].att) & static_cast<size_t>(tokens::user_defined);
		}

		bool is_unused(token id) const {
			return static_cast<size_t>(id_to_token[id].att) & static_cast<size_t>(tokens::unused);
		}

		bool is_eog(token id) const {
			return id != token_null && special_eog_ids.count(id) > 0;
		}

		uint8_t token_to_byte(token id) const;
		tokens token_get(token id) const;
		std::string token_to_piece_for_cache(token token, bool special) const;
		std::vector<token> tokenize(const std::string& raw_text, bool add_special, bool parse_special = false) const;
		int32_t tokenize(const char* text, int32_t text_len, token* tokens, int32_t n_tokens_max, bool add_special, bool parse_special) const;
		int32_t token_to_piece(token token, char* buf, int32_t length, int32_t lstrip, bool special) const;
		const std::string& token_to_piece(token token) const;
		int32_t detokenize(const token* tokens, int32_t n_tokens, char* text, int32_t text_len_max, bool remove_special, bool unparse_special) const;
		std::string detokenize(const std::vector<token>& tokens, bool special) const;
	};
	/*
	struct llama_vocab {

		llama_vocab();
		~llama_vocab();

		void load(llama_model_loader& ml, const LLM_KV& kv);

		enum llama_vocab_type get_type() const;
		enum llama_vocab_pre_type get_pre_type() const;
		f
		uint32_t n_tokens() const;
		uint32_t n_token_types() const;

		std::string type_name() const;

		bool is_normal(token id) const;
		bool is_unknown(token id) const;
		bool is_control(token id) const;
		bool is_byte(token id) const;
		bool is_user_defined(token id) const;
		bool is_unused(token id) const;
		bool is_eog(token id) const;

		uint8_t token_to_byte(token id) const;
		token byte_to_token(uint8_t ch) const;

		token text_to_token(const std::string& text) const;

		const token_data& get_token_data(token id) const;

		const char* token_get_text(token id) const;
		float token_get_score(token id) const;
		token token_get(token id) const;

		token token_bos() const;
		token token_eos() const;
		token token_eot() const;
		token token_eom() const;
		token token_unk() const;
		token token_sep() const;
		token token_nl() const;
		token token_pad() const;

		token token_prefix() const;
		token token_middle() const;
		token token_suffix() const;

		token token_fim_pre() const;
		token token_fim_suf() const;
		token token_fim_mid() const;
		token token_fim_pad() const;
		token token_fim_rep() const;
		token token_fim_sep() const;

		bool get_add_space_prefix() const;
		bool get_add_bos() const;
		bool get_add_eos() const;
		bool get_ignore_merges() const;
		bool get_clean_spaces() const;
		bool get_remove_extra_whitespaces() const;
		bool get_escape_whitespaces() const;
		bool get_treat_whitespace_as_suffix() const;

		int32_t max_token_len() const;

		int32_t find_bpe_rank(const std::string& token_left, const std::string& token_right) const;

		int32_t tokenize(const char* text, int32_t text_len, token* tokens, int32_t n_tokens_max, bool add_special, bool parse_special) const;

		std::vector<token> tokenize(const std::string& raw_text, bool add_special, bool parse_special = false) const;

		// does not write null-terminator to buf
		int32_t token_to_piece(token token, char* buf, int32_t length, int32_t lstrip, bool special) const;

		// use cached data
		const std::string& token_to_piece(token token) const;

		int32_t detokenize(const token* tokens, int32_t n_tokens, char* text, int32_t text_len_max, bool remove_special, bool unparse_special) const;

		std::string detokenize(const std::vector<token>& tokens, bool special) const;

		void print_info() const;

	  private:
		struct impl;
		std::unique_ptr<impl> pimpl;
	};*/


}