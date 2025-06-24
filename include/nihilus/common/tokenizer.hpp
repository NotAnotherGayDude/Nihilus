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

#include <nihilus/common/tokenizer.hpp>
#include <nihilus/common/config.hpp>
#include <iterator>

namespace nihilus {

	struct bpe_token {
		std::string token;
		int32_t id;
		float score;
	};

	template<model_arch arch> struct tokenizer;

	template<> struct tokenizer<model_arch::llama> {
		NIHILUS_FORCE_INLINE tokenizer() noexcept = default;

		NIHILUS_FORCE_INLINE void load_from_model(const std::string& model_path) {
			load_sentencepiece_vocab(model_path);
			build_merge_rules();
		}

		template<typename token_input_type> NIHILUS_FORCE_INLINE size_t tokenize(std::string_view input_text, token_input_type* output_tokens, size_t max_tokens) {
			if (input_text.empty())
				return 0;

			size_t token_count = 0;
			if (add_bos_token && token_count < max_tokens) {
				output_tokens[token_count++] = bos_token_id;
			}


			auto tokens = encode_bpe(input_text);

			// Copy to output buffer (respecting max_tokens limit)
			for (size_t i = 0; i < tokens.size() && token_count < max_tokens; ++i) {
				output_tokens[token_count++] = static_cast<token_input_type>(tokens[i]);
			}

			return token_count;
		}

		// Decode tokens back to text (useful for debugging/output)
		NIHILUS_FORCE_INLINE std::string decode(const std::vector<int32_t>& tokens) {
			std::string result;
			result.reserve(tokens.size() * 4);// Rough estimate

			for (int32_t token_id: tokens) {
				if (token_id >= 0 && token_id < static_cast<int32_t>(id_to_token.size())) {
					std::string token = id_to_token[token_id];

					// Handle special tokens
					if (token == "<s>" || token == "</s>" || token == "<unk>") {
						continue;// Skip special tokens in output
					}

					// Replace underscores with spaces (SentencePiece format)
					//if (token.front() == '▁') {
					//result += ' ';
					//						result += token.substr(3);// Skip UTF-8 encoded ▁
					//} else {
					//result += token;
					//}
				}
			}

			return result;
		}

		// Get vocab size for model validation
		NIHILUS_FORCE_INLINE size_t vocab_size() const {
			return id_to_token.size();
		}

	  private:
		// Core tokenizer data
		std::vector<std::string> id_to_token;
		std::unordered_map<std::string, int32_t> token_to_id;
		std::vector<std::pair<std::string, std::string>> merge_rules;

		// Special token IDs
		int32_t bos_token_id = 1;// <s>
		int32_t eos_token_id = 2;// </s>
		int32_t unk_token_id = 0;// <unk>
		bool add_bos_token	 = true;

		// Load SentencePiece vocabulary from GGUF file
		NIHILUS_FORCE_INLINE void load_sentencepiece_vocab(const std::string& model_path) {
			// Simplified - in practice you'd parse the GGUF file
			// For now, assume we have the vocab loaded

			// Reserve space for typical Llama vocab
			id_to_token.reserve(32000);
			token_to_id.reserve(32000);

			// Example special tokens (you'd load these from the model file)
			add_token("<unk>", 0);
			add_token("<s>", 1);
			add_token("</s>", 2);

			// Load the rest of the vocabulary from model file
			// This is where you'd parse the GGUF tokenizer section
		}

		// Build BPE merge rules for subword tokenization
		NIHILUS_FORCE_INLINE void build_merge_rules() {
			// Load merge rules from model or build from vocab
			// BPE merges are typically stored in the model file
		}

		// Add token to vocabulary
		NIHILUS_FORCE_INLINE void add_token(const std::string& token, int32_t id) {
			if (id >= static_cast<int32_t>(id_to_token.size())) {
				id_to_token.resize(id + 1);
			}
			id_to_token[id]	   = token;
			token_to_id[token] = id;
		}

		// Core BPE encoding algorithm
		NIHILUS_FORCE_INLINE std::vector<int32_t> encode_bpe(std::string_view text) {
			std::vector<int32_t> result;
			result.reserve(text.length());// Rough estimate

			// Simplified BPE implementation
			// In practice, you'd implement full BPE algorithm here

			// For now, simple whitespace + subword splitting
			std::string current_token;
			for (char c: text) {
				if (c == ' ') {
					if (!current_token.empty()) {
						encode_token(current_token, result);
						current_token.clear();
					}
					// Add space token
					encode_token("▁", result);
				} else {
					current_token += c;
				}
			}

			if (!current_token.empty()) {
				encode_token(current_token, result);
			}

			return result;
		}

		// Encode a single token
		NIHILUS_FORCE_INLINE void encode_token(const std::string& token, std::vector<int32_t>& output) {
			auto it = token_to_id.find(token);
			if (it != token_to_id.end()) {
				output.push_back(it->second);
			} else {
				// Handle unknown token - could implement subword fallback here
				output.push_back(unk_token_id);
			}
		}
	};


}
