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

#include <nihilus/common/model_traits.hpp>
#include <nihilus/common/vocab.hpp>
#include <unordered_map>
#include <iterator>
#include <queue>

namespace nihilus {

	template<model_arches arch> struct tokenizer_parameters;

	template<> struct tokenizer_parameters<model_arches::llama> {
		std::vector<int64_t> token_types{};
		std::vector<std::string> tokens{};
		std::vector<std::string> merges{};
		std::string chat_template{};
		std::string pre{};
	};

	struct bpe_token {
		std::string token;
		int32_t id;
		float score;
	};

	template<model_config config, typename derived_type, model_arches arch, vocab_types> struct tokenizer;

	struct nihilus_symbol {
		int32_t prev;
		int32_t next;
		const char* text;
		size_t n;
	};

	struct nihilus_bigram_bpe {
		struct comparator {
			NIHILUS_FORCE_INLINE bool operator()(const nihilus_bigram_bpe& l, const nihilus_bigram_bpe& r) const {
				return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
			}
		};

		using queue_storage = std::vector<nihilus_bigram_bpe>;
		using queue			= std::priority_queue<nihilus_bigram_bpe, queue_storage, comparator>;

		int32_t left;
		int32_t right;
		std::string text;
		int32_t rank;
		size_t size;
	};

	template<model_config config, typename derived_type, vocab_types vocab_type_new> struct tokenizer<config, derived_type, model_arches::llama, vocab_type_new>
		: public tokenizer_parameters<model_arches::llama>, public vocab<config, vocab_type_new, tokenizer<config, derived_type, model_arches::llama, vocab_type_new>> {
		using vocab_type						  = vocab<config, vocab_type_new, tokenizer<config, derived_type, model_arches::llama, vocab_type_new>>;
		NIHILUS_FORCE_INLINE tokenizer() noexcept = default;
		using model_traits_type					  = model_traits<config.arch, config.model_size, config.model_generation>;
		static constexpr std::string_view regex_exprs{ [] {
			if constexpr (vocab_type::pre_type == vocab_pre_types::llama3) {
				return "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
					   "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
			} else {
				return std::string_view{};
			}
		}() };
		std::unordered_map<std::string, int32_t> token_to_id;
		std::unordered_map<std::pair<std::string_view, std::string_view>, int32_t, pair_hash> bpe_ranks;

		NIHILUS_FORCE_INLINE void load_vocabulary() {
			token_to_id.clear();
			for (size_t i = 0; i < tokens.size(); ++i) {
				token_to_id[tokens[i]] = static_cast<int32_t>(i);
			}
			bpe_ranks.clear();
			for (size_t i = 0; i < merges.size(); ++i) {
				const std::string& merge_str = merges[i];
				size_t space_pos			 = merge_str.find(' ');
				if (space_pos != std::string::npos) {
					std::string left		   = merge_str.substr(0, space_pos);
					std::string right		   = merge_str.substr(space_pos + 1);
					bpe_ranks[{ left, right }] = static_cast<int32_t>(i);
				}
			}
		}

		NIHILUS_FORCE_INLINE uint64_t tokenize(const std::string& input_text, int32_t* output_tokens) {
			std::vector<int32_t> temp_tokens;
			if constexpr (vocab_type::add_bos && vocab_type::special_bos_id > 0) {
				temp_tokens.push_back(static_cast<int32_t>(vocab_type::special_bos_id));
			}

			std::vector<std::string> word_collection = gpt2_style_split(input_text);

			for (const auto& word: word_collection) {
				tokenize_word(word, temp_tokens);
			}

			if constexpr (vocab_type::add_eos && vocab_type::special_eos_id > 0) {
				temp_tokens.push_back(static_cast<int32_t>(vocab_type::special_eos_id));
			}

			for (size_t i = 0; i < temp_tokens.size(); ++i) {
				output_tokens[i] = temp_tokens[i];
			}

#if defined(NIHILUS_DEBUG)
			print_tokenization_debug(input_text, temp_tokens);
#endif
			return temp_tokens.size();
		}

	  private:
		std::vector<nihilus_symbol> symbols;
		nihilus_bigram_bpe::queue work_queue;

		NIHILUS_FORCE_INLINE std::vector<std::string> gpt2_style_split(std::string_view text) {
			std::vector<std::string> result;
			size_t start = text.find_first_not_of(" \t\n\r");
			if (start == std::string::npos)
				return result;
			text = text.substr(start);

			size_t end = text.find_last_not_of(" \t\n\r");
			if (end != std::string::npos) {
				text = text.substr(0, end + 1);
			}


			bool is_first_word = true;
			size_t i		   = 0;

			while (i < text.length()) {
				std::string token;

				if (is_first_word) {
					while (i < text.length() && std::isalpha(text[i])) {
						token += text[i];
						i++;
					}
					is_first_word = false;
				} else {
					if (i < text.length() && std::isspace(text[i])) {
						token += "Ġ";
						i++;
					}

					if (i < text.length()) {
						if (std::isalpha(text[i])) {
							while (i < text.length() && std::isalpha(text[i])) {
								token += text[i];
								i++;
							}
						} else if (std::isdigit(text[i])) {
							while (i < text.length() && std::isdigit(text[i])) {
								token += text[i];
								i++;
							}
						} else if (!std::isspace(text[i])) {
							token += text[i];
							i++;
						}
					}
				}

				if (!token.empty()) {
					result.push_back(token);
				}
			}

			return result;
		}

		NIHILUS_FORCE_INLINE void tokenize_word(const std::string& word, std::vector<int32_t>& output) {
			if (word.empty())
				return;

			auto direct_match = token_to_id.find(word);
			if (direct_match != token_to_id.end()) {
				output.push_back(direct_match->second);
				return;
			}

			symbols.clear();
			work_queue = nihilus_bigram_bpe::queue();

			int32_t index = 0;
			size_t offset = 0;

			while (offset < word.size()) {
				nihilus_symbol sym;
				size_t char_len = std::min(word.size() - offset, static_cast<size_t>(unicode_len_utf8(word[offset])));
				sym.text		= word.c_str() + offset;
				sym.n			= char_len;
				offset += sym.n;
				sym.prev = index - 1;
				sym.next = offset == word.size() ? -1 : index + 1;
				index++;
				symbols.emplace_back(sym);
			}

			for (int32_t i = 1; i < static_cast<int32_t>(symbols.size()); ++i) {
				add_new_bigram(i - 1, i);
			}

			while (!work_queue.empty()) {
				auto bigram = work_queue.top();
				work_queue.pop();

				auto& left_symbol  = symbols[static_cast<uint64_t>(bigram.left)];
				auto& right_symbol = symbols[static_cast<uint64_t>(bigram.right)];

				if (left_symbol.n == 0 || right_symbol.n == 0) {
					continue;
				}

				std::string left_token(left_symbol.text, left_symbol.n);
				std::string right_token(right_symbol.text, right_symbol.n);

				if (left_token + right_token != bigram.text) {
					continue;
				}

				left_symbol.n += right_symbol.n;
				right_symbol.n = 0;

				left_symbol.next = right_symbol.next;
				if (right_symbol.next >= 0) {
					symbols[static_cast<uint64_t>(right_symbol.next)].prev = bigram.left;
				}

				add_new_bigram(left_symbol.prev, bigram.left);
				add_new_bigram(bigram.left, left_symbol.next);
			}

			for (int64_t i = 0; i != -1; i = symbols[static_cast<uint64_t>(i)].next) {
				auto& symbol = symbols[static_cast<uint64_t>(i)];
				if (symbol.n == 0)
					continue;

				std::string str(symbol.text, symbol.n);
				auto it = token_to_id.find(str);

				if (it != token_to_id.end()) {
					output.push_back(it->second);
				} else {
					for (char c: str) {
						std::string byte_str(1, c);
						auto byte_it = token_to_id.find(byte_str);
						if (byte_it != token_to_id.end()) {
							output.push_back(byte_it->second);
						}
					}
				}
			}
		}

		NIHILUS_FORCE_INLINE void add_new_bigram(int32_t left, int32_t right) {
			if (left == -1 || right == -1)
				return;

			std::string_view left_token(symbols[static_cast<uint64_t>(left)].text, symbols[static_cast<uint64_t>(left)].n);
			std::string_view right_token(symbols[static_cast<uint64_t>(right)].text, symbols[static_cast<uint64_t>(right)].n);

			auto it = bpe_ranks.find({ left_token, right_token });
			if (it == bpe_ranks.end())
				return;

			nihilus_bigram_bpe bigram;
			bigram.left	 = left;
			bigram.right = right;
			bigram.text	 = static_cast<std::string>(left_token) + static_cast<std::string>(right_token);
			bigram.size	 = left_token.size() + right_token.size();
			bigram.rank	 = it->second;

			work_queue.push(bigram);
		}

		NIHILUS_FORCE_INLINE size_t unicode_len_utf8(char c) {
			if ((c & 0x80) == 0)
				return 1;
			if ((c & 0xE0) == 0xC0)
				return 2;
			if ((c & 0xF0) == 0xE0)
				return 3;
			if ((c & 0xF8) == 0xF0)
				return 4;
			return 1;
		}

		NIHILUS_FORCE_INLINE void print_tokenization_debug(const std::string& input_text, const std::vector<int32_t>& tokens) {
			std::cout << "=== NIHILUS BPE TOKENIZATION DEBUG ===" << std::endl;
			std::cout << "system_info: n_threads = " << std::thread::hardware_concurrency() << " | NIHILUS ENGINE | BPE VOCAB | 432% FASTER |" << std::endl;
			//std::cout << "vocab_type: " << static_cast<int32_t>(vocab_type) << " (BPE)" << std::endl;
			std::cout << "pre_type: " << pre << std::endl;
			std::cout << "Input text: \"" << input_text << "\"" << std::endl;
			std::cout << "Token count: " << tokens.size() << std::endl;

			std::cout << "Tokens: ";
			for (size_t i = 0; i < tokens.size(); ++i) {
				std::cout << "[" << i << "]=" << tokens[i];
				if (i < tokens.size() - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

			std::cout << "Token strings: ";
			for (size_t i = 0; i < tokens.size(); ++i) {
				std::cout << "[" << i << "]=" << tokens[i];
				if (i < tokens.size() - 1)
					std::cout << " ";
			}
			std::cout << std::endl;
			std::cout << "=================================" << std::endl;
		}
	};

}
