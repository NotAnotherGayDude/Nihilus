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

#include <nihilus-incl/common/model_traits.hpp>
#include <nihilus-incl/common/tokenizer_traits.hpp>
#include <unordered_map>
#include <iterator>
#include <queue>

namespace nihilus {

	template<model_arches arch> struct tokenizer_parameters;

	template<> struct tokenizer_parameters<model_arches::llama> {
		std::unordered_map<std::string_view, token> tokens{};
		vector<std::string_view> merges{};
		vector<int32_t> token_types{};
		std::string_view chat_template{};
		std::string_view pre{};
	};

	struct bpe_token {
		std::string token{};
		float score{};
		int32_t id{};
	};

	template<model_config config, model_arches arch, tokenizer_types> struct tokenizer;

	struct nihilus_symbol {
		const char* text{};
		int32_t prev{};
		int32_t next{};
		uint64_t n{};
	};

	template<typename stored_type, typename comparator> struct priority_queue {
		array<stored_type, 200> data{};
		size_t currently_stored_size{};
		comparator comp{};

		NIHILUS_INLINE bool empty() const {
			return currently_stored_size == 0;
		}

		NIHILUS_INLINE const stored_type& top() const {
			return data[0];
		}

		template<typename stored_type_new> NIHILUS_INLINE void push(stored_type_new&& value) {
			data[currently_stored_size] = std::forward<stored_type_new>(value);
			bubble_up(currently_stored_size);
			++currently_stored_size;
		}

		NIHILUS_INLINE void pop() {
			if (currently_stored_size > 0) {
				data[0] = data[currently_stored_size - 1];
				--currently_stored_size;
				if (currently_stored_size > 0) {
					bubble_down(0);
				}
			}
		}

	  private:
		NIHILUS_INLINE void bubble_up(size_t index) {
			while (index > 0) {
				size_t parent = (index - 1) / 2;
				if (!comp(data[index], data[parent]))
					break;
				std::swap(data[index], data[parent]);
				index = parent;
			}
		}

		NIHILUS_INLINE void bubble_down(size_t index) {
			while (true) {
				size_t left	   = 2 * index + 1;
				size_t right   = 2 * index + 2;
				size_t largest = index;

				if (left < currently_stored_size && comp(data[left], data[largest])) {
					largest = left;
				}
				if (right < currently_stored_size && comp(data[right], data[largest])) {
					largest = right;
				}
				if (largest == index)
					break;

				std::swap(data[index], data[largest]);
				index = largest;
			}
		}
	};

	struct nihilus_bigram_bpe {
		struct comparator {
			NIHILUS_INLINE bool operator()(const nihilus_bigram_bpe& l, const nihilus_bigram_bpe& r) const {
				return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
			}
		};

		using queue			= priority_queue<nihilus_bigram_bpe, comparator>;

		std::string text{};
		uint64_t size{};
		int32_t right{};
		int32_t left{};
		int32_t rank{};
	};

	struct pair_hash {
		NIHILUS_INLINE uint64_t operator()(const std::pair<std::string_view, std::string_view>& p) const noexcept {
			return std::hash<std::string_view>{}(p.first) ^ (std::hash<std::string_view>{}(p.second) << 1);
		}
	};

	template<model_config config, tokenizer_types tokenizer_type> struct tokenizer<config, model_arches::llama, tokenizer_type>
		: public tokenizer_parameters<model_arches::llama>, public tokenizer_traits<config.arch, tokenizer_type, config.tokenizer_pre_type> {
		using model_traits_type		= model_traits<config.arch, config.model_size, config.model_generation>;
		using tokenizer_traits_type = tokenizer_traits<config.arch, config.tokenizer_type, config.tokenizer_pre_type>;

		NIHILUS_INLINE tokenizer() noexcept {
			candidate_tokens.reserve(32768);
			cumulative_probs.reserve(32768);
		}

		NIHILUS_INLINE void tokenize_init(int32_t* output_tokens) {
			output_tokens[0] = tokenizer_traits_type::special_bos_id;
			output_tokens[1] = tokenizer_traits_type::special_eos_id;
		}

		NIHILUS_INLINE uint64_t tokenize(char input_text, int32_t* output_tokens) {
			vector<int32_t> temp_tokens;
			if constexpr (tokenizer_traits_type::add_bos && tokenizer_traits_type::special_bos_id > 0) {
				temp_tokens.push_back(static_cast<int32_t>(tokenizer_traits_type::special_bos_id));
			}

			vector<std::string> word_collection = gpt2_style_split(input_text);

			for (const auto& word: word_collection) {
				tokenize_word(word, temp_tokens);
			}

			if constexpr (tokenizer_traits_type::add_eos && tokenizer_traits_type::special_eos_id > 0) {
				temp_tokens.push_back(static_cast<int32_t>(tokenizer_traits_type::special_eos_id));
			}

			for (uint64_t i = 0; i < temp_tokens.size(); ++i) {
				output_tokens[i] = temp_tokens[i];
			}

			if constexpr (config.dev) {
				print_tokenization_debug(input_text, temp_tokens);
			}

			return temp_tokens.size();
		}

		NIHILUS_INLINE uint64_t tokenize(std::string_view input_text, int32_t* output_tokens) {
			vector<int32_t> temp_tokens;
			if constexpr (tokenizer_traits_type::add_bos && tokenizer_traits_type::special_bos_id > 0) {
				temp_tokens.push_back(static_cast<int32_t>(tokenizer_traits_type::special_bos_id));
			}

			vector<std::string> word_collection = gpt2_style_split(input_text);

			for (const auto& word: word_collection) {
				tokenize_word(word, temp_tokens);
			}

			if constexpr (tokenizer_traits_type::add_eos && tokenizer_traits_type::special_eos_id > 0) {
				temp_tokens.push_back(static_cast<int32_t>(tokenizer_traits_type::special_eos_id));
			}

			for (uint64_t i = 0; i < temp_tokens.size(); ++i) {
				output_tokens[i] = temp_tokens[i];
			}

			if constexpr (config.dev) {
				print_tokenization_debug(input_text, temp_tokens);
			}

			return temp_tokens.size();
		}

		struct nihilus_rng {
			uint64_t state{};

			NIHILUS_INLINE nihilus_rng(uint64_t seed = 0) : state(seed == 0 ? static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : seed) {
			}

			NIHILUS_INLINE uint64_t next() {
				state ^= state << 13;
				state ^= state >> 7;
				state ^= state << 17;
				return state;
			}

			NIHILUS_INLINE float next_float() {
				return static_cast<float>(next()) / static_cast<float>(UINT64_MAX);
			}
		};

		struct sampling_params {
			float temperature		 = 1.0f;
			int32_t top_k			 = 50;
			float top_p				 = 0.9f;
			float min_p				 = 0.0f;
			float repetition_penalty = 1.1f;
			uint64_t repeat_last_n	 = 64;
			uint64_t seed			 = 0;
			bool use_mirostat		 = false;
			float mirostat_tau		 = 5.0f;
			float mirostat_eta		 = 0.1f;
		};

		struct token_prob {
			int32_t token_id{};
			float probability{};

			NIHILUS_INLINE bool operator>(const token_prob& other) const {
				return probability > other.probability;
			}
		};

		mutable vector<token_prob> candidate_tokens{};
		mutable vector<float> cumulative_probs{};
		mutable float mirostat_mu{};
		mutable nihilus_rng rng{};

		NIHILUS_INLINE void apply_temperature(float* logits, uint64_t vocab_size, float temperature) const {
			if (temperature == 1.0f)
				return;

			const float inv_temp = 1.0f / temperature;

			for (uint64_t i = 0; i < vocab_size; ++i) {
				logits[i] *= inv_temp;
			}
		}

		NIHILUS_INLINE void apply_repetition_penalty(float* logits, const int32_t* recent_tokens, uint64_t recent_count, float penalty) const {
			if (penalty == 1.0f)
				return;

			for (uint64_t i = 0; i < recent_count; ++i) {
				int32_t token_new = recent_tokens[i];
				if (token_new >= 0 && token_new < static_cast<int32_t>(tokens.size())) {
					if (logits[token_new] > 0.0f) {
						logits[token_new] /= penalty;
					} else {
						logits[token_new] *= penalty;
					}
				}
			}
		}

		NIHILUS_INLINE void logits_to_probs(float* logits, uint64_t vocab_size) const {
			float max_logit = logits[0];
			for (uint64_t i = 1; i < vocab_size; ++i) {
				max_logit = detail::max(max_logit, logits[i]);
			}

			float sum = 0.0f;
			for (uint64_t i = 0; i < vocab_size; ++i) {
				logits[i] = std::exp(logits[i] - max_logit);
				sum += logits[i];
			}

			const float inv_sum = 1.0f / sum;
			for (uint64_t i = 0; i < vocab_size; ++i) {
				logits[i] *= inv_sum;
			}
		}

		NIHILUS_INLINE void apply_top_k(vector<token_prob>& candidates, int32_t k) const {
			if (k <= 0 || k >= static_cast<int32_t>(candidates.size()))
				return;

			std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), [](const token_prob& a, const token_prob& b) {
				return a.probability > b.probability;
			});

			candidates.resize(k);
		}

		NIHILUS_INLINE void apply_top_p(vector<token_prob>& candidates, float p) const {
			if (p >= 1.0f)
				return;

			std::sort(candidates.begin(), candidates.end(), [](const token_prob& a, const token_prob& b) {
				return a.probability > b.probability;
			});

			float cumulative_prob = 0.0f;
			uint64_t cutoff		  = candidates.size();

			for (uint64_t i = 0; i < candidates.size(); ++i) {
				cumulative_prob += candidates[i].probability;
				if (cumulative_prob >= p) {
					cutoff = i + 1;
					break;
				}
			}

			candidates.resize(cutoff);
		}

		NIHILUS_INLINE void apply_min_p(vector<token_prob>& candidates, float min_p) const {
			if (min_p <= 0.0f)
				return;

			candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
								 [min_p](const token_prob& candidate) {
									 return candidate.probability < min_p;
								 }),
				candidates.end());
		}

		NIHILUS_INLINE int32_t sample_from_candidates(const vector<token_prob>& candidates) const {
			if (candidates.empty())
				return 0;
			if (candidates.size() == 1)
				return candidates[0].token_id;

			cumulative_probs.clear();
			cumulative_probs.reserve(candidates.size());

			float cumulative = 0.0f;
			for (const auto& candidate: candidates) {
				cumulative += candidate.probability;
				cumulative_probs.push_back(cumulative);
			}

			const float random_val = rng.next_float() * cumulative;

			auto it		   = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), random_val);
			uint64_t index = std::distance(cumulative_probs.begin(), it);

			return candidates[detail::min(index, candidates.size() - 1)].token_id;
		}

		NIHILUS_INLINE int32_t sample_next_token(float* logits, uint64_t vocab_size, const int32_t* recent_tokens, uint64_t recent_count, const sampling_params& params) const {
			if (params.seed != 0) {
				rng = nihilus_rng(params.seed);
			}

			apply_repetition_penalty(logits, recent_tokens, detail::min(recent_count, params.repeat_last_n), params.repetition_penalty);

			apply_temperature(logits, vocab_size, params.temperature);

			logits_to_probs(logits, vocab_size);

			candidate_tokens.clear();
			candidate_tokens.reserve(vocab_size);

			for (uint64_t i = 0; i < vocab_size; ++i) {
				if (logits[i] > 0.0f) {
					candidate_tokens.push_back({ static_cast<int32_t>(i), logits[i] });
				}
			}

			if (candidate_tokens.empty()) {
				return 0;
			}

			apply_min_p(candidate_tokens, params.min_p);
			apply_top_k(candidate_tokens, params.top_k);
			apply_top_p(candidate_tokens, params.top_p);

			float total_prob = 0.0f;
			for (auto& candidate: candidate_tokens) {
				total_prob += candidate.probability;
			}

			if (total_prob > 0.0f) {
				const float inv_total = 1.0f / total_prob;
				for (auto& candidate: candidate_tokens) {
					candidate.probability *= inv_total;
				}
			}

			return sample_from_candidates(candidate_tokens);
		}

		NIHILUS_INLINE int32_t sample_next_token(float* logits, uint64_t vocab_size) {
			sampling_params default_params{};
			return sample_next_token(logits, vocab_size, nullptr, 0, default_params);
		}

		NIHILUS_INLINE int32_t sample_next_token(float* logits, uint64_t vocab_size, const vector<int32_t>& context, const sampling_params& params = sampling_params{}) {
			return sample_next_token(logits, vocab_size, context.empty() ? nullptr : context.data(), context.size(), params);
		}

	  protected:
		static constexpr std::string_view regex_exprs{ [] {
			if constexpr (tokenizer_traits_type::pre_type == tokenizer_pre_types::llama3) {
				return "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
					   "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
			} else {
				return std::string_view{};
			}
		}() };
		std::unordered_map<std::pair<std::string_view, std::string_view>, int32_t, pair_hash> bpe_ranks{};
		nihilus_bigram_bpe::queue work_queue{};
		vector<nihilus_symbol> symbols{};

		NIHILUS_INLINE vector<std::string> gpt2_style_split(std::string_view text) {
			vector<std::string> result;
			uint64_t start = text.find_first_not_of(" \t\n\r");
			if (start == std::string::npos)
				return result;
			text = text.substr(start);

			uint64_t end = text.find_last_not_of(" \t\n\r");
			if (end != std::string::npos) {
				text = text.substr(0, end + 1);
			}

			bool is_first_word = true;
			uint64_t i		   = 0;

			while (i < text.length()) {
				std::string token_new;
				while (i < text.length() && is_space(text[i])) {
					i++;
				}

				if (i >= text.length())
					break;

				if (is_first_word) {
					if (is_alpha(text[i])) {
						while (i < text.length() && is_alpha(text[i])) {
							token_new += text[i];
							i++;
						}
					} else if (is_digit(text[i])) {
						while (i < text.length() && is_digit(text[i])) {
							token_new += text[i];
							i++;
						}
					} else {
						token_new += text[i];
						i++;
					}
					is_first_word = false;
				} else {
					if (is_alpha(text[i])) {
						token_new += "Ġ";
						while (i < text.length() && is_alpha(text[i])) {
							token_new += text[i];
							i++;
						}
					} else if (is_digit(text[i])) {
						token_new += "Ġ";
						while (i < text.length() && is_digit(text[i])) {
							token_new += text[i];
							i++;
						}
					} else {
						token_new += text[i];
						i++;
					}
				}

				if (!token_new.empty()) {
					result.push_back(token_new);
				}
			}

			return result;
		}

		NIHILUS_INLINE std::string gpt2_style_split(char c) {
			if (!is_space(c)) {
				return std::string{ c, 1 };
			}
			return {};
		}

		NIHILUS_INLINE void tokenize_word(const std::string_view word, vector<int32_t>& output) {
			if (word.empty())
				return;

			auto direct_match = tokens.find(word);
			if (direct_match != tokens.end()) {
				output.push_back(direct_match->second);
				return;
			}

			symbols.clear();
			work_queue = nihilus_bigram_bpe::queue();

			int32_t index	= 0;
			uint64_t offset = 0;

			while (offset < word.size()) {
				nihilus_symbol sym;
				uint64_t char_len = detail::min(word.size() - offset, static_cast<uint64_t>(unicode_len_utf8(word[offset])));
				sym.text		  = word.data() + offset;
				sym.n			  = char_len;
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
				auto it = tokens.find(str);

				if (it != tokens.end()) {
					output.push_back(it->second);
				} else {
					for (char c: str) {
						std::string byte_str(1, c);
						auto byte_it = tokens.find(byte_str);
						if (byte_it != tokens.end()) {
							output.push_back(byte_it->second);
						}
					}
				}
			}
		}

		NIHILUS_INLINE void add_new_bigram(int32_t left, int32_t right) {
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

		NIHILUS_INLINE uint64_t unicode_len_utf8(char c) {
			if ((c & 0xE0) == 0xC0) {
				return 2;
			}
			if ((c & 0xF0) == 0xE0) {
				return 3;
			}
			if ((c & 0xF8) == 0xF0) {
				return 4;
			}
			return 1;
		}

		NIHILUS_INLINE void print_tokenization_debug(std::string_view input_text, const vector<int32_t>& tokens_new) {
			std::cout << "=== NIHILUS BPE TOKENIZATION DEBUG ===" << std::endl;
			std::cout << "system_info: n_threads = " << std::thread::hardware_concurrency() << " | NIHILUS ENGINE | BPE VOCAB | 432% FASTER |" << std::endl;
			//std::cout << "tokenizer_traits_type: " << static_cast<int32_t>(tokenizer_traits_type) << " (BPE)" << std::endl;
			std::cout << "pre_type: " << pre << std::endl;
			std::cout << "Input text: \"" << input_text << "\"" << std::endl;
			std::cout << "Token count: " << tokens_new.size() << std::endl;

			std::cout << "Tokens: ";
			for (uint64_t i = 0; i < tokens_new.size(); ++i) {
				std::cout << "[" << i << "]=" << tokens_new[i];
				if (i < tokens_new.size() - 1)
					std::cout << " ";
			}
			std::cout << std::endl;

			std::cout << "Token strings: ";
			for (uint64_t i = 0; i < tokens_new.size(); ++i) {
				std::cout << "[" << i << "]=" << tokens_new[i];
				if (i < tokens_new.size() - 1)
					std::cout << " ";
			}
			std::cout << std::endl;
			std::cout << "=================================" << std::endl;
		}

		NIHILUS_INLINE ~tokenizer() {
		}
	};


}
