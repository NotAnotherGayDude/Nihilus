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

#include <nihilus-incl/infra/tokenizer_traits.hpp>
#include <nihilus-incl/infra/model_traits.hpp>
#include <nihilus-incl/cpu/memory_buffer.hpp>
#include <nihilus-incl/common/rt_string.hpp>
#include <unordered_map>
#include <iterator>
#include <queue>

namespace nihilus {

	template<model_arches model_arch> struct tokenizer_parameters;

	template<> struct tokenizer_parameters<model_arches::llama> {
		std::unordered_map<std::string_view, token> tokens{};
		aligned_vector<std::string_view> merges{};
		aligned_vector<int32_t> token_types{};
		std::string_view chat_template{};
		std::string_view pre{};
	};

	template<typename config_type, model_arches model_arch, tokenizer_types> struct tokenizer;

	struct nihilus_symbol {
		const char* text{};
		int32_t prev{};
		int32_t next{};
		uint64_t n{};
	};

	template<typename stored_type, typename comparator> struct priority_queue : comparator {
		array<stored_type, 200> data{};
		size_t currently_stored_size{};

		NIHILUS_HOST bool empty() const {
			return currently_stored_size == 0;
		}

		NIHILUS_HOST void clear() {
			currently_stored_size = 0;
		}

		NIHILUS_HOST const stored_type& top() const {
			return data[0];
		}

		template<typename stored_type_new> NIHILUS_HOST void push(stored_type_new&& value) {
			data[currently_stored_size] = std::forward<stored_type_new>(value);
			bubble_up(currently_stored_size);
			++currently_stored_size;
		}

		NIHILUS_HOST void pop() {
			if (currently_stored_size > 0) {
				data[0] = data[currently_stored_size - 1];
				--currently_stored_size;
				if (currently_stored_size > 0) {
					bubble_down(0);
				}
			}
		}

	  protected:
		NIHILUS_HOST void bubble_up(size_t index) {
			while (index > 0) {
				size_t parent = (index - 1) / 2;
				if (!comparator::operator()(data[index], data[parent]))
					break;
				std::swap(data[index], data[parent]);
				index = parent;
			}
		}

		NIHILUS_HOST void bubble_down(size_t index) {
			while (true) {
				size_t left	   = 2 * index + 1;
				size_t right   = 2 * index + 2;
				size_t largest = index;

				if (left < currently_stored_size && comparator::operator()(data[left], data[largest])) {
					largest = left;
				}
				if (right < currently_stored_size && comparator::operator()(data[right], data[largest])) {
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
			NIHILUS_HOST_DEVICE bool operator()(const nihilus_bigram_bpe& l, const nihilus_bigram_bpe& r) const {
				return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
			}
		};

		using queue = priority_queue<nihilus_bigram_bpe, comparator>;

		rt_string text{};
		uint64_t size{};
		int32_t right{};
		int32_t left{};
		int32_t rank{};
	};

	struct pair_hash : std::hash<std::string_view> {
		NIHILUS_HOST uint64_t operator()(const std::pair<const std::string_view, const std::string_view>& p) const noexcept {
			return std::hash<std::string_view>::operator()(p.first) ^ (std::hash<std::string_view>::operator()(p.second) << 1);
		}
	};

	template<typename config_type, tokenizer_types tokenizer_type> struct tokenizer<config_type, model_arches::llama, tokenizer_type>
		: public tokenizer_parameters<model_arches::llama>, public tokenizer_traits<tokenizer_type, config_type::tokenizer_pre_type> {
		using model_traits_type		= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;
		using tokenizer_traits_type = tokenizer_traits<config_type::tokenizer_type, config_type::tokenizer_pre_type>;

		NIHILUS_HOST tokenizer() noexcept {
			candidate_tokens.reserve(32768);
			cumulative_probs.reserve(32768);
		}

		NIHILUS_HOST void init_rng(uint64_t seed) {
			auto x	   = seed >> 12ull;
			auto x01   = x ^ x << 25ull;
			auto x02   = x01 ^ x01 >> 27ull;
			uint64_t s = x02 * 0x2545F4914F6CDD1Dull;
			for (uint64_t y = 0; y < 4; ++y) {
				state[y] = splitmix64(s);
			}
		}

		NIHILUS_HOST void tokenizer_init(int32_t* output_tokens) {
			memory_transfer<config_type>::host_to_device(tokenizer_traits_type::special_bos_id, output_tokens);
			memory_transfer<config_type>::host_to_device(tokenizer_traits_type::special_eos_id, output_tokens + 1);
		}

		NIHILUS_HOST uint64_t tokenize(rt_string& input_text, int32_t* output_tokens) {
			aligned_vector<int32_t> temp_tokens;
			if constexpr (tokenizer_traits_type::add_bos && tokenizer_traits_type::special_bos_id > 0) {
				temp_tokens.emplace_back(static_cast<int32_t>(tokenizer_traits_type::special_bos_id));
			}

			gpt2_style_split(input_text);

			for (const auto& word: result) {
				tokenize_word(word, temp_tokens);
			}

			if constexpr (tokenizer_traits_type::add_eos && tokenizer_traits_type::special_eos_id > 0) {
				temp_tokens.emplace_back(static_cast<int32_t>(tokenizer_traits_type::special_eos_id));
			}

			for (uint64_t i = 0; i < temp_tokens.size(); ++i) {
				memory_transfer<config_type>::host_to_device(temp_tokens[i], output_tokens + i);
			}

			if constexpr (config_type::dev) {
				print_tokenization_debug(input_text, temp_tokens);
			}

			return temp_tokens.size();
		}

		NIHILUS_HOST uint64_t tokenize(const char* string, uint64_t size, int32_t* output_tokens) {
			aligned_vector<int32_t> temp_tokens;
			if constexpr (tokenizer_traits_type::add_bos && tokenizer_traits_type::special_bos_id > 0) {
				temp_tokens.emplace_back(static_cast<int32_t>(tokenizer_traits_type::special_bos_id));
			}

			gpt2_style_split({ string, size });

			for (const auto& word: result) {
				tokenize_word(word, temp_tokens);
			}

			if constexpr (tokenizer_traits_type::add_eos && tokenizer_traits_type::special_eos_id > 0) {
				temp_tokens.emplace_back(static_cast<int32_t>(tokenizer_traits_type::special_eos_id));
			}

			for (uint64_t i = 0; i < temp_tokens.size(); ++i) {
				memory_transfer<config_type>::host_to_device(temp_tokens[i], output_tokens + i);
			}

			if constexpr (config_type::dev) {
				print_tokenization_debug({ string, size }, temp_tokens);
			}

			return temp_tokens.size();
		}

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

			NIHILUS_HOST bool operator>(const token_prob& other) const {
				return probability > other.probability;
			}
		};

		NIHILUS_HOST void apply_temperature(float* logits, uint64_t vocab_size, float temperature) {
			if (temperature == 1.0f)
				return;

			const float inv_temp = 1.0f / temperature;

			for (uint64_t i = 0; i < vocab_size; ++i) {
				logits[i] *= inv_temp;
			}
		}

		NIHILUS_HOST void apply_repetition_penalty(float* logits, const int32_t* recent_tokens, uint64_t recent_count, float penalty) {
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

		NIHILUS_HOST void logits_to_probs(float* logits, uint64_t vocab_size) {
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

		NIHILUS_HOST void apply_top_k(aligned_vector<token_prob>& candidates, int32_t k) {
			if (k <= 0 || k >= static_cast<int32_t>(candidates.size()))
				return;

			std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), [](const token_prob& a, const token_prob& b) {
				return a.probability > b.probability;
			});

			candidates.resize(k);
		}

		NIHILUS_HOST void apply_top_p(aligned_vector<token_prob>& candidates, float p) {
			if (p >= 1.0f)
				return;

			sort<sort_methods::greater_than, token_prob>::impl(candidates.data(), candidates.size());

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

		NIHILUS_HOST void apply_min_p(aligned_vector<token_prob>& candidates, float min_p) {
			if (min_p <= 0.0f)
				return;

			candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
								 [min_p](const token_prob& candidate) {
									 return candidate.probability < min_p;
								 }),
				candidates.end());
		}

		NIHILUS_HOST int32_t sample_from_candidates(const aligned_vector<token_prob>& candidates) {
			if (candidates.empty())
				return 0;
			if (candidates.size() == 1)
				return candidates[0].token_id;

			cumulative_probs.clear();
			cumulative_probs.reserve(candidates.size());

			float cumulative = 0.0f;
			for (const auto& candidate: candidates) {
				cumulative += candidate.probability;
				cumulative_probs.emplace_back(cumulative);
			}

			const float random_val = next_float() * cumulative;

			auto it		   = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), random_val);
			uint64_t index = std::distance(cumulative_probs.begin(), it);

			return candidates[detail::min(index, candidates.size() - 1)].token_id;
		}

		NIHILUS_HOST int32_t sample_next_token(float* logits, uint64_t vocab_size, const int32_t* recent_tokens, uint64_t recent_count, const sampling_params& params) const {
			apply_repetition_penalty(logits, recent_tokens, detail::min(recent_count, params.repeat_last_n), params.repetition_penalty);

			apply_temperature(logits, vocab_size, params.temperature);

			logits_to_probs(logits, vocab_size);

			candidate_tokens.clear();
			candidate_tokens.reserve(vocab_size);

			for (uint64_t i = 0; i < vocab_size; ++i) {
				if (logits[i] > 0.0f) {
					candidate_tokens.emplace_back({ static_cast<int32_t>(i), logits[i] });
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

		NIHILUS_HOST int32_t sample_next_token(float* logits, uint64_t vocab_size) {
			sampling_params default_params{};
			return sample_next_token(logits, vocab_size, nullptr, 0, default_params);
		}

		NIHILUS_HOST int32_t sample_next_token(float* logits, uint64_t vocab_size, const aligned_vector<int32_t>& context, const sampling_params& params = sampling_params{}) {
			return sample_next_token(logits, vocab_size, context.empty() ? nullptr : context.data(), context.size(), params);
		}

	  protected:
		static constexpr const std::string_view regex_exprs{ [] {
			if constexpr (tokenizer_traits_type::tokenizer_pre_type == tokenizer_pre_types::llama3) {
				return "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
					   "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
			} else {
				return std::string_view{};
			}
		}() };
		std::unordered_map<std::pair<const std::string_view, const std::string_view>, int32_t, pair_hash> bpe_ranks{};
		aligned_vector<token_prob> candidate_tokens{};
		aligned_vector<float> cumulative_probs{};
		aligned_vector<nihilus_symbol> symbols{};
		nihilus_bigram_bpe::queue work_queue{};
		aligned_vector<rt_string> result{};
		rt_string right_token{};
		rt_string left_token{};
		rt_string byte_str{};
		float mirostat_mu{};
		uint64_t state[4]{};
		rt_string str{};

		NIHILUS_HOST uint64_t next() noexcept {
			const uint64_t result_new = rotl(state[1ull] * 5ull, 7ull) * 9ull;
			const uint64_t t		  = state[1ull] << 17ull;

			state[2ull] ^= state[0ull];
			state[3ull] ^= state[1ull];
			state[1ull] ^= state[2ull];
			state[0ull] ^= state[3ull];

			state[2ull] ^= t;

			state[3ull] = rotl(state[3ull], 45ull);

			return result_new;
		}

		NIHILUS_HOST float next_float() {
			return static_cast<float>(next()) / static_cast<float>(std::numeric_limits<uint64_t>::max());
		}

		NIHILUS_HOST uint64_t rotl(const uint64_t x, const uint64_t k) const noexcept {
			return (x << k) | (x >> (64ull - k));
		}

		NIHILUS_HOST uint64_t splitmix64(uint64_t& seed64) const noexcept {
			uint64_t result_new = seed64 += 0x9E3779B97F4A7C15ull;
			result_new			= (result_new ^ (result_new >> 30ull)) * 0xBF58476D1CE4E5B9ull;
			result_new			= (result_new ^ (result_new >> 27ull)) * 0x94D049BB133111EBull;
			return result_new ^ (result_new >> 31ull);
		}

		NIHILUS_HOST void gpt2_style_split(rt_string_view text) {
			uint64_t start = text.find_first_non_whitespace();
			result.clear();
			if (start == std::string::npos) {
				return;
			}
			text = text.substr(start);

			uint64_t end = text.find_last_non_whitespace();
			if (end != std::string::npos) {
				text = text.substr(0, end + 1);
			}

			bool is_first_word = true;
			uint64_t i		   = 0;

			while (i < text.size()) {
				rt_string token_new;
				while (i < text.size() && is_space(text[i])) {
					i++;
				}

				if (i >= text.size())
					break;

				if (is_first_word) {
					if (is_alpha(text[i])) {
						while (i < text.size() && is_alpha(text[i])) {
							token_new += text[i];
							i++;
						}
					} else if (is_digit(text[i])) {
						while (i < text.size() && is_digit(text[i])) {
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
						while (i < text.size() && is_alpha(text[i])) {
							token_new += text[i];
							i++;
						}
					} else if (is_digit(text[i])) {
						token_new += "Ġ";
						while (i < text.size() && is_digit(text[i])) {
							token_new += text[i];
							i++;
						}
					} else {
						token_new += text[i];
						i++;
					}
				}

				if (!token_new.empty()) {
					result.emplace_back(token_new);
				}
			}

			return;
		}

		NIHILUS_HOST void tokenize_word(const rt_string& word, aligned_vector<int32_t>& output) {
			if (word.empty())
				return;

			auto direct_match = tokens.find({ word.data(), word.size() });
			if (direct_match != tokens.end()) {
				output.emplace_back(direct_match->second);
				return;
			}

			symbols.clear();
			work_queue.clear();

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
				auto& bigram = work_queue.top();
				work_queue.pop();

				auto& left_symbol  = symbols[static_cast<uint64_t>(bigram.left)];
				auto& right_symbol = symbols[static_cast<uint64_t>(bigram.right)];

				if (left_symbol.n == 0 || right_symbol.n == 0) {
					continue;
				}

				left_token.set_values(left_symbol.text, left_symbol.n);
				right_token.set_values(right_symbol.text, right_symbol.n);

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

				str.set_values(symbol.text, symbol.n);
				auto it = tokens.find({ str.data(), str.size() });

				if (it != tokens.end()) {
					output.emplace_back(it->second);
				} else {
					for (char c: str) {
						byte_str	 = c;
						auto byte_it = tokens.find({ byte_str.data(), byte_str.size() });
						if (byte_it != tokens.end()) {
							output.emplace_back(byte_it->second);
						}
					}
				}
			}
		}

		NIHILUS_HOST void add_new_bigram(int32_t left, int32_t right) {
			if (left == -1 || right == -1)
				return;

			left_token.set_values(symbols[static_cast<uint64_t>(left)].text, symbols[static_cast<uint64_t>(left)].n);
			right_token.set_values(symbols[static_cast<uint64_t>(right)].text, symbols[static_cast<uint64_t>(right)].n);

			auto it = bpe_ranks.find({ left_token, right_token });
			if (it == bpe_ranks.end())
				return;

			nihilus_bigram_bpe bigram;
			bigram.left	 = left;
			bigram.right = right;
			bigram.text	 = left_token + right_token;
			bigram.size	 = left_token.size() + right_token.size();
			bigram.rank	 = it->second;

			work_queue.push(bigram);
		}

		NIHILUS_HOST uint64_t unicode_len_utf8(char c) {
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

		NIHILUS_HOST void print_tokenization_debug(rt_string_view input_text, const aligned_vector<int32_t>& tokens_new) {
			std::cout << "=== NIHILUS BPE TOKENIZATION DEBUG ===" << std::endl;
			//std::cout << "tokenizer_traits_type: " << static_cast<int32_t>(tokenizer_traits_type) << " (BPE)" << std::endl;
			std::cout << "pre_type: " << pre << std::endl;
			std::cout << "Input text: \"" << input_text.operator std::string_view() << "\"" << std::endl;
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
	};

}
