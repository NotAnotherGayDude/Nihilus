// BnchSwt/EventCounter.hpp
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
/// Dec 6, 2024
#pragma once

#include <nihilus-incl/common/config.hpp>
#include <optional>
#include <chrono>

namespace nihilus::benchmarking {

	struct event_count {
		template<typename value_type> friend struct event_collector_type;

		NIHILUS_HOST event_count() noexcept = default;

		NIHILUS_HOST double elapsedNs() const noexcept {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}

		NIHILUS_HOST bool byes_processed(uint64_t& byes_processed_new) const noexcept {
			if (bytes_processed_val.has_value()) {
				byes_processed_new = bytes_processed_val.value();
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_HOST bool cycles(double& cycles_new) const {
			if (cycles_val.has_value()) {
				cycles_new = static_cast<double>(cycles_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_HOST bool instructions(double& instructions_new) const noexcept {
			if (instructions_val.has_value()) {
				instructions_new = static_cast<double>(instructions_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_HOST bool branches(double& branches_new) const noexcept {
			if (branches_val.has_value()) {
				branches_new = static_cast<double>(branches_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_HOST bool branch_misses(double& branch_misses_new) const noexcept {
			if (branch_misses_val.has_value()) {
				branch_misses_new = static_cast<double>(branch_misses_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_HOST bool cache_misses(double& cache_misses_new) const noexcept {
			if (cache_misses_val.has_value()) {
				cache_misses_new = static_cast<double>(cache_misses_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_HOST bool cache_references(double& cache_references_new) const noexcept {
			if (cache_references_val.has_value()) {
				cache_references_new = static_cast<double>(cache_references_val.value());
				return true;
			} else {
				return false;
			}
		}

	  protected:
		std::optional<uint64_t> cache_references_val{};
		std::optional<uint64_t> bytes_processed_val{};
		std::optional<uint64_t> branch_misses_val{};
		std::optional<uint64_t> instructions_val{};
		std::optional<uint64_t> cache_misses_val{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> branches_val{};
		std::optional<uint64_t> cycles_val{};
	};

}
