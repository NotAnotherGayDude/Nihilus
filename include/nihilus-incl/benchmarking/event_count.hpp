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

namespace nihilus::benchmarking::internal {

	struct event_count {
		template<typename value_type> friend struct event_collector_type;

		NIHILUS_INLINE event_count() noexcept = default;

		NIHILUS_INLINE double elapsedNs() const noexcept {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}

		NIHILUS_INLINE bool bytesProcessed(uint64_t& bytesProcessedNew) const noexcept {
			if (bytes_processed_val.has_value()) {
				bytesProcessedNew = bytes_processed_val.value();
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool cycles(double& cyclesNew) const {
			if (cycles_val.has_value()) {
				cyclesNew = static_cast<double>(cycles_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool instructions(double& instructionsNew) const noexcept {
			if (instructions_val.has_value()) {
				instructionsNew = static_cast<double>(instructions_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool branches(double& branchesNew) const noexcept {
			if (branches_val.has_value()) {
				branchesNew = static_cast<double>(branches_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool branch_misses(double& branch_missesNew) const noexcept {
			if (branch_misses_val.has_value()) {
				branch_missesNew = static_cast<double>(branch_misses_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool cache_misses(double& cache_missesNew) const noexcept {
			if (cache_misses_val.has_value()) {
				cache_missesNew = static_cast<double>(cache_misses_val.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool cache_references(double& cache_referencesNew) const noexcept {
			if (cache_references_val.has_value()) {
				cache_referencesNew = static_cast<double>(cache_references_val.value());
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
