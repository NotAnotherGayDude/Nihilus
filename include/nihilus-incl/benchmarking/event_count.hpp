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
			if (bytesProcessedVal.has_value()) {
				bytesProcessedNew = bytesProcessedVal.value();
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool cycles(double& cyclesNew) const {
			if (cyclesVal.has_value()) {
				cyclesNew = static_cast<double>(cyclesVal.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool instructions(double& instructionsNew) const noexcept {
			if (instructionsVal.has_value()) {
				instructionsNew = static_cast<double>(instructionsVal.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool branches(double& branchesNew) const noexcept {
			if (branchesVal.has_value()) {
				branchesNew = static_cast<double>(branchesVal.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool branchMisses(double& branchMissesNew) const noexcept {
			if (branchMissesVal.has_value()) {
				branchMissesNew = static_cast<double>(branchMissesVal.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool cacheMisses(double& cacheMissesNew) const noexcept {
			if (cacheMissesVal.has_value()) {
				cacheMissesNew = static_cast<double>(cacheMissesVal.value());
				return true;
			} else {
				return false;
			}
		}

		NIHILUS_INLINE bool cacheReferences(double& cacheReferencesNew) const noexcept {
			if (cacheReferencesVal.has_value()) {
				cacheReferencesNew = static_cast<double>(cacheReferencesVal.value());
				return true;
			} else {
				return false;
			}
		}

	  protected:
		std::optional<uint64_t> cacheReferencesVal{};
		std::optional<uint64_t> bytesProcessedVal{};
		std::optional<uint64_t> branchMissesVal{};
		std::optional<uint64_t> instructionsVal{};
		std::optional<uint64_t> cacheMissesVal{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> branchesVal{};
		std::optional<uint64_t> cyclesVal{};
	};

}
