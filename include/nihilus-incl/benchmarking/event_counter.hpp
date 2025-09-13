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
#include <nihilus-incl/benchmarking/apple_arm_pref_events.hpp>
#include <nihilus-incl/benchmarking/windows_pref_events.hpp>
#include <nihilus-incl/benchmarking/linux_pref_events.hpp>
#include <optional>
#include <chrono>

namespace nihilus::benchmarking {	

	using event_collector = event_collector_type<event_count>;

}
