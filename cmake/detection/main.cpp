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
#if defined(NIHILUS_DETECT_CPU_ARCH)
	#include <cstring>
	#include <cstdint>
	#include <cstdlib>
	#if defined(_MSC_VER)
		#include <intrin.h>
	#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
		#include <cpuid.h>
	#endif

enum class instruction_set {
	FALLBACK = 0x0,
	AVX2	 = 0x1,
	AVX512f	 = 0x2,
	NEON	 = 0x4,
	SVE2	 = 0x8,
};

	#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
		#if defined(__linux__)
			#include <sys/auxv.h>
			#include <asm/hwcap.h>
		#elif defined(__APPLE__)
			#include <sys/sysctl.h>
		#endif

inline static uint32_t detect_supported_architectures() {
	uint32_t host_isa = static_cast<uint32_t>(instruction_set::NEON);

		#if defined(__linux__)
	unsigned long hwcap = getauxval(AT_HWCAP);
	if (hwcap & HWCAP_SVE) {
		host_isa |= static_cast<uint32_t>(instruction_set::SVE2);
	}
		#endif

	return host_isa;
}

	#elif defined(__x86_64__) || defined(_M_AMD64)
static constexpr uint32_t cpuid_avx2_bit	 = 1ul << 5;
static constexpr uint32_t cpuid_avx512_bit	 = 1ul << 16;
static constexpr uint64_t cpuid_avx256_saved = 1ull << 2;
static constexpr uint64_t cpuid_avx512_saved = 7ull << 5;
static constexpr uint32_t cpuid_osx_save	 = (1ul << 26) | (1ul << 27);

inline static void cpuid(uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
		#if defined(_MSC_VER)
	int32_t cpu_info[4];
	__cpuidex(cpu_info, *eax, *ecx);
	*eax = cpu_info[0];
	*ebx = cpu_info[1];
	*ecx = cpu_info[2];
	*edx = cpu_info[3];
		#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
	uint32_t level = *eax;
	__get_cpuid(level, eax, ebx, ecx, edx);
		#else
	uint32_t a = *eax, b, c = *ecx, d;
	asm volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(a), "c"(c));
	*eax = a;
	*ebx = b;
	*ecx = c;
	*edx = d;
		#endif
}

inline static uint64_t xgetbv() {
		#if defined(_MSC_VER)
	return _xgetbv(0);
		#else
	uint32_t eax, edx;
	asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return (( uint64_t )edx << 32) | eax;
		#endif
}

inline static uint32_t detect_supported_architectures() {
	std::uint32_t eax	   = 0;
	std::uint32_t ebx	   = 0;
	std::uint32_t ecx	   = 0;
	std::uint32_t edx	   = 0;
	std::uint32_t host_isa = static_cast<uint32_t>(instruction_set::FALLBACK);

	eax = 0x1;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if ((ecx & cpuid_osx_save) != cpuid_osx_save) {
		return host_isa;
	}

	uint64_t xcr0 = xgetbv();
	if ((xcr0 & cpuid_avx256_saved) == 0) {
		return host_isa;
	}

	eax = 0x7;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if (ebx & cpuid_avx2_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX2);
	}

	if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
		return host_isa;
	}

	if (ebx & cpuid_avx512_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX512f);
	}

	return host_isa;
}

	#else
inline static uint32_t detect_supported_architectures() {
	return static_cast<uint32_t>(instruction_set::FALLBACK);
}
	#endif

int32_t main() {
	const auto supported_isa = detect_supported_architectures();
	return supported_isa;
}
#elif defined(NIHILUS_DETECT_GPU_ARCH)
	#include <cuda_runtime.h>
	#include <iostream>
enum class instruction_set {
	cuda_9	= 0x1,
	cuda_10 = 0x2,
	cuda_11 = 0x4,
	cuda_12 = 0x8,
};

int32_t main() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	if (deviceProp.major == 9) {
		return static_cast<int32_t>(instruction_set::cuda_9);
	} else if (deviceProp.major == 10) {
		return static_cast<int32_t>(instruction_set::cuda_10);
	} else if (deviceProp.major == 11) {
		return static_cast<int32_t>(instruction_set::cuda_11);
	} else if (deviceProp.major == 12) {
		return static_cast<int32_t>(instruction_set::cuda_12);
	}
}
#elif defined(NIHILUS_DETECT_GPU_CACHE_SIZE)
	#include <cuda_runtime.h>
	#include <iostream>

uint32_t get_gpu_l2_cache_size() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	return deviceProp.l2CacheSize;
}

int32_t main() {
	return get_gpu_l2_cache_size();
}
#elif defined(NIHILUS_DETECT_CPU_CACHE_SIZE)
	#include <iostream>
	#include <vector>
	#if NIHILUS_PLATFORM_WINDOWS
		#include <Windows.h>
	#endif
	#if NIHILUS_PLATFORM_LINUX || NIHILUS_PLATFORM_ANDROID
		#include <fstream>
		#include <string>
	#endif
	#if NIHILUS_PLATFORM_MAC
		#include <sys/sysctl.h>
		#include <sys/types.h>
		#include <string>
	#endif

enum class cache_level {
	one	  = 1,
	two	  = 2,
	three = 3,
};

inline size_t get_cache_size(cache_level level) {
	#if NIHILUS_PLATFORM_WINDOWS
	DWORD bufferSize = 0;
	std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};
	GetLogicalProcessorInformation(nullptr, &bufferSize);
	buffer.resize(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

	if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
		std::cerr << "Failed to retrieve processor information!" << std::endl;
		return 0;
	}

	const auto infoCount = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
	for (size_t i = 0; i < infoCount; ++i) {
		if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == static_cast<int32_t>(level)) {
			if (level == cache_level::one && buffer[i].Cache.Type == CacheData) {
				return buffer[i].Cache.Size;
			}
			else if (level != cache_level::one && buffer[i].Cache.Type == CacheUnified) {
				return buffer[i].Cache.Size;
			}
		}
	}
	return 0;

	#elif NIHILUS_PLATFORM_LINUX || NIHILUS_PLATFORM_ANDROID
	auto get_cache_size_from_file = [](const std::string& index) {
		const std::string cacheFilePath = "/sys/devices/system/cpu/cpu0/cache/index" + index + "/size";
		std::ifstream file(cacheFilePath);
		if (!file.is_open()) {
			std::cerr << "Failed to open cache info file: " << cacheFilePath << std::endl;
			return static_cast<size_t>(0);
		}

		std::string sizeStr;
		file >> sizeStr;
		file.close();

		size_t size = 0;
		if (sizeStr.back() == 'K') {
			size = std::stoul(sizeStr) * 1024;
		} else if (sizeStr.back() == 'M') {
			size = std::stoul(sizeStr) * 1024 * 1024;
		} else {
			size = std::stoul(sizeStr);
		}
		return size;
	};

	if (level == cache_level::one) {
		return get_cache_size_from_file("0");
	} else {
		std::string index = (level == cache_level::two) ? "2" : "3";
		return get_cache_size_from_file(index);
	}

	#elif NIHILUS_PLATFORM_MAC
	auto get_cache_size = [](const std::string& cacheType) {
		size_t cacheSize		= 0;
		size_t size				= sizeof(cacheSize);
		std::string sysctlQuery = "hw." + cacheType + "cachesize";
		if (sysctlbyname(sysctlQuery.c_str(), &cacheSize, &size, nullptr, 0) != 0) {
			return size_t{ 0 };
		}
		return cacheSize;
	};

	if (level == cache_level::one) {
		return get_cache_size("l1d");
	} else if (level == cache_level::two) {
		return get_cache_size("l2");
	} else {
		return get_cache_size("l3");
	}
	#endif

	return 0;
}

int32_t main() {
	return get_cache_size(cache_level::one);
}
#else
	#include <thread>
int32_t main() {
	return std::thread::hardware_concurrency();
}
#endif