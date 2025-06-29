#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <BnchSwt/BenchmarkSuite.hpp>
#include <nihilus/index.hpp>

template<auto valueNew> struct make_static {
	static constexpr auto value{ valueNew };
};

template<typename value_type> extern const value_type external;

#if defined(NIHILUS_COMPILER_CLANG)
constexpr auto pretty_function_tail = "]";
#elif defined(NIHILUS_COMPILER_GNUCXX)
constexpr auto pretty_function_tail = ";";
#endif

template<typename member_type> struct remove_member_pointer {
	using type = member_type;
};

template<typename value_type, typename member_type> struct remove_member_pointer<member_type value_type::*> {
	using type = value_type;
};

template<typename value_type, typename member_type, typename... arg_types> struct remove_member_pointer<member_type (value_type::*)(arg_types...)> {
	using type = value_type;
};

template<typename value_type> using remove_member_pointer_t = typename remove_member_pointer<value_type>::type;

template<typename member_type> struct remove_class_pointer {
	using type = member_type;
};

template<typename class_type, typename member_type> struct remove_class_pointer<member_type class_type::*> {
	using type = member_type;
};

template<typename class_type, typename member_type, typename... arg_types> struct remove_class_pointer<member_type (class_type::*)(arg_types...)> {
	using type = member_type;
};

template<typename value_type> using remove_class_pointer_t = typename remove_class_pointer<value_type>::type;

template<size_t N> constexpr auto stringLiteralFromView(std::string_view str) noexcept {
	nihilus::string_literal<N + 1> sl{};
	std::copy_n(str.data(), str.size(), sl.values);
	sl[N] = '\0';
	return sl;
}

#if defined(NIHILUS_COMPILER_MSVC) && !defined(NIHILUS_COMPILER_CLANG)
template<typename value_type, auto p> consteval std::string_view getNameImpl() noexcept {
	std::string_view str = std::source_location::current().function_name();
	str					 = str.substr(str.find("->") + 2);
	return str.substr(0, str.find(">"));
}
#else
template<auto p> consteval std::string_view getNameImpl() noexcept {
	std::string_view str = std::source_location::current().function_name();
	str					 = str.substr(str.find("&") + 1);
	str					 = str.substr(0, str.find(pretty_function_tail));
	return str.substr(str.rfind("::") + 2);
}
#endif

template<auto p>
	requires(std::is_member_pointer_v<decltype(p)>)
constexpr auto getName() noexcept {
#if defined(NIHILUS_COMPILER_MSVC) && !defined(NIHILUS_COMPILER_CLANG)
	using value_type		 = remove_member_pointer_t<decltype(p)>;
	constexpr auto pNew		 = p;
	constexpr auto newString = getNameImpl<value_type, &(external<value_type>.*pNew)>();
#else
	constexpr auto newString = getNameImpl<p>();
#endif
	return make_static<stringLiteralFromView<newString.size()>(newString)>::value.operator std::string_view();
}

template<typename value_type> struct base_parse_entity {
	using member_type = value_type;
	inline static constexpr size_t index{ 0 };
};

template<auto memberPtrNew, nihilus::string_literal nameNew> struct parse_entity_temp {
	using member_type = remove_class_pointer_t<decltype(memberPtrNew)>;
	using class_type  = remove_member_pointer_t<decltype(memberPtrNew)>;
	inline static constexpr member_type class_type::* memberPtr{ memberPtrNew };
	inline static constexpr nihilus::string_literal name{ nameNew };
};

template<auto memberPtrNew, nihilus::string_literal nameNew, size_t indexNew, size_t maxIndex> struct parse_entity {
	using member_type = remove_class_pointer_t<decltype(memberPtrNew)>;
	using class_type  = remove_member_pointer_t<decltype(memberPtrNew)>;
	inline static constexpr member_type class_type::* memberPtr{ memberPtrNew };
	inline static constexpr bool isItLast{ indexNew == maxIndex - 1 };
	inline static constexpr nihilus::string_literal name{ nameNew };
	inline static constexpr size_t index{ indexNew };
};

template<typename value_type>
concept is_base_parse_entity = requires { typename value_type::member_type; } && !std::is_member_pointer_v<value_type>;

template<typename value_type>
concept is_parse_entity_temp = requires {
	typename value_type::class_type;
	value_type::memberPtr;
} && is_base_parse_entity<value_type>;

template<size_t maxIndex, size_t index, auto value> inline static constexpr auto makeJsonEntityAuto() noexcept {
	if constexpr (is_parse_entity_temp<decltype(value)>) {
		constexpr parse_entity<value.memberPtr, value.name, index, maxIndex> parseEntity{};
		return parseEntity;
	} else {
		constexpr auto nameNew = getName<value>();
		constexpr parse_entity<value, stringLiteralFromView<nameNew.size()>(nameNew), index, maxIndex> parseEntity{};
		return parseEntity;
	}
}

struct test_struct {
	int32_t test_val{};
};

template<typename value_type>
concept convertible_to_parse_entity = is_parse_entity_temp<value_type> || std::is_member_pointer_v<value_type>;

template<auto... values, size_t... indices> inline static constexpr auto createValueImpl(std::index_sequence<indices...>) {
	static_assert((convertible_to_parse_entity<decltype(values)> && ...), "All arguments passed to createValue must be convertible to a parse_entity.");
	return nihilus::makeTuple(makeJsonEntityAuto<sizeof...(values), indices, values>()...);
}

template<auto memberPtr, nihilus::string_literal nameNew> inline static constexpr auto makeJsonEntity() {
	return parse_entity_temp<memberPtr, nameNew>{};
}

template<auto memberPtr> inline static constexpr auto makeJsonEntity() {
	return parse_entity_temp<memberPtr, getName<memberPtr>()>{};
}

template<auto... values> inline static constexpr auto createValue() noexcept {
	return createValueImpl<values...>(std::make_index_sequence<sizeof...(values)>{});
}

struct tuple_reference {
	uint8_t oldIndex{};
	std::string_view key{};
};

struct tuple_references {
	const tuple_reference* rootPtr{};
	size_t count{};
};

template<typename value_type>
concept has_name = requires(std::remove_cvref_t<value_type> value) { value.name; };

template<size_t maxIndex, size_t currentIndex = 0, typename tuple_type>
static constexpr auto collectTupleRefsImpl(const tuple_type& tuple, std::array<tuple_reference, maxIndex>& tupleRefsRaw) {
	if constexpr (currentIndex < maxIndex) {
		auto potentialKey = nihilus::get<currentIndex>(tuple);
		if constexpr (has_name<decltype(potentialKey)>) {
			tupleRefsRaw[currentIndex].key = potentialKey.name.operator std::string_view();
		}
		tupleRefsRaw[currentIndex].oldIndex = currentIndex;
		return collectTupleRefsImpl<maxIndex, currentIndex + 1>(tuple, tupleRefsRaw);
	}
	return tupleRefsRaw;
}

template<typename tuple_type> static constexpr auto collectTupleRefs(const tuple_type& tuple) {
	constexpr auto tupleSize = nihilus::tuple_size_v<tuple_type>;
	std::array<tuple_reference, tupleSize> tupleRefsRaw{};
	return collectTupleRefsImpl<tupleSize>(tuple, tupleRefsRaw);
}

template<size_t size> static constexpr auto consolidateTupleRefs(const std::array<tuple_reference, size>& tupleRefsRaw) {
	tuple_references returnValues{};
	if constexpr (size > 0) {
		returnValues.rootPtr = &tupleRefsRaw[0];
		returnValues.count	 = size;
	}
	return returnValues;
}

template<typename value_type> struct core;

template<typename value_type> inline constexpr auto tupleRefs{ collectTupleRefs(core<value_type>::parse_value) };
template<typename value_type> inline constexpr auto tupleReferences{ consolidateTupleRefs(tupleRefs<value_type>) };

struct key_stats_t {
	size_t minLength{ (std::numeric_limits<size_t>::max)() };
	size_t uniqueIndex{};
	size_t maxLength{};
};

inline static constexpr size_t findUniqueColumnIndex(const tuple_references& tupleRefsRaw, size_t maxIndex, size_t startingIndex = 0) noexcept {
	constexpr size_t alphabetSize = 256;
	std::string_view key{};
	for (size_t index = startingIndex; index < maxIndex; ++index) {
		std::array<bool, alphabetSize> seen{};
		bool allDifferent = true;

		for (size_t x = 0; x < tupleRefsRaw.count; ++x) {
			key				  = tupleRefsRaw.rootPtr[x].key;
			const char c	  = key[index];
			uint8_t charIndex = static_cast<uint8_t>(c);

			if (seen[charIndex]) {
				allDifferent = false;
				break;
			}
			seen[charIndex] = true;
		}

		if (allDifferent) {
			return index;
		}
	}

	return std::numeric_limits<size_t>::max();
}

inline static constexpr auto keyStatsImpl(const tuple_references& tupleRefsRaw) noexcept {
	key_stats_t stats{};
	for (size_t x = 0; x < tupleRefsRaw.count; ++x) {
		const std::string_view& key{ tupleRefsRaw.rootPtr[x].key };
		auto num{ key.size() };
		if (num > stats.maxLength) {
			stats.maxLength = num;
		}
		if (num < stats.minLength) {
			stats.minLength = num;
		}
	}
	stats.uniqueIndex = findUniqueColumnIndex(tupleRefsRaw, stats.minLength);
	return stats;
}

template<typename value_type> inline static constexpr auto keyStatsVal = keyStatsImpl(tupleReferences<value_type>);

template<> struct core<test_struct> {
	using value_type				  = test_struct;
	static constexpr auto parse_value = createValue<&test_struct::test_val>();
};
/*
template<typename value_type> struct hash_map_construction_data {
	using simd_type = map_simd_t<2048>;
	std::array<size_t, 2048 / setSimdWidth(2048)> bucketSizes{};
	NIHILUS_ALIGN(cpu_alignment) array<uint8_t, 2049> controlBytes{};
	std::array<uint8_t, 256> uniqueIndices{};
	std::array<size_t, 2049> indices{};
	size_t bucketSize{ setSimdWidth(2048) };
	size_t numGroups{ 2048 / bucketSize };
	std::array<size_t, 256> jsonTypeIndices{};
	ct_key_hasher hasher{};
	hash_map_type type{};
	size_t uniqueIndex{};
	char firstChar{};

	constexpr hash_map_construction_data() noexcept = default;
};

struct simd_full_length_data {
	inline static constexpr size_t storageSize{ 2048 };
	constexpr simd_full_length_data(const hash_map_construction_data& newData) noexcept
		: controlBytes{ newData.controlBytes }, bucketSize{ newData.bucketSize }, numGroups{ newData.numGroups }, indices{ newData.indices }, uniqueIndex{ newData.uniqueIndex },
		  type{ newData.type }, seed{ newData.hasher.seed } {};
	NIHILUS_ALIGN(cpu_alignment) std::array<uint8_t, storageSize + 1> controlBytes {};
	char padding01[cpu_alignment - ((storageSize + 1) % 8)]{};
	size_t bucketSize{ setSimdWidth(storageSize) };
	size_t numGroups{ storageSize / bucketSize };
	array<uint16_t, storageSize + 1> indices{};
	size_t uniqueIndex{};
	hash_map_type type{};
	size_t seed{};
};

template<typename value_type> inline static constexpr auto collectSimdFullLengthHashMapData(const tuple_references& pairsNew) noexcept {
	hash_map_construction_data returnValues{};
	bool collided{};
	for (size_t w = keyStatsVal<value_type>.minLength; w < keyStatsVal<value_type>.maxLength; ++w) {
		returnValues.uniqueIndex = w;
		for (size_t x = 0; x < 2; ++x) {
			returnValues.controlBytes.fill(std::numeric_limits<uint8_t>::max());
			returnValues.indices.fill(static_cast<uint16_t>(returnValues.indices.size() - 1));
			returnValues.hasher.updateSeed();
			collided = false;
			for (size_t y = 0; y < pairsNew.count; ++y) {
				const auto keyLength	 = returnValues.uniqueIndex > pairsNew.rootPtr[y].key.size() ? pairsNew.rootPtr[y].key.size() : returnValues.uniqueIndex;
				const auto hash			 = returnValues.hasher.hashKeyCt(pairsNew.rootPtr[y].key.data(), keyLength);
				const auto groupPos		 = (hash >> 8) % returnValues.numGroups;
				const auto ctrlByte		 = static_cast<uint8_t>(hash);
				const auto bucketSizeNew = returnValues.bucketSizes[groupPos]++;
				const auto slot			 = ((groupPos * returnValues.bucketSize) + bucketSizeNew);

				if (bucketSizeNew >= returnValues.bucketSize || returnValues.indices[slot] != returnValues.indices.size() - 1 ||
					contains(returnValues.controlBytes.data() + groupPos * returnValues.bucketSize, ctrlByte, returnValues.bucketSize)) {
					returnValues.bucketSizes.fill(0);
					collided = true;
					break;
				}
				returnValues.controlBytes[slot] = ctrlByte;
				returnValues.indices[slot]		= pairsNew.rootPtr[y].oldIndex;
			}
			if (!collided) {
				break;
			}
		}
		if (!collided) {
			break;
		}
	}
	if (collided) {
		returnValues.type		 = hash_map_type::unset;
		returnValues.uniqueIndex = std::numeric_limits<size_t>::max();
		return returnValues;
	} else {
		returnValues.type = hash_map_type::simd_full_length;
		return returnValues;
	}
}
*/
//static constexpr auto parse_value = createValue<&test_struct::test_val>();

int main(int argc, char** argv) {
	try {
		keyStatsVal<test_struct>.maxLength;
		core<test_struct> parse_val{};
		parse_val.parse_value.operator[](nihilus::tag<0>{});
		static constexpr auto config_v1_v2_1b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_fp16_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_fp16_moe = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_bf16_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_bf16_gqa = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q4_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q4_gqa = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q4_moe = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q8_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q8_gqa = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q8_moe = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_mixed_fp16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_mixed_bf16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_fp16_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_fp16_moe = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_bf16_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_bf16_gqa = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q4_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q4_gqa = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q4_moe = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q8_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q8_gqa = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q8_moe = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_mixed_fp16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_mixed_bf16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);

		nihilus::stop_watch stop_watch_val{ 0 };
		static constexpr auto model_config = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B, nihilus::kernel_type_profiles::q8_gqa,
			nihilus::model_arches::llama, false);
		nihilus::cli_params cli_args_final{ nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv) };
		auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args_final) };
		bnch_swt::benchmark_stage<"nihilus-vs_llama.cpp", 4, 2, true, "Token">::runBenchmark<"nihilus">([&] {
			while (model_new->process_input(cli_args_final.prompt)) {
			}
			return cli_args_final.n_tokens;
		});
		bnch_swt::benchmark_stage<"nihilus-vs_llama.cpp", 4, 2, true, "Token">::printResults();
	} catch (const std::exception& error) {
		std::cout << "Error: " << error.what() << std::endl;
	}
	return 0;
}