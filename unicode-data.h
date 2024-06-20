// generated with python gguf.cli.unicode
#ifndef UNICODE_DATA_H
#define UNICODE_DATA_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

/**
* @brief Represents a Unicode character range with normalized form D (NFD)
*/
struct range_nfd {
uint32_t first;
uint32_t last;
uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 1114112;

/**
* @brief Externally linked variables for Unicode data structures
*/
extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase;
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase;
extern const std::vector<range_nfd> unicode_ranges_nfd;
#endif // UNICODE_DATA_H
