#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <string>

struct Token {
    size_t      id;      // Unique identifier of the token
    std::string content; // Content of the token
};

// allocate memory to a token
struct Token* malloc_token(size_t id, std::string content);

// deallocate memory from a token
void free_token(struct Token* token);

struct AddedToken {
    struct Token* token = nullptr;

    bool single_word; // Whether this token is a single word
    bool left_strip;  // Left strip flag
    bool right_strip; // Right strip flag
    bool normalized;  // Normalized flag
    bool special;     // Special flag
};

// allocate memory to a vector of added tokens
std::vector<struct AddedToken*> malloc_added_tokens(nlohmann::json added_tokens);

// free memory from a vector of a added tokens
void free_added_tokens(std::vector<struct AddedToken*> added_tokens);

// TODO/WIP
struct Normalizer {
    // there are 10 possible types of normalizers
    std::string type; // type is always available if normalizer is not null

    // TODO: Handle sequences if type is Sequence
    std::vector<nlohmann::json> rules;

    // default to false because these member attributes may not be available
    bool add_prefix_space = false;
    bool trim_offsets     = false;
};

// TODO/WIP
struct PreTokenizer {
    // there are 9 possible types of pre_tokenizers
    std::string type; // type is always available if pre_tokenizer is not null

    // TODO: Handle sequences if type is Sequence
    std::vector<nlohmann::json> rules;

    // default to false because these member attributes may not be available
    bool add_prefix_space = false;
    bool trim_offsets     = false;
};

struct TokenizerModel {
    // BPE, WPM, etc...
    std::string type;

    // V* ≅ [N_V] where V* is set of tokens and N_V is the vocab size
    // e.g. The set of tokens is congruent with the vocab size
    size_t size;

    // V* : t -> i where V* is set of tokens, t is token, and i is id
    // e.g. this is a "forward mapping"
    std::map<std::string, size_t> vocab;

    // V*: i -> t where V* is set of tokens, i is id, and t is token
    // e.g. this is a "reverse mapping"
    std::vector<std::string> tokens;

    // The array of merged tokens as strings.
    // NOTE: These should be split before use
    // with start and end indices included, respectively.
    std::vector<std::string> merges;

    bool byte_fallback;
    bool ignore_merges;

    float dropout;
};

struct TokenizerModel* malloc_tokenizer_model(nlohmann::json model);

void free_tokenizer_model(struct TokenizerModel* model);

// NOTE: This is a public class
struct Tokenizer {
    // the huggingface tokenizers compatible model metadata
    TokenizerModel model;

    std::vector<struct AddedToken*> added_tokens;

    // Need to know these advance. Must be set on a model-by-model basis as a result.
    struct Token* bos_token = nullptr;
    struct Token* eos_token = nullptr;
    struct Token* unk_token = nullptr;

    // tokenizer model type: only supported implementation will be BPE
    std::string type() {
        return model.type;
    };

    size_t size() {
        return model.size;
    };

    size_t token_to_id(const std::string &token) {
        return model.vocab[token];
    };

    std::string id_to_token(size_t encoding) const {
        return model.tokens[encoding];
    };

    // TODO/WIP: Note that normalize and pre_tokenizer are variable objects
    nlohmann::json normalizer;
    nlohmann::json pre_tokenizer;
};

struct Tokenizer* malloc_tokenizer(nlohmann::json data);

void free_tokenizer(struct TokenizerModel* data);

#endif // TOKENIZER_H
