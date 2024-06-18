#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <string>

struct Token {
    size_t      id;      // Unique identifier of the token
    std::string content; // Content of the token
};

struct AddedToken {
    struct Token* token = nullptr;

    bool single_word; // Whether this token is a single word
    bool left_strip;  // Left strip flag
    bool right_strip; // Right strip flag
    bool normalized;  // Normalized flag
    bool special;     // Special flag
};

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
  private:
    // model is the metadata as a JSON object
    nlohmann::json           __model__; // guard against name conflicts
    std::vector<std::string> __reverse_vocab__;

  public:
    TokenizerModel(const nlohmann::json &model) {
        __model__ = model;

        // map vocab to vector elements
        for (auto &[token, id] : __model__["vocab"].items()) {
            __reverse_vocab__[id] = token;
        }
    }

    size_t size() {
        return __model__["vocab"].size();
    };

    std::map<std::string, size_t> vocab() {
        return __model__["vocab"]; // vocab is the result of V* : t -> id
    };

    std::vector<std::string> tokens() const {
        return __reverse_vocab__; // vocab is the result of V* : id -> t
    };

    std::vector<std::string> merges() {
        return __model__["merges"]; // merges is a vector of strings
    };

    std::string type() {
        return __model__.contains("type") ? __model__["type"] : "BPE";
    };

    std::string unk_token() {
        return __model__.contains("unk_token") ? __model__["unk_token"] : "";
    };

    // note: may be null or string
    std::string continuing_subword_prefix() {
        return __model__.contains("continuing_subword_prefix")
                   ? __model__["continuing_subword_prefix"]
                   : "";
    };

    std::string end_of_word_suffix() {
        return __model__.contains("end_of_word_suffix") ? __model__["end_of_word_suffix"] : "";
    };

    float dropout() {
        return __model__.contains("dropout") ? __model__["dropout"].template get<float>() : 0.0f;
    };

    bool fuse_unk() {
        return __model__.contains("fuse_unk") ? __model__["fuse_unk"].template get<bool>() : false;
    };

    bool byte_fallback() {
        return __model__.contains("byte_fallback") ? __model__["byte_fallback"].template get<bool>()
                                                   : false;
    };

    bool ignore_merges() {
        return __model__.contains("ignore_merges") ? __model__["ignore_merges"].template get<bool>()
                                                   : false;
    };
};

// NOTE: This is a public class
struct Tokenizer {
  public:
    // tokenizer model type: only supported implementation will be BPE
    std::string    type;
    // the huggingface tokenizers compatible model metadata
    TokenizerModel model;

    std::vector<struct AddedToken*> added_tokens;

    // Need to know these advance. Must be set on a model-by-model basis as a result.
    struct SpecialToken* bos_token = nullptr;
    struct SpecialToken* eos_token = nullptr;
    struct SpecialToken* unk_token = nullptr;

    size_t size() {
        return model.size();
    };

    size_t token_to_id(const std::string &token) {
        return model.vocab()[token];
    };

    std::string id_to_token(size_t encoding) const {
        return model.tokens()[encoding];
    };

    // TODO/WIP: Note that normalize and pre_tokenizer are variable objects
    nlohmann::json normalizer;
    nlohmann::json pre_tokenizer;
};

#endif // TOKENIZER_H
