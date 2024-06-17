#include <climits>
#include <cstdarg>
#include <cstring>
#include <format>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <string>
#include <sys/wait.h>
#include <vector>
#include <wait.h>

// NOTE: AddedToken represents an object element found within added_tokens in the tokenizer.json
struct AddedToken {
  private:
    nlohmann::json __token__;

  public:
    AddedToken(nlohmann::json token) {
        __token__ = token;

        id          = token["id"];
        content     = token["content"];
        single_word = token["single_word"];
        lstrip      = token["lstrip"];
        rstrip      = token["rstrip"];
        normalized  = token["normalized"];
        special     = token["special"];
    }

    size_t      id;      // Unique identifier of the token
    std::string content; // Content of the token

    bool single_word; // Whether this token is a single word
    bool lstrip;      // Left strip flag
    bool rstrip;      // Right strip flag
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

    std::vector<struct AddedToken*> added_tokens;

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

struct Tokenizer {
    // tokenizer model type: only supported implementation will be BPE
    std::string    type;
    // the huggingface tokenizers compatible model metadata
    TokenizerModel model;

    // Need to know these advance. Must be set on a model-by-model basis as a result.
    struct AddedToken* bos_token = nullptr;
    struct AddedToken* eos_token = nullptr;
    struct AddedToken* unk_token = nullptr;

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

std::vector<struct AddedToken*> allocate_added_tokens(nlohmann::json added_tokens) {

    if (added_tokens.is_null()) {
        throw std::invalid_argument("Expected a valid added_tokens argument, got null instead.");
    }

    std::vector<struct AddedToken*> token_set;

    // added_tokens is a JSON list of JSON objects
    for (nlohmann::json object : added_tokens) {
        struct AddedToken* token = new AddedToken(object);
        // technically, we can have these in the stack. tbh, not sure if matters.
        token_set.push_back(token);
    }

    return token_set;
}

void deallocate_added_tokens(std::vector<struct AddedToken*> token_set) {
    // Deallocate memory for all struct AddedToken objects
    for (struct AddedToken* token : token_set) {
        delete token;
    }
}

int main(int argc, char* argv[]) {
    if (1 == argc) {
        puts("Usage: vocab [-f <file>] [-v <vocab-type>]");
        return 1;
    }

    const char* const   short_options = "f:v:";
    const struct option long_options[]
        = {{"registry-file", required_argument, nullptr, 'f'},
           {"vocab-type", optional_argument, nullptr, 'v'}};

    int  opt;
    char registry_file_path[1024];
    char vocab_name[5] = "BPE"; // NOTE: avoid name conflict

    while ((opt = getopt_long(argc, argv, short_options, long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                if (optarg == nullptr || strlen(optarg) < 1) {
                    puts("Error: Invalid file path specified.");
                    return 1;
                }

                strcpy(registry_file_path, optarg);
                break;

            case 'v':
                if (optarg == nullptr || strlen(optarg) > 3) {
                    puts("Error: Invalid vocab type specified.");
                    return 1;
                }

                strncpy(vocab_name, optarg, 5);
                break;

            default:
                puts("Usage: vocab [-f <file>] [-v <vocab-type>]");
                return 1;
        }
    }

    std::ifstream  f(registry_file_path);
    nlohmann::json data = nlohmann::json::parse(f);

    if (data.is_null()) {
        puts("Error: Unable to parse registry file.");
        return 1;
    }

    const std::string version = data["version"];
    std::cout << "version: " << version << std::endl;
    const nlohmann::json model_arch = data["truncation"];
    const nlohmann::json padding    = data["padding"];

    // for (const auto &config : data) {
    //     std::cout << "Model Repository:\t" << model_repo << '\n';
    //     std::cout << "Model Architecture:\t" << model_arch << '\n';

    //     // normalizer and pre_tokenizer are either null or an object
    //     const nlohmann::json normalizer    = config["normalizer"];
    //     const nlohmann::json pre_tokenizer = config["pre_tokenizer"];

    //     // NOTE: Normalizer may be one of null, Sequence, NFC, NFD, NFKC, NFKD...
    //     // Seems to be null, Sequence, or NFC in most cases
    //     // Attempt to handle cases where a key may or may not be present and have a null value
    //     // Not sure whether to default to NFD or NFC if normalizer is null
    //     // NOTE: The normalizer type key is not guaranteed
    //     if (!normalizer.is_null()) {
    //         const std::string norm_type = normalizer["type"];
    //         std::cout << "Normalizer Type:\t" << norm_type << std::endl;
    //         // sequence is an array
    //         if (0 == norm_type.compare("Sequence")) {
    //             const nlohmann::json normalizers = normalizer["normalizers"];
    //             if (normalizers.is_array()) {
    //                 for (const auto &norm : normalizers.items()) {
    //                     std::cout << "Norm Sequence Object:\t" << norm.key() << ":\t"
    //                               << norm.value() << std::endl;
    //                 }
    //             }
    //         } else {
    //             // otherwise norm_type is an object potentially containing properties
    //             // this varies from model to model
    //             // maybe we can just dump the entire object? i don't know yet...
    //             for (const auto &norm : normalizer.items()) {
    //                 std::cout << "Pre Object Pair:\t" << norm.key() << ":\t" << norm.value()
    //                           << std::endl;
    //             }
    //         }
    //     }
    // }

    return 0;
}
