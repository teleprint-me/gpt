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

struct special_token {
    int         id;
    std::string content;
    bool        single_word;
    bool        lstrip;
    bool        rstrip;
    bool        normalized;
    bool        special;
};

// TODO/WIP
struct normalizer {
    std::string type;
    bool        add_prefix_space;
    bool        trim_offsets;
};

// TODO/WIP
struct pre_tokenizer {
    std::string type;
    bool        add_prefix_space;
    bool        trim_offsets;
};

struct tokenizer_model {

    std::string type;

    struct special_token* bos_token;
    struct special_token* eos_token;
    struct special_token* unk_token;

    bool fuse_unk;
    bool byte_fallback;

    // TODO/WIP
    nlohmann::json normalizer;
    nlohmann::json pre_tokenizer;

    std::map<std::string, uint32_t> vocab;
    std::vector<std::string>        merges;
};

void set_special_tokens(nlohmann::json added_tokens, struct tokenizer_model*) {
    struct special_token object;
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
