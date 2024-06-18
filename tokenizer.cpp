#include "tokenizer.h"

#include <cstring>
#include <fstream>
#include <getopt.h>

struct Token* malloc_token(size_t id, std::string content) {
    // allocate memory for the token object
    struct Token* token = (struct Token*) malloc(sizeof(struct Token));

    if (nullptr == token) {
        throw std::bad_alloc();
    }

    // set the token object values
    token->id      = id;
    token->content = content;

    // return the newly constructed token
    return token;
}

void free_token(struct Token* token) {
    // free token if and only if token is not null
    if (nullptr != token) {
        free(token);
    }
}

std::vector<struct AddedToken*> malloc_added_tokens(nlohmann::json added_tokens) {
    if (added_tokens.is_null()) {
        throw std::invalid_argument("Expected a valid added_tokens argument, got null instead.");
    }

    std::vector<struct AddedToken*> tokens;

    // added_tokens is a JSON list of JSON objects
    for (nlohmann::json object : added_tokens) {
        struct AddedToken* added = (struct AddedToken*) malloc(sizeof(struct AddedToken));

        if (nullptr == added) {
            throw std::bad_alloc();
        }

        size_t      id      = object["id"].template get<size_t>();
        std::string content = object["content"].template get<std::string>();
        added->token        = malloc_token(id, content);

        added->single_word = object["single_word"].template get<bool>();
        added->left_strip  = object["lstrip"].template get<bool>();
        added->right_strip = object["rstrip"].template get<bool>();
        added->normalized  = object["normalized"].template get<bool>();
        added->special     = object["special"].template get<bool>();

        // technically, we can have these in the stack. tbh, not sure if matters.
        tokens.push_back(added);
    }

    return tokens;
}

void free_added_tokens(std::vector<struct AddedToken*> added_tokens) {
    // Deallocate memory for all struct AddedToken objects
    for (struct AddedToken* added : added_tokens) {
        if (nullptr != added) {
            free_token(added->token);
            free(added);
        }
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

    fprintf(stdout, "version: %s\n", version.c_str());

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
