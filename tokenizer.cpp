#include "tokenizer.h"

#include <cstring>
#include <filesystem>
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

struct TokenizerModel* malloc_tokenizer_model(nlohmann::json model) {
    if (model.is_null()) {
        throw std::invalid_argument("Expected a valid model argument, got null instead.");
    }

    struct TokenizerModel* tokenizer
        = (struct TokenizerModel*) malloc(sizeof(struct TokenizerModel));

    if (nullptr == tokenizer) {
        throw std::bad_alloc();
    }

    // NOTE: This is not always available
    tokenizer->type = model.contains("type") ? model["type"] : "BPE";

    // V* ≅ [N_V] where V* is set of tokens and N_V is the vocab size
    // e.g. The set of tokens is congruent with the vocab size
    tokenizer->size  = model["vocab"].size();
    // V* : t -> i where V* is set of tokens, t is token, and i is id
    // e.g. this is a "forward mapping"
    tokenizer->vocab = model["vocab"];
    // V*: i -> t where V* is set of tokens, i is id, and t is token
    // e.g. this is a "reverse mapping"
    for (auto &[token, id] : model["vocab"].items()) {
        tokenizer->tokens[id] = token;
    }

    tokenizer->byte_fallback
        = model.contains("byte_fallback") ? model["byte_fallback"].template get<bool>() : false;

    // merges is a vector of strings
    tokenizer->merges = model["merges"];
    tokenizer->ignore_merges
        = model.contains("ignore_merges") ? model["ignore_merges"].template get<bool>() : false;

    tokenizer->dropout = model.contains("dropout") ? model["dropout"].template get<float>() : 0.0f;

    return tokenizer;
}

void free_tokenizer_model(struct TokenizerModel* tokenizer_model) {
    if (nullptr != tokenizer_model) {
        free(tokenizer_model);
    }
}

struct Tokenizer* malloc_tokenizer(nlohmann::json data) {
    if (data.is_null()) {
        throw std::invalid_argument("Expected a valid model argument, got null instead.");
    }

    struct Tokenizer* tokenizer = (struct Tokenizer*) malloc(sizeof(struct Tokenizer));

    if (nullptr == tokenizer) {
        throw std::bad_alloc();
    }

    tokenizer->model        = malloc_tokenizer_model(data["model"]);
    tokenizer->added_tokens = malloc_added_tokens(data["added_tokens"]);

    // TODO/WIP: Note that normalize and pre_tokenizer are variable objects.
    // using nlohmann::json data types to ensure sane defaults for now.
    tokenizer->normalizer    = data["normalizer"];
    tokenizer->pre_tokenizer = data["pre_tokenizer"];

    return tokenizer;
}

void free_tokenizer(struct Tokenizer* data) {
    if (nullptr != data) {
        free_tokenizer_model(data->model);
        free_added_tokens(data->added_tokens);
        free(data);
    }
}

int main(int argc, char* argv[]) {
    if (1 == argc) {
        puts("Usage: vocab [-p <path>]");
        return 1;
    }

    const char* const   short_options = "p:";
    const struct option long_options[]
        = {{"tokenizer-path", required_argument, nullptr, 'p'}, NULL};

    int                   opt;
    std::filesystem::path directory;

    while ((opt = getopt_long(argc, argv, short_options, long_options, nullptr)) != -1) {
        switch (opt) {
            case 'p':
                if (optarg == nullptr || strlen(optarg) < 1) {
                    puts("Error: Invalid file path specified.");
                    return 1;
                }

                directory = std::filesystem::path(optarg);
                break;

            default:
                puts("Usage: vocab [-p <tokenizer-path>]");
                return 1;
        }
    }

    std::filesystem::path tokenizer_json = directory / "tokenizer.json";

    fprintf(stdout, "using: %s\n", tokenizer_json.c_str());

    std::ifstream  f(tokenizer_json);
    nlohmann::json data = nlohmann::json::parse(f);

    if (data.is_null()) {
        fprintf(stderr, "Error: Unable to parse tokenizer.json file.\n");
        return 1;
    }

    const std::string version = data["version"];

    fprintf(stdout, "version: %s\n", version.c_str());

    struct Tokenizer* tokenizer = malloc_tokenizer(data);

    fprintf(stdout, "tokenizer->model->type: %s\n", tokenizer->type().c_str());

    free_tokenizer(tokenizer);

    return 0;
}
