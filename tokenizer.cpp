#include "tokenizer.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <stdexcept>

struct Token* malloc_token(size_t id, std::string content) {
    // allocate memory for the token object
    struct Token* token = new Token{};

    if (nullptr == token) {
        puts("token");
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
    if (token) {
        free(token);
    }
}

std::vector<struct AddedToken*> malloc_added_tokens(nlohmann::json added_tokens) {
    if (added_tokens.is_null()) {
        throw std::invalid_argument("Expected a valid added_tokens argument, got null instead.");
    }

    std::vector<struct AddedToken*>(tokens);
    tokens.reserve(added_tokens.size()); // allocate space for added tokens

    // added_tokens is a JSON list of JSON objects
    for (nlohmann::json object : added_tokens) {
        struct AddedToken* added = new AddedToken{};

        if (nullptr == added) {
            puts("added tokens");
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
        if (added) {
            free_token(added->token);
            free(added);
        }
    }
}

// note: what a fucking nightmare! the variable state of a tokenizer.json makes this challenging.
// will need to dig deeper into huggingface/tokenizers source code to figure out an optimal path
// forward.
struct TokenizerModel* malloc_tokenizer_model(nlohmann::json model) {
    if (model.is_null()) {
        throw std::invalid_argument("Expected a valid model argument, got null instead.");
    }

    struct TokenizerModel* tokenizer = new TokenizerModel{};

    if (!tokenizer) {
        throw std::bad_alloc();
    }

    // NOTE: model["type"] is not always available, so tokenizer->type will default to BPE if
    // unavailable. The value may be present and null, so we ignore this edge case.
    if (model.contains("type") && !model["type"].is_null()) {
        tokenizer->type = model["type"];
    }
    fprintf(stderr, "set type: %s\n", tokenizer->type.c_str());

    // V* ≅ [N_V] where V* is set of tokens and N_V is the vocab size
    // e.g. The set of tokens is congruent with the vocab size
    if (!model.contains("vocab")) { // if vocab is missing, something is wrong.
        throw std::runtime_error("Missing key: tokenizer['model'] must contain a 'vocab' key.");
    }
    tokenizer->size = model["vocab"].size();
    fprintf(stderr, "set size: %zd\n", tokenizer->size);

    // V* : t -> i where V* is set of tokens, t is token, and i is id
    // e.g. this is a "forward mapping"
    tokenizer->vocab = model["vocab"];
    fprintf(stderr, "set vocab\n"); // too large to print

    /* map is unordered, but we need an ordered vector of tokens
     * so we reserve an allocated amount based on the number of tokens
     * this spares us the hassle of applying std::ordered_map to nlohmann::json::object
     */
    tokenizer->tokens.reserve(tokenizer->size);

    // V*: i -> t where V* is set of tokens, i is id, and t is token
    // e.g. this is a "reverse mapping"
    for (auto &[token, id] : model["vocab"].items()) {
        // todo: token rendering needs to be fixed.
        // note: expected utf-8, but most likely got utf-16. need to investigate further.
        std::string t                       = token; // this is still producing mojibaked tokens
        // fprintf(stderr, "token: %s, id: %d\n", t, i); // temp to view the issue with tokens
        tokenizer->tokens[id.get<size_t>()] = token;
    }
    fprintf(stderr, "set tokens\n"); // too large to print

    // merges is a vector of strings
    if (!model.contains("merges")) {
        throw std::domain_error("Missing key: tokenizer['model'] must contain a 'merges' key.");
    }
    tokenizer->merges.reserve(model["merges"].size());
    tokenizer->merges = model["merges"]; // if merges is missing, something is wrong.
    fprintf(stderr, "set merges\n");     // too large to print

    if (model.contains("byte_fallback") && !model["byte_fallback"].is_null()) {
        tokenizer->byte_fallback = model["byte_fallback"].get<bool>();
    }
    fprintf(stderr, "byte fallback: %d\n", tokenizer->byte_fallback);

    if (model.contains("ignore_merges") && !model["ignore_merges"].is_null()) {
        tokenizer->ignore_merges = model["ignore_merges"].get<bool>();
    }
    fprintf(stderr, "ignore merges: %d\n", tokenizer->ignore_merges);

    if (model.contains("dropout") && !model["dropout"].is_null()) {
        tokenizer->dropout = model["dropout"].get<float>();
    }
    fprintf(stderr, "dropout: %f\n", tokenizer->dropout);

    fprintf(stderr, "created tokenizer <3\n");
    return tokenizer;
}

void free_tokenizer_model(struct TokenizerModel* tokenizer_model) {
    if (tokenizer_model) {
        free(tokenizer_model);
    }
}

struct Tokenizer* malloc_tokenizer(nlohmann::json data) {
    if (data.is_null()) {
        throw std::invalid_argument("Expected a valid model argument, got null instead.");
    }

    struct Tokenizer* tokenizer = new Tokenizer{};

    if (nullptr == tokenizer) {
        puts("tokenizer");
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
