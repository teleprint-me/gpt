#!/usr/bin/env bash

declare -A URL_MAP=(
    ["zip"]="https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT"
    ["gguf"]="https://raw.githubusercontent.com/ggerganov/ggml/master/docs/gguf.md"
    ["1508.07909v5"]="https://arxiv.org/pdf/1508.07909v5"
    ["1706.03762"]="https://arxiv.org/pdf/1706.03762"
    ["2207.09238"]="https://arxiv.org/pdf/2207.09238"
    ["1910.13267"]="https://arxiv.org/pdf/1910.13267"

)

declare -A OUTPUT_MAP=(
    ["zip"]="specs/zip.txt"
    ["gguf"]="specs/gguf.md"
    ["1508.07909v5"]="specs/1508.07909v5.pdf"
    ["1706.03762"]="specs/1706.03762.pdf"
    ["2207.09238"]="specs/2207.09238.pdf"
    ["1910.13267"]="specs/1910.13267.pdf"
)

for key in "${!URL_MAP[@]}"
do
  wget "${URL_MAP[$key]}" -O "${OUTPUT_MAP[$key]}"
done
