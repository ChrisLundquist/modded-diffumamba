#!/bin/bash
# Download pre-tokenized FineWeb-10B via curl (no Python deps needed).
# Each shard is ~200MB (100M tokens as uint16).
# Same data as modded-nanogpt: kjj0/fineweb10B-gpt2
set -e

DIR="$(cd "$(dirname "$0")" && pwd)/fineweb10B"
mkdir -p "$DIR"
BASE="https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"
N=${1:-10}  # train shards (100M tokens each), default 10 = 1B tokens

# Val shard
if [ ! -f "$DIR/fineweb_val_000000.bin" ]; then
    echo "Downloading val shard..."
    curl -L -o "$DIR/fineweb_val_000000.bin" "$BASE/fineweb_val_000000.bin" -#
fi

# Train shards
for i in $(seq 1 "$N"); do
    FNAME=$(printf "fineweb_train_%06d.bin" "$i")
    if [ ! -f "$DIR/$FNAME" ]; then
        echo "Downloading $FNAME ($i/$N)..."
        curl -L -o "$DIR/$FNAME" "$BASE/$FNAME" -#
    else
        echo "Skipping $FNAME (exists)"
    fi
done

echo "Done! $((N * 100))M train tokens + 100M val tokens in $DIR/"
