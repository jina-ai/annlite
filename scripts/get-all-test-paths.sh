#!/usr/bin/env bash

set -ex

BATCH_SIZE=5

declare -a array1=( "tests/test_*.py" )
dest=( "${array1[@]}" )

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .
