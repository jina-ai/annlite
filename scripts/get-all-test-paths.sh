#!/usr/bin/env bash

set -ex

BATCH_SIZE=5

declare -a array1=( "tests/test_*.py" )
declare -a array2=( "tests/docarray/test_*.py" )
dest=( "${array1[@]}" "${array2[@]}")

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .
