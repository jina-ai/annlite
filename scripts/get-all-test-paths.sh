#!/usr/bin/env bash

set -ex

BATCH_SIZE=5

declare -a array1=( "test_*.py" )
declare -a array2=( "docarray/test_*.py" )
declare -a array3=( "executor/test_*.py" )
dest=( "${array1[@]}" "${array2[@]}" "${array3[@]}" )

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .
