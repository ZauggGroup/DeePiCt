#!/bin/bash

srcdir=$(dirname ${BASH_SOURCE[0]})
config_file=$1
cp "${srcdir}/config.yaml" $(pwd)/"${config_file}"
