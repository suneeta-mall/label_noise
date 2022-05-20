#!/bin/bash
set -euo pipefail
set -x


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd ${SCRIPT_DIR}/..

poetry run isort . && \
poetry run black . && \
poetry run pylint -j 4 --reports=y -v --rcfile=.dev/.pylintrc ./label_noise ./tests && \
poetry run mypy .


popd