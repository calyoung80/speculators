#!/bin/bash
# Start or resume the validated 20K-step 8K/512/FSDP training configuration.
set -euo pipefail

ARCHIVE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${ARCHIVE_DIR}/../start_formal_qwen38_500k_train.sh" "$@"
