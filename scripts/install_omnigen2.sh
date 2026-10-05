#!/usr/bin/env bash
# Fetch OmniGen2 pipeline code (pinned commit) and weights for the
# omnigen2_rectifier backend. The code is put on sys.path at load time, not
# pip-installed: its requirements pin torch 2.6 / transformers 4.51.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${OMNIGEN2_CODE_DIR:-${HOME}/.cache/wildtrace/OmniGen2}"
CODE_COMMIT="18e6f9d5271b517fcb32e999f10df943ae9b8f20"
WEIGHTS_REVISION="df5dca8a981d74e6c3af214c145f5c735fe72367"

if [[ ! -d "${CODE_DIR}/.git" ]]; then
  git clone https://github.com/VectorSpaceLab/OmniGen2 "${CODE_DIR}"
fi
git -C "${CODE_DIR}" fetch --depth 1 origin "${CODE_COMMIT}"
git -C "${CODE_DIR}" checkout --quiet "${CODE_COMMIT}"

# ~31GB of fp32 shards, then a one-time bf16 export (~16GB) that the
# rectifier loads instead, so loads never hold an fp32 copy in RAM.
cd "${REPO_ROOT}"
uv run python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('OmniGen2/OmniGen2', revision='${WEIGHTS_REVISION}',
                        allow_patterns=['*.json', '*.py', '*.safetensors', '*.txt']))
"
uv run python -c "
from pathlib import Path
from wildtrace.config import load_runtime_config
from wildtrace.diagram import OmniGen2Rectifier
settings = load_runtime_config(Path('.').resolve())['models']['outline_rectifier']
print('bf16 export:', OmniGen2Rectifier(settings).export_bf16())
"
