#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # Load repo-local overrides such as OLLAMA_VALIDATOR_MODEL.
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

OLLAMA_MODEL_LIST="${OLLAMA_MODELS:-${OLLAMA_VALIDATOR_MODEL:-qwen2.5vl:7b}}"
OLLAMA_MODEL_LIST="${OLLAMA_MODEL_LIST//,/ }"

install_ollama() {
  if command -v ollama >/dev/null 2>&1; then
    return
  fi

  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        brew install ollama
      else
        curl -fsSL https://ollama.com/install.sh | sh
      fi
      ;;
    Linux)
      curl -fsSL https://ollama.com/install.sh | sh
      ;;
    *)
      echo "Unsupported operating system for automated Ollama installation: $(uname -s)" >&2
      exit 1
      ;;
  esac
}

ensure_ollama_server() {
  if ollama list >/dev/null 2>&1; then
    return
  fi

  if [[ "$(uname -s)" == "Darwin" && -x /opt/homebrew/bin/brew ]]; then
    brew services start ollama >/dev/null 2>&1 || true
  elif [[ "$(uname -s)" == "Darwin" && -x /usr/local/bin/brew ]]; then
    brew services start ollama >/dev/null 2>&1 || true
  else
    nohup ollama serve >/tmp/wildtrace-ollama.log 2>&1 &
  fi

  for _ in {1..30}; do
    if ollama list >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done

  echo "Ollama did not become ready. Check the service or /tmp/wildtrace-ollama.log." >&2
  exit 1
}

install_ollama
ensure_ollama_server

for model in ${OLLAMA_MODEL_LIST}; do
  echo "Pulling Ollama model: ${model}"
  ollama pull "${model}"
done

echo "Ollama installation and model pull complete."
