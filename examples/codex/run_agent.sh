#!/usr/bin/env bash
# In-sandbox entrypoint for the Codex sidecar.
set -uo pipefail

TOOL_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PROJECT_DIR="${CODEX_PROJECT_DIR:-${PWD}}"
CODEX_HOME="${CODEX_HOME:-${PROJECT_DIR}/.agent-home}"
MODEL="${CODEX_MODEL:?missing CODEX_MODEL}"
API_BASE="${CODEX_API_BASE:?missing CODEX_API_BASE}"
API_KEY="${CODEX_API_KEY:-EMPTY}"
PROXY_PORT="${CODEX_PROXY_PORT:-18999}"
PROXY_BASE="http://127.0.0.1:${PROXY_PORT}/v1"

mkdir -p "${CODEX_HOME}"
cat >"${CODEX_HOME}/config.toml" <<EOF
model_provider = "gateway"
model = "${MODEL}"
disable_response_storage = true
check_for_update_on_startup = false

[model_providers.gateway]
name = "Policy Gateway"
base_url = "${PROXY_BASE}"
wire_api = "responses"
requires_openai_auth = true
EOF

export CODEX_HOME
export OPENAI_API_KEY="${API_KEY}"
export CODEX_MANAGED_PACKAGE_ROOT="${CODEX_MANAGED_PACKAGE_ROOT:-${TOOL_ROOT}}"
export NO_PROXY="*"
export no_proxy="*"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

"${TOOL_ROOT}/bin/python" "${TOOL_ROOT}/bin/responses_proxy.py" \
  --listen "127.0.0.1:${PROXY_PORT}" \
  --upstream "${API_BASE}" &
PROXY_PID=$!
trap 'kill "${PROXY_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 100); do
  if "${TOOL_ROOT}/bin/python" -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PROXY_PORT}/health', timeout=1).read()" \
    >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done

cd "${PROJECT_DIR}"
"${TOOL_ROOT}/bin/codex" exec \
  --json \
  --ephemeral \
  --skip-git-repo-check \
  --dangerously-bypass-approvals-and-sandbox \
  --cd "${PROJECT_DIR}" \
  --model "${MODEL}" \
  -
STATUS=$?
printf '{"type":"process.completed","exit_code":%d}\n' "${STATUS}"
# The host parses the child status from the final JSONL event. Returning 0
# preserves stdout on sandbox providers that raise on non-zero commands.
exit 0
