#!/usr/bin/env bash
#
# qwencode-setup.sh — point Qwen Code (the qwen CLI) at the Qwen3.8 endpoint on
# Bunya.
#
# Run this wherever qwen-code runs: the GPU node itself (bun159/160/161), the
# login node, or your laptop (with an SSH tunnel to the node).
#
# Usage:
#   ./qwencode-setup.sh                          # endpoint on localhost:30000
#   ./qwencode-setup.sh --host bun159 --port 30000
#   ./qwencode-setup.sh --host localhost --api-key sk-...   # laptop w/ tunnel
#   ./qwencode-setup.sh --env-only               # print the env-var route, write nothing
#
# TWO ROUTES, AND THIS SCRIPT SETS UP BOTH.
#
# Qwen Code resolves configuration in this priority order:
#   1. CLI flags
#   2. shell exports
#   3. .env files  (.qwen/.env -> .env -> ~/.qwen/.env -> ~/.env)
#   4. ~/.qwen/settings.json 'env' field
#
# The three environment variables — OPENAI_API_KEY, OPENAI_BASE_URL and
# OPENAI_MODEL — are the documented, stable interface and outrank the settings
# file. The settings.json 'modelProviders' schema is newer and has been
# described more than one way upstream, so this script writes BOTH: the
# settings block (so the model appears by name in the picker) and ~/.qwen/.env
# (so it works regardless). If the settings block ever stops matching upstream's
# schema, the .env route keeps you serving.
#
# The advertised context limit and model name are taken from CONTEXT_LEN and
# SERVED_MODEL_NAME in qwen38.env, so they stay in step with what the server is
# actually configured to accept. Override with --context / --model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/qwencode.qwen38.json"

# Pick up defaults from qwen38.env if it's here (cluster-side).
if [[ -f "$SCRIPT_DIR/qwen38.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/qwen38.env"
fi

HOST="localhost"
PORT="${PORT:-30000}"
MODEL="${SERVED_MODEL_NAME:-qwen3.8}"
# CONTEXT_LEN is empty by default (server uses the model max), so advertise
# Qwen3.8's native 262,144 window unless it has been explicitly capped.
CONTEXT="${CONTEXT_LEN:-}"
CONTEXT="${CONTEXT:-262144}"
API_KEY="${QWEN38_API_KEY:-}"
ENV_ONLY=0
NO_KEY=0
QWEN_HOME="${QWEN_CODE_HOME:-$HOME/.qwen}"
CONFIG_PATH="$QWEN_HOME/settings.json"
ENV_PATH="$QWEN_HOME/.env"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --context)   CONTEXT="$2"; shift 2 ;;
        --api-key)   API_KEY="$2"; shift 2 ;;
        --env-only)  ENV_ONLY=1; shift ;;
        --no-key)    NO_KEY=1; shift ;;
        --config)    CONFIG_PATH="$2"; shift 2 ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           die "Unknown argument: $1 (see --help)" ;;
    esac
done

[[ -f "$TEMPLATE" ]] || die "Template not found: $TEMPLATE"

BASE_URL="http://$HOST:$PORT/v1"

# Resolve the API key if not given: try the persisted key file on scratch.
if [[ "$NO_KEY" -eq 0 ]]; then
    if [[ -z "$API_KEY" && -n "${MODEL_CACHE_DIR:-}" && -r "$MODEL_CACHE_DIR/qwen38-api-key" ]]; then
        API_KEY="$(<"$MODEL_CACHE_DIR/qwen38-api-key")"
    fi
    if [[ -z "$API_KEY" ]]; then
        die "No API key found. Export QWEN38_API_KEY, pass --api-key, use --no-key, or run on the cluster where \$MODEL_CACHE_DIR/qwen38-api-key exists.
On the cluster:  cat \$MODEL_CACHE_DIR/qwen38-api-key"
    fi
fi

# ── --env-only: print and stop ──────────────────────────────────────────────

if [[ "$ENV_ONLY" -eq 1 ]]; then
    echo "Export these in the shell you start qwen from:"
    echo
    echo "  export OPENAI_BASE_URL=\"$BASE_URL\""
    echo "  export OPENAI_MODEL=\"$MODEL\""
    echo "  export OPENAI_API_KEY=\"${API_KEY:-<key>}\""
    echo
    echo "Then:  qwen"
    exit 0
fi

# ── Route 1: ~/.qwen/.env (the documented, stable interface) ────────────────
#
# Written with umask 077 because it carries the key. Managed as a fenced block
# so re-running replaces rather than duplicates, and anything else in the file
# survives.

BEGIN_MARK='# >>> qwen38-bunya (managed by qwencode-setup.sh) >>>'
END_MARK='# <<< qwen38-bunya <<<'

umask 077
mkdir -p "$QWEN_HOME"

env_block="$(printf '%s\nOPENAI_BASE_URL=%s\nOPENAI_MODEL=%s\n' \
    "$BEGIN_MARK" "$BASE_URL" "$MODEL")"
if [[ "$NO_KEY" -eq 0 ]]; then
    env_block+=$'\n'"OPENAI_API_KEY=$API_KEY"
fi
env_block+=$'\n'"$END_MARK"

if [[ ! -f "$ENV_PATH" ]]; then
    printf '%s\n' "$env_block" > "$ENV_PATH"
    echo "Wrote $ENV_PATH"
else
    cp "$ENV_PATH" "$ENV_PATH.bak"
    tmp="$(mktemp)"
    # Strip any previously managed block, keeping everything else untouched,
    # then re-append.
    awk -v s="$BEGIN_MARK" -v e="$END_MARK" '
        index($0, s) == 1 { skip = 1 }
        !skip             { print }
        index($0, e) == 1 { skip = 0 }
    ' "$ENV_PATH" \
    | awk 'NF { last = NR } { line[NR] = $0 } END { for (i = 1; i <= last; i++) print line[i] }' \
    > "$tmp"
    [[ -s "$tmp" ]] && printf '\n' >> "$tmp"
    printf '%s\n' "$env_block" >> "$tmp"
    mv "$tmp" "$ENV_PATH"
    if grep -qF "$BEGIN_MARK" "$ENV_PATH.bak"; then
        echo "Replaced the qwen38-bunya block in $ENV_PATH (backup at $ENV_PATH.bak)"
    else
        echo "Added the qwen38-bunya block to $ENV_PATH (backup at $ENV_PATH.bak)"
    fi
fi

# ── Route 2: ~/.qwen/settings.json (so it shows up by name) ─────────────────

provider_json="$(sed -e "s/__HOST__/$HOST/" \
                     -e "s/__PORT__/$PORT/" \
                     -e "s/__MODEL__/$MODEL/" \
                     -e "s/__CONTEXT__/$CONTEXT/" "$TEMPLATE")"

if [[ ! -f "$CONFIG_PATH" ]]; then
    printf '%s\n' "$provider_json" > "$CONFIG_PATH"
    echo "Wrote new Qwen Code settings: $CONFIG_PATH"
elif command -v jq >/dev/null 2>&1; then
    tmp="$(mktemp)"
    # Merge our provider in without disturbing anything else. The openai
    # provider list is rebuilt with our entry removed and re-added, so
    # re-running is idempotent instead of appending a duplicate every time.
    jq --argjson new "$provider_json" --arg id "$MODEL" '
        .modelProviders            = (.modelProviders // {})
      | .modelProviders.openai     = ((.modelProviders.openai // [])
                                      | map(select(.id != $id)))
                                     + $new.modelProviders.openai
      | .security                  = ((.security // {}) * $new.security)
      | .model                     = ((.model // {}) * $new.model)
    ' "$CONFIG_PATH" > "$tmp"
    cp "$CONFIG_PATH" "$CONFIG_PATH.bak"
    mv "$tmp" "$CONFIG_PATH"
    echo "Merged provider 'qwen38-bunya' into $CONFIG_PATH (backup at $CONFIG_PATH.bak)"
else
    echo
    echo "Existing settings at $CONFIG_PATH and 'jq' is not available for a safe merge."
    echo "The .env route above is already set up and is sufficient — qwen will work."
    echo "If you also want the model listed by name, merge this in by hand:"
    echo
    printf '%s\n' "$provider_json"
fi

echo
echo "Model '$MODEL' advertised with a ${CONTEXT}-token context limit."
if [[ "$NO_KEY" -eq 1 ]]; then
    echo "No key written. Supply it yourself before starting qwen:"
    echo "  export OPENAI_API_KEY=\"<key>\""
else
    echo "API key written into $ENV_PATH (mode 600)."
fi
if [[ "$HOST" == "localhost" || "$HOST" == "127.0.0.1" ]]; then
    echo
    echo "If qwen runs on a different machine than the server, keep a tunnel open:"
    echo "  ssh -N -L $PORT:<gpu-node>:$PORT \${USER}@bunya1.rcc.uq.edu.au   # <gpu-node> = bun159/160/161"
fi
echo
echo "Then start it:"
echo "  qwen"
echo
echo "If it authenticates against Qwen's cloud instead of your endpoint, the"
echo "settings block is not being read — fall back to the env-var route, which"
echo "outranks settings.json:"
echo "  ./qwencode-setup.sh --env-only --host $HOST --port $PORT"
