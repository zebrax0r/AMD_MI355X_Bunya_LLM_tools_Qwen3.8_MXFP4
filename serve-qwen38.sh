#!/usr/bin/env bash
#
# serve-qwen38.sh — one-click Qwen3.8 (2.4T-A95B, MXFP4) serving on Bunya's AMD
# MI355X nodes (bun159/160/161, 8x gfx950) via SGLang inside an Apptainer
# container.
#
# Reproduces SGLang's verified single-node MI355X MXFP4 cookbook cell
# (hw=mi355x, quant=mxfp4, strategy=balanced, nodes=single), with three
# deliberate deviations for our environment — see the README "Our deviations
# from upstream":
#   1. binds 0.0.0.0 (not 127.0.0.1) so SSH tunnels and off-node clients work
#   2. adds --api-key and --served-model-name
#   3. forces SGLANG_SET_CPU_AFFINITY=0 (SLURM cgroup; upstream has none)
#
# Usage:
#   ./serve-qwen38.sh [serve]    start the server (default; runs until killed)
#   ./serve-qwen38.sh serve --detach
#                                start the server, wait until healthy, then
#                                return the shell (server keeps running in the
#                                background for the life of the SLURM job) —
#                                use this to run a client on the GPU node itself
#   ./serve-qwen38.sh pull       build the .sif from the container image (once)
#   ./serve-qwen38.sh check      can this image load this model, and is the
#                                checkpoint even reachable? (arch vs registry,
#                                plus the MXFP4 candidate ladder)
#   ./serve-qwen38.sh gpucheck   can this image reach this node's GPUs? (~1 min)
#   ./serve-qwen38.sh toolcheck  tool-call + reasoning round trip vs a running server
#   ./serve-qwen38.sh parsers    list tool-call/reasoning parsers this image has
#   ./serve-qwen38.sh loadstat   why the last cold start was slow (reads the log)
#   ./serve-qwen38.sh download   prefetch model weights only (no GPU needed)
#   ./serve-qwen38.sh stop       stop a running server
#   ./serve-qwen38.sh status     show server state + health endpoint
#
# Run 'check' BEFORE 'download' — it is the cheap gate on a ~1.4 TB commitment,
# and right now it is also the only thing that tells you whether the checkpoint
# you have configured can be fetched at all. See the README, "The canonical
# MXFP4 checkpoint is not public yet".
#
# Apptainer lives ONLY on Bunya compute nodes, never the login nodes — so every
# mode except a bare 'stop'/'status' must run inside a salloc/sbatch allocation.
#
# Configuration comes from qwen38.env next to this script (or $QWEN38_ENV),
# see qwen38-env.example. Environment variables you export beforehand win.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="serve"
DETACH=0
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *)           MODE="$arg" ;;
    esac
done

log()  { printf '\033[1;34m[qwen38]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[qwen38 WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[qwen38 ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config ─────────────────────────────────────────────────────────────

ENV_FILE="${QWEN38_ENV:-$SCRIPT_DIR/qwen38.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    log "Loaded config from $ENV_FILE"
else
    warn "No config file at $ENV_FILE (copy qwen38-env.example to qwen38.env); using environment only."
fi

# ── Shadowed-config detector ────────────────────────────────────────────────
#
# Every line in qwen38-env.example is `export VAR="${VAR:-default}"` so that a
# shell export beats the file. That is deliberate and useful (sbatch --export,
# one-off overrides on the command line) — but it has a nasty failure mode:
#
#   $ source qwen38.env          # exports MODEL_ID=<old value> into your shell
#   $ vi qwen38.env              # change MODEL_ID
#   $ source qwen38.env          # ${MODEL_ID:-<new>} sees the OLD export...
#   $ ./serve-qwen38.sh check    # ...so your edit is silently discarded
#
# HIT FOR REAL ON BUN161, 14 AUG 2026. The edit to MODEL_ID had no effect and
# 'check' then advised setting MODEL_ID — which had already been done. A stale
# value here is expensive: it points a 1.4 TB download at the wrong repo, or
# serves a different checkpoint than the one you think you are benchmarking.
#
# So: re-evaluate the file in a CLEAN subshell, with these variables unset, and
# compare. Warn rather than override — "exported wins" is documented behaviour
# and sometimes exactly what you meant. But say so, loudly, and name the fix.
#
# NOTE you do not need to source qwen38.env to use these scripts at all; they
# read it themselves. Sourcing is only for shells that want $MODEL_CACHE_DIR.
SHADOW_VARS=(MODEL_ID MODEL_CACHE_DIR SGLANG_IMAGE WEIGHTS_GB SERVED_MODEL_NAME
             SPECULATIVE TP_SIZE CONTEXT_LEN MEM_FRACTION SIF_PATH)
SHADOWED=()

if [[ -f "$ENV_FILE" ]]; then
    unset_args=()
    for v in "${SHADOW_VARS[@]}"; do unset_args+=(-u "$v"); done
    # printf one value per line, in SHADOW_VARS order, from a shell that has
    # never seen these variables. That is what the file ALONE produces.
    file_only="$(env "${unset_args[@]}" bash -c '
        source "$1" >/dev/null 2>&1
        for v in "${@:2}"; do printf "%s\n" "${!v-}"; done
    ' _ "$ENV_FILE" "${SHADOW_VARS[@]}" 2>/dev/null || true)"

    if [[ -n "$file_only" ]]; then
        i=0
        while IFS= read -r want; do
            v="${SHADOW_VARS[$i]}"; i=$(( i + 1 ))
            got="${!v-}"
            # Only interesting when the file names a value and the live value
            # differs. A file that leaves something empty is not being shadowed.
            [[ -n "$want" && "$got" != "$want" ]] && SHADOWED+=("$v"$'\t'"$want"$'\t'"$got")
        done <<<"$file_only"
    fi
fi

if (( ${#SHADOWED[@]} )); then
    shadow_names=()
    warn "An exported variable is OVERRIDING $ENV_FILE:"
    for row in "${SHADOWED[@]}"; do
        IFS=$'\t' read -r v want got <<<"$row"
        shadow_names+=("$v")
        printf '    %-18s file: %-46s  in use: %s\n' \
            "$v" "${want:-<empty>}" "${got:-<empty>}" >&2
    done
    env_base="$(basename "$ENV_FILE")"
    warn "  If you did not mean to override these, you almost certainly sourced
  $env_base BEFORE editing it — the export then wins over every later edit,
  so the file now says one thing and the server does another. Clear them:
      unset ${shadow_names[*]}
  You do NOT need to source $env_base to run this script; it reads the file
  itself. Source it only in shells that want \$MODEL_CACHE_DIR."
fi

# The id the verified cookbook cell names. It is NOT publicly readable as of
# 14 Aug 2026 — see MODEL_CANDIDATES below and './serve-qwen38.sh check'.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8}"

# Every MXFP4 checkpoint we know of, canonical first. 'check' probes each one
# for reachability, size and declared quant_method so you can pick a working
# MODEL_ID before committing to the download rather than after.
MODEL_CANDIDATES="${MODEL_CANDIDATES:-Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4 amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 Inferact/Qwen3.8-2.4T-A95B-MXFP4}"

SGLANG_IMAGE="${SGLANG_IMAGE:-docker://lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260812}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
# EMPTY on purpose. The cookbook's MI355X cell passes no --attention-backend at
# all: trtllm_mha is SM100-only, and on gfx950 the model hook picks the right
# backend itself. Setting one here is how you quietly opt out of that.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
QUANTIZATION="${QUANTIZATION:-}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
PAGE_SIZE="${PAGE_SIZE:-}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Speculative decoding. Empty = the verified cell verbatim.
#   nextn   the checkpoint's own MTP head — no draft download
#   dspark  RadixArk's trained draft — a separate 6.6 GB repo
SPECULATIVE="${SPECULATIVE:-}"
DSPARK_MODEL="${DSPARK_MODEL:-RadixArk/Qwen3.8-2.4T-A95B-DSpark}"
# Unpinned by default, unlike the K3 repo. That pin existed because
# RadixArk/Kimi-K3-DSpark was rewritten three times in four days, twice breaking
# the launch. This draft has one revision and no history of churn yet — but the
# same failure mode applies the moment it gets one, so the plumbing is here.
DSPARK_REVISION="${DSPARK_REVISION:-}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-}"
REPLAYSSM_SPEC="${REPLAYSSM_SPEC:-0}"

# GDN (Gated DeltaNet) state pool. 69 of Qwen3.8's 92 layers are linear
# attention holding a recurrent state per request; these size and shape that
# pool. All empty by default — the argv is unchanged until one is set.
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-}"
MAMBA_RADIX_STRATEGY="${MAMBA_RADIX_STRATEGY:-}"
INT8_MAMBA_CHECKPOINT="${INT8_MAMBA_CHECKPOINT:-0}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-}"

ENABLE_AITER="${ENABLE_AITER:-1}"
# DEFAULT 0 AS OF 14 AUG 2026 (was 1). AITER_FLYDSL_FORCE is OURS, not upstream's
# — the verified MI355X cell sets SGLANG_USE_AITER=1 and nothing else, and the
# variable appears nowhere in SGLang mainline (aiter reads it). It was the
# measured fast path on Kimi K3's MoE shapes; on Qwen3.8's it is unvalidated and
# could route gemms to a JIT path with no tuned config for these shapes — the
# heuristic fallback bench-qwen38.sh warns about. Off means the first run is
# upstream's cell verbatim; turn it on as the FIRST A/B, with a measurement.
FLYDSL_FORCE="${FLYDSL_FORCE:-0}"

# ── ROCm/AITER tuning ladder — all default OFF ──────────────────────────────
# None of these are in upstream's verified cell; all are SGLang defaults-off that
# look matched to this architecture. Untested here. One at a time, re-bench each.
ROCM_MULTI_STREAM="${ROCM_MULTI_STREAM:-0}"
GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-}"
AITER_KV_LAYOUT="${AITER_KV_LAYOUT:-}"
AITER_FP8_PER_TOKEN="${AITER_FP8_PER_TOKEN:-0}"

case "$AITER_KV_LAYOUT" in
    ""|nhd|vectorized_5d) ;;
    *) die "AITER_KV_LAYOUT must be empty, nhd or vectorized_5d, got '$AITER_KV_LAYOUT'." ;;
esac

# Validated here because it is used in an arithmetic context below, where a
# non-numeric value is a hard shell error rather than a wrong answer.
[[ -z "$GPU_MAX_HW_QUEUES" || "$GPU_MAX_HW_QUEUES" =~ ^[0-9]+$ ]] \
    || die "GPU_MAX_HW_QUEUES must be a positive integer or empty, got '$GPU_MAX_HW_QUEUES'."

ROCM_MODE="${ROCM_MODE:-auto}"
AITER_GPU_ARCHS="${AITER_GPU_ARCHS:-}"
ROCMINFO_SHIM="${ROCMINFO_SHIM:-auto}"
SET_CPU_AFFINITY="${SET_CPU_AFFINITY:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-14400}"
LAUNCH_CMD="${LAUNCH_CMD:-sglang serve}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-}"
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:-}"

# aiter's get_user_jit_dir() branches on `"AITER_JIT_DIR" in os.environ` rather
# than on the variable having a VALUE, then calls os.makedirs("") and raises
# `FileNotFoundError: [Errno 2] No such file or directory: ''`. qwen38-env.example
# ships AITER_JIT_DIR empty and Apptainer passes the host environment straight
# through, so every probe that imports sglang inherits the trap. 'serve' escapes
# it only because it resolves the variable to a real path further down — long
# after 'check' and 'parsers' have run.
#
# This is not hypothetical: on a mainline image SGLang's model registry imports
# something that touches aiter for EVERY model, so all ~200 modules fail to
# import, 'check' reports "0 architectures registered", and the diagnosis it
# prints is completely wrong. A probe that cannot import is not a probe that
# answered no. The v0.5.17 mainline image this repo pins is exactly that kind.
#
# Probes need a writable path, not the real JIT cache. /tmp is writable inside an
# Apptainer container with no bind, so this works regardless of what a given exec
# mounts. 'serve' keeps using the seeded, bound AITER_JIT_DIR resolved below.
PROBE_ENV=(--env AITER_JIT_DIR=/tmp/aiter-jit)

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
SIF_PATH="${SIF_PATH:-}"
WEIGHT_LOAD_THREADS="${WEIGHT_LOAD_THREADS:-8}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
PRESHARDED_PATH="${PRESHARDED_PATH:-}"
PREFETCH_BLOCK_SIZE_MB="${PREFETCH_BLOCK_SIZE_MB:-}"

# Measured from amd/Qwen3.8-2.4T-A95B-Quark-MXFP4's file list (1372 GB); the
# Inferact checkpoint is 1597 GB and the canonical Qwen one is unmeasurable
# while it 401s. Used for the effective-GB/s readout, the free-space preflight
# and the presharded space check. 'check' prints the real number per candidate.
WEIGHTS_GB="${WEIGHTS_GB:-1372}"

case "$SPECULATIVE" in
    ""|none|nextn|dspark) ;;
    *) die "SPECULATIVE must be empty, 'nextn' or 'dspark', got '$SPECULATIVE'." ;;
esac
[[ "$SPECULATIVE" == "none" ]] && SPECULATIVE=""

case "$ROCM_MODE" in
    auto|rocm|devices) ;;
    *) die "ROCM_MODE must be auto, rocm or devices, got '$ROCM_MODE'." ;;
esac

case "$ROCMINFO_SHIM" in
    auto|off|force) ;;
    *) die "ROCMINFO_SHIM must be auto, off or force, got '$ROCMINFO_SHIM'." ;;
esac

[[ "$WEIGHT_LOAD_THREADS" =~ ^[0-9]+$ ]] \
    || die "WEIGHT_LOAD_THREADS must be a non-negative integer, got '$WEIGHT_LOAD_THREADS' (0 disables the flag)."
# WEIGHTS_GB feeds integer arithmetic (the free-space preflight) and awk (the
# GB/s readout). A non-numeric value here fails deep inside a $(( )) after the
# .sif has already been pulled, so check it up front.
[[ "$WEIGHTS_GB" =~ ^[0-9]+$ && "$WEIGHTS_GB" -gt 0 ]] \
    || die "WEIGHTS_GB must be a positive integer (GB), got '$WEIGHTS_GB'.
  './serve-qwen38.sh check' prints the real size of each checkpoint candidate."
[[ -z "$PREFETCH_BLOCK_SIZE_MB" || "$PREFETCH_BLOCK_SIZE_MB" =~ ^[0-9]+$ ]] \
    || die "PREFETCH_BLOCK_SIZE_MB must be a positive integer or empty, got '$PREFETCH_BLOCK_SIZE_MB'."

# ── GDN state pool validation ───────────────────────────────────────────────

case "$MAMBA_RADIX_STRATEGY" in
    ""|auto|extra_buffer|extra_buffer_lazy|no_buffer) ;;
    *) die "MAMBA_RADIX_STRATEGY must be empty, auto, extra_buffer, extra_buffer_lazy or no_buffer, got '$MAMBA_RADIX_STRATEGY'." ;;
esac

case "$MAMBA_SSM_DTYPE" in
    ""|bfloat16|float16|float32) ;;
    *) die "MAMBA_SSM_DTYPE must be empty, bfloat16, float16 or float32, got '$MAMBA_SSM_DTYPE'." ;;
esac

[[ -z "$MAMBA_FULL_MEMORY_RATIO" || "$MAMBA_FULL_MEMORY_RATIO" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
    || die "MAMBA_FULL_MEMORY_RATIO must be a positive number or empty, got '$MAMBA_FULL_MEMORY_RATIO'."
[[ -z "$MAX_MAMBA_CACHE_SIZE" || "$MAX_MAMBA_CACHE_SIZE" =~ ^[0-9]+$ ]] \
    || die "MAX_MAMBA_CACHE_SIZE must be a positive integer or empty, got '$MAX_MAMBA_CACHE_SIZE'."

# extra_buffer is what 'auto' resolves to for this model, and it is the strategy
# that buys prefix reuse over the mutable GDN state — but mamba_extra_buffer_of()
# requires disable_radix_cache to be false. With the radix cache off the strategy
# goes INERT and the budget drops from 5 state slots per request to 1. That is a
# 5x change in concurrency headroom, and nothing in the server log says so.
if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
    case "$MAMBA_RADIX_STRATEGY" in
        extra_buffer|extra_buffer_lazy)
            warn "MAMBA_RADIX_STRATEGY=$MAMBA_RADIX_STRATEGY is INERT while DISABLE_RADIX_CACHE=1.
  mamba_extra_buffer_of() needs the radix cache on; with it off the GDN state
  budget silently drops to 1 slot per request instead of $( [[ "$MAMBA_RADIX_STRATEGY" == extra_buffer ]] && echo 5 || echo 4 ).
  Either set DISABLE_RADIX_CACHE=0, or drop the strategy and accept 1 slot." ;;
    esac
fi

# GPU_ARCHS is a build-time hint that aiter also consults at runtime, and a
# LIST makes it pick the first entry regardless of the actual device
# (ROCm/aiter#3807). One arch or nothing.
if [[ "$AITER_GPU_ARCHS" == *";"* || "$AITER_GPU_ARCHS" == *","* ]]; then
    die "AITER_GPU_ARCHS must name exactly one architecture (e.g. gfx950), got '$AITER_GPU_ARCHS'.
  A list makes aiter select the first entry at runtime even on other hardware
  — see https://github.com/ROCm/aiter/issues/3807."
fi

if [[ -n "$PRESHARDED_PATH" && "$LOAD_FORMAT" != "presharded" ]]; then
    warn "PRESHARDED_PATH is set but LOAD_FORMAT is '${LOAD_FORMAT:-auto}' — the path will be ignored.
  Set LOAD_FORMAT=presharded to use it."
fi

# The buffered multi-thread loader holds ~(num_threads + 2) shards in host RAM at
# once. The MXFP4 shards are ~6.5 GB (1372 GB / 213), so 8 threads is ~65 GB
# against --mem=1800G. Well past that and you are trading a weight-load win for a
# host OOM.
if (( WEIGHT_LOAD_THREADS > 32 )); then
    warn "WEIGHT_LOAD_THREADS=$WEIGHT_LOAD_THREADS holds roughly $(( (WEIGHT_LOAD_THREADS + 2) * 7 )) GB of shards in host RAM.
  Check that against your --mem. Past ~16 the GPFS client, not the thread count, is usually the limit."
fi

# DSPARK-only flags. Fail here rather than after a 1.4 TB load.
if [[ -n "$DSPARK_BLOCK_SIZE" && "$SPECULATIVE" != "dspark" ]]; then
    warn "DSPARK_BLOCK_SIZE=$DSPARK_BLOCK_SIZE is ignored while SPECULATIVE is '${SPECULATIVE:-off}'."
fi

# ReplaySSM spec-verify is linear-chain only: server_args.py restricts it to
# --speculative-eagle-topk in {None, 1}. Catch a contradictory pair here, not
# after the weights are resident.
if [[ "$REPLAYSSM_SPEC" == "1" ]]; then
    [[ -n "$SPECULATIVE" ]] \
        || die "REPLAYSSM_SPEC=1 needs a speculative algorithm — the ReplaySSM ring is
  spec-verify-only scratch, and a server that never runs verify rejects the flag.
  Set SPECULATIVE=nextn or SPECULATIVE=dspark, or REPLAYSSM_SPEC=0."
    if [[ -n "$SPEC_EAGLE_TOPK" && "$SPEC_EAGLE_TOPK" != "1" ]]; then
        die "REPLAYSSM_SPEC=1 is linear-chain only: --speculative-eagle-topk must be unset
  or 1, got SPEC_EAGLE_TOPK=$SPEC_EAGLE_TOPK. SGLang rejects the pair at argument
  parse time (server_args.py, enable_linear_replayssm_spec)."
    fi
fi

# Runtime state (PID + log) lives under MODEL_CACHE_DIR so it survives detach
# and is reachable by 'stop'/'status' from any shell in the allocation.
if [[ -n "$MODEL_CACHE_DIR" ]]; then
    PID_FILE="${PID_FILE:-$MODEL_CACHE_DIR/qwen38-server.pid}"
    LOG_FILE="${LOG_FILE:-$MODEL_CACHE_DIR/qwen38-server.log}"
    LOADTIMES_FILE="${LOADTIMES_FILE:-$MODEL_CACHE_DIR/qwen38-loadtimes.log}"
else
    PID_FILE="${PID_FILE:-$SCRIPT_DIR/qwen38-server.pid}"
    LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/qwen38-server.log}"
    LOADTIMES_FILE="${LOADTIMES_FILE:-$SCRIPT_DIR/qwen38-loadtimes.log}"
fi

server_running() { [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; }

stop_server() {
    if server_running; then
        local pid; pid="$(<"$PID_FILE")"
        log "Stopping server (pid $pid) ..."
        kill "$pid" 2>/dev/null || true
        # Give SGLang's worker processes a moment, then make sure they're gone.
        for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    else
        warn "No running server recorded in $PID_FILE."
    fi
    # Belt-and-braces: this node is a single-user 8-GPU grab, so clean up strays
    # from either invocation form.
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    pkill -f 'sglang serve'         2>/dev/null || true
    rm -f "$PID_FILE"
}

# ── Simple modes first ──────────────────────────────────────────────────────

case "$MODE" in
    stop)
        stop_server
        log "Stopped."
        exit 0
        ;;
    status)
        if server_running; then
            log "Server process alive (pid $(<"$PID_FILE"))."
        else
            log "No server process recorded (pidfile $PID_FILE)."
        fi
        if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            log "Health check on port $PORT: OK"
            key=""
            [[ -r "${MODEL_CACHE_DIR:-}/qwen38-api-key" ]] && key="$(<"$MODEL_CACHE_DIR/qwen38-api-key")"
            curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" \
                 -H "Authorization: Bearer ${QWEN38_API_KEY:-$key}" 2>/dev/null | head -c 400 || true
            echo
        else
            warn "http://127.0.0.1:${PORT}/health not responding (not started, or still loading)."
            [[ -f "$LOG_FILE" ]] && log "Follow progress with: tail -f $LOG_FILE"
        fi
        exit 0
        ;;
    toolcheck)
        # Tool-call round trip AND reasoning-parser check against a RUNNING
        # server. No .sif and no GPU needed — it is pure HTTP, so it also works
        # through a tunnel.
        #
        # Half the value is in turn 2. A model emitting a plausible tool_calls
        # object proves the parser serialises; it does not prove the loop
        # closes. So the tool returns a value the model cannot guess, and we
        # check that value appears in the final answer.
        #
        # The reasoning check is the part K3 did not need. Qwen3.8 ALWAYS
        # reasons — thinking cannot be turned off, and every response opens with
        # a <think>…</think> block. If REASONING_PARSER is wrong, the server
        # still answers 200 and every reply arrives with raw <think> tags glued
        # to the front of `content`. That is not an error anywhere in the stack;
        # it just makes agentic clients behave strangely. So assert it directly.
        key=""
        [[ -r "${MODEL_CACHE_DIR:-}/qwen38-api-key" ]] && key="$(<"$MODEL_CACHE_DIR/qwen38-api-key")"
        BASE="${TOOLCHECK_URL:-http://127.0.0.1:$PORT}" \
        KEY="${QWEN38_API_KEY:-$key}" \
        MODEL="$SERVED_MODEL_NAME" \
        python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error

BASE, KEY, MODEL = os.environ["BASE"], os.environ["KEY"], os.environ["MODEL"]
SECRET = 61.4          # unguessable: only the "tool" knows it
NODE   = "bun161"

TOOLS = [{"type": "function", "function": {
    "name": "get_gpu_temperature",
    "description": "Return the current GPU temperature in Celsius for a named Bunya compute node.",
    "parameters": {"type": "object",
                   "properties": {"node": {"type": "string",
                                           "description": "Node hostname, e.g. bun161"}},
                   "required": ["node"]}}}]

def post(payload):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:400]}")
        sys.exit(1)
    except Exception as e:
        print(f"  request failed: {e}\n  Is the server up? ./serve-qwen38.sh status")
        sys.exit(1)

fails = []
def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(label)

# ── Part 1: reasoning parser ────────────────────────────────────────────────
# reasoning_effort=low keeps this cheap; the point is the SHAPE of the reply,
# not the depth of the thought.
print("Reasoning — is the <think> block being split out?")
r0 = post({"model": MODEL, "reasoning_effort": "low", "max_tokens": 2048,
           "temperature": 0,
           "messages": [{"role": "user", "content": "What is 15% of 240? Answer with the number."}]})
m0 = r0["choices"][0]["message"]
content0   = (m0.get("content") or "")
reasoning0 = (m0.get("reasoning_content") or "")

check(bool(reasoning0), "reasoning_content is populated",
      f"{len(reasoning0)} chars" if reasoning0 else "empty")
check("<think>" not in content0 and "</think>" not in content0,
      "no raw <think> tag leaked into content", repr(content0[:120]))
check("36" in content0, "answer is correct (36)", repr(content0[:120]))
if not reasoning0 or "<think>" in content0:
    print("\n  Qwen3.8 always reasons and cannot be told not to. If reasoning_content")
    print("  is empty while <think> appears in content, REASONING_PARSER is wrong.")
    print("  Check './serve-qwen38.sh parsers' — it should be 'qwen3'.")

# ── Part 2: tool calling, two turns ─────────────────────────────────────────
msgs = [{"role": "user",
         "content": f"What is the current GPU temperature on {NODE}? "
                    "Use the tool, then state the number."}]

print("\nTurn 1 — does the model emit a tool call?")
r1 = post({"model": MODEL, "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
           "max_tokens": 4096, "temperature": 0})
m1 = r1["choices"][0]["message"]
fin = r1["choices"][0]["finish_reason"]
calls = m1.get("tool_calls") or []

check(bool(calls), "model returned tool_calls",
      f"finish_reason={fin}" + ("" if calls else f", content={(m1.get('content') or '')[:120]!r}"))
if not calls:
    # A thinking model that ran out of budget mid-thought looks like a parser
    # failure but is not. Say which one it is. Qwen3.8 reasons at xhigh by
    # default, so this is the likelier of the two here.
    if fin == "length":
        print("\n  finish_reason=length: it never finished thinking. Raise max_tokens,")
        print("  or send reasoning_effort='low' / 'medium'.")
    print("\n  If content contains a raw tool call as text, the tool parser is not")
    print("  matching this model. Check './serve-qwen38.sh parsers' and TOOL_PARSER")
    print("  (it should be 'qwen3_coder').")
    sys.exit(1)

fn = calls[0].get("function", {})
check(fn.get("name") == "get_gpu_temperature", "correct function name", repr(fn.get("name")))

raw = fn.get("arguments")
try:
    args = json.loads(raw) if isinstance(raw, str) else raw
    ok_json = isinstance(args, dict)
except Exception as e:
    args, ok_json = None, False
    print(f"        arguments did not parse: {e}")
check(ok_json, "arguments are valid JSON", repr(raw)[:160])
check(bool(args) and NODE in str(args.get("node", "")), "argument value carried through",
      repr(args.get("node") if args else None))
check(bool(calls[0].get("id")), "tool_call has an id", repr(calls[0].get("id")))

print("\nTurn 2 — does the model use the tool result?")
msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls})
msgs.append({"role": "tool", "tool_call_id": calls[0].get("id"),
             "name": "get_gpu_temperature",
             "content": json.dumps({"node": NODE, "celsius": SECRET})})

r2 = post({"model": MODEL, "messages": msgs, "tools": TOOLS,
           "max_tokens": 4096, "temperature": 0})
m2 = r2["choices"][0]["message"]
final = (m2.get("content") or "").strip()
check(bool(final), "final answer has content",
      f"finish_reason={r2['choices'][0]['finish_reason']}")
check(str(SECRET) in final, f"final answer contains the tool's value ({SECRET})")
print(f"\n  final answer: {final[:300]}")

print()
if fails:
    print(f"ROUND TRIP FAILED: {len(fails)} check(s) — {', '.join(fails)}")
    sys.exit(1)
print("TOOL-CALL + REASONING ROUND TRIP OK — an agentic client should work.")
PY
        exit $?
        ;;
    serve|pull|download|check|gpucheck|parsers|loadstat) ;;
    *)
        die "Unknown mode '$MODE'. Use: serve | pull | download | check | gpucheck | toolcheck | parsers | loadstat | stop | status"
        ;;
esac

# ── Preflight (shared) ──────────────────────────────────────────────────────

command -v apptainer >/dev/null 2>&1 \
    || die "apptainer not found — run this INSIDE a compute-node allocation (salloc/sbatch). Apptainer is not installed on Bunya login nodes."
command -v curl >/dev/null 2>&1 || die "curl not found on PATH."

[[ -n "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR is not set. Point it at scratch (e.g. /scratch/user/\$USER/qwen38/hf-cache). See qwen38-env.example."
mkdir -p "$MODEL_CACHE_DIR" 2>/dev/null || true
[[ -d "$MODEL_CACHE_DIR" && -w "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR '$MODEL_CACHE_DIR' does not exist or is not writable."

SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/qwen38-mi355x.sif}"
# SIF_PATH must name a .sif FILE, not a directory. If it points at a directory
# (or ends with '/'), treat it as a folder and drop the default filename in —
# 'apptainer pull' otherwise refuses ("Image file already exists").
if [[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]]; then
    SIF_PATH="${SIF_PATH%/}/qwen38-mi355x.sif"
    log "SIF_PATH was a directory — using $SIF_PATH"
fi

# Keep Apptainer's cache and scratch off /home (which has a tight quota) — point
# them at scratch. Set both APPTAINER_* and the SINGULARITY_* aliases.
: "${APPTAINER_CACHEDIR:=$MODEL_CACHE_DIR/../apptainer-cache}"
: "${APPTAINER_TMPDIR:=$MODEL_CACHE_DIR/../apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" 2>/dev/null || true
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

# Resolve HF token early: 'check' needs it to tell "gated" from "does not exist".
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
    [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi

# ── Building the .sif ───────────────────────────────────────────────────────

# Apptainer 1.5.0 (6 May 2026) started wrapping mksquashfs in a bundled `proot`
# so that an unprivileged build preserves the original owners and groups of files
# coming out of an OCI registry. proot works by ptrace, so on a host that refuses
# ptrace(PTRACE_TRACEME) — yama ptrace_scope=3, a seccomp filter, some SELinux
# policies — the pull dies with:
#
#   proot error: ptrace(TRACEME): Operation not permitted
#   ... mksquashfs command failed: exit status 1
#
# Apptainer 1.5.3 (21 Jul 2026) turned that into an INFO message and carries on,
# so this only bites the 1.5.0–1.5.2 window. The escape hatch is the hidden
# `--ignore-proot` build flag, which drops back to exactly the pre-1.5.0
# behaviour: ownership inside the SIF is not preserved. That costs us nothing —
# we mount the image read-only and run as ourselves — and it is how every
# unprivileged Apptainer build worked before May 2026.
#
# Note --ignore-proot is registered on `build`, NOT on `pull`, so the retry has
# to switch subcommand. `build <sif> docker://…` is otherwise equivalent.
pull_image() {
    local logf rc
    logf="$(mktemp "${APPTAINER_TMPDIR:-/tmp}/qwen38-pull.XXXXXX.log")"
    log "Pulling $SGLANG_IMAGE -> $SIF_PATH (~24 GB compressed; one-time) ..."

    # tee, not command substitution: this downloads tens of GB and the progress
    # bar has to stay on screen. PIPESTATUS[0] to get apptainer's status rather
    # than tee's — and read via if/else, because a trailing `|| true` would run a
    # new command and reset PIPESTATUS to 0, i.e. silently swallow every failure.
    if apptainer pull "$SIF_PATH" "$SGLANG_IMAGE" 2>&1 | tee "$logf"; then rc=0; else rc=${PIPESTATUS[0]}; fi
    if (( rc == 0 )); then
        rm -f "$logf"
        log "Image ready: $SIF_PATH"
        return 0
    fi

    # A failed pull can leave a truncated .sif behind, and every later check in
    # this script is just [[ -f ]]. Clear it on any failure so the next run
    # retries instead of reporting "already present" and dying hours later.
    rm -f "$SIF_PATH"

    if ! grep -qi 'proot\|ptrace' "$logf"; then
        rm -f "$logf"
        return "$rc"
    fi

    warn "Apptainer could not use proot on this host (ptrace is not permitted here)."
    warn "Retrying with --ignore-proot; the image is identical apart from file"
    warn "ownership inside it, which we do not rely on."
    if PROOT_NO_SECCOMP=1 apptainer build --ignore-proot "$SIF_PATH" "$SGLANG_IMAGE" 2>&1 | tee "$logf"
    then rc=0; else rc=${PIPESTATUS[0]}; fi
    if (( rc == 0 )); then
        rm -f "$logf"
        log "Image ready: $SIF_PATH (built with --ignore-proot)."
        return 0
    fi
    rm -f "$SIF_PATH"

    if grep -qi 'unknown flag' "$logf"; then
        rm -f "$logf"
        die "This Apptainer predates --ignore-proot (added in 1.5.0) yet failed inside
  proot, which should not happen. Report it with the output above.
  Workaround: build the .sif on a host that allows ptrace, then copy it to
  $SIF_PATH and re-run."
    fi
    rm -f "$logf"
    return "$rc"
}

# ── Pull mode: build the .sif from the container image ──────────────────────

if [[ "$MODE" == "pull" ]]; then
    if [[ -f "$SIF_PATH" ]]; then
        log "Image already present: $SIF_PATH (delete it to re-pull)."
        exit 0
    fi
    pull_image \
        || die "apptainer pull failed. Does this node have outbound internet? Check APPTAINER_CACHEDIR space ($APPTAINER_CACHEDIR).
     The default image is the mi35x (gfx950, ROCm 7.20) build. Do NOT substitute
     the mi30x one — it targets CDNA3 and a different ROCm, and aiter's kernels
     will not load. See the README 'Upstream drift' for how to find a current tag."
    exit 0
fi

# For every remaining mode we need the .sif. Auto-build it if missing.
if [[ ! -f "$SIF_PATH" ]]; then
    log "No .sif at $SIF_PATH yet — pulling it now (one-time)."
    pull_image \
        || die "apptainer pull failed. Run './serve-qwen38.sh pull' explicitly to debug."
fi

# ── GPU passthrough: how does the container reach the GPUs? ─────────────────
#
# The image ships its own ROCm. The HOST's ROCm reaches into it through exactly
# three channels, and all three have broken a run on these nodes:
#
#   1. '--rocm' binds the host's ROCm libraries into /.singularity.d/libs and
#      PREPENDS that to LD_LIBRARY_PATH, so the container's binaries run against
#      the host's libhsa/libamdhip. Apptainer's docs are explicit that this
#      requires the two ROCm versions to be compatible. When RCC moved a node to
#      ROCm 7.14 under a 7.2 image, the container's OWN rocminfo started
#      exiting 1 and aiter died on import:
#        RuntimeError: Get GPU arch from rocminfo failed:
#          Command '['/opt/rocm-7.2.0/bin/rocminfo']' returned non-zero exit status 1
#      This image is also ROCm 7.2.0 (the 'rocm720' in the tag), so the same
#      mismatch applies.
#   2. The kernel driver, via /dev/kfd. A KFD ioctl ABI break is not fixable
#      from here — it needs an image built for the node's ROCm.
#   3. Inherited environment: a *_VISIBLE_DEVICES value the container's ROCr
#      cannot parse (e.g. UUID form) takes down agent enumeration entirely.
#
# So don't assume: probe. 'devices' mode drops --rocm and passes the device
# nodes only, leaving the container's ROCm userspace intact end to end.

ROCM_MODES=(rocm devices)

# SLURM on a newer ROCm can hand out UUID-form device lists ("GPU-a1b2..."). An
# older ROCr in the container cannot parse those, and an unparseable value takes
# down agent enumeration for the WHOLE container — which looks exactly like a
# broken driver. Only forward index-form lists.
visible_devices_ok() { [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; }

# Identity of the .sif, cheap. Used both to key the cached passthrough mode and
# to detect that the seeded aiter jit/ dir came from a different image.
sif_stamp() {
    stat -c '%s:%Y' "$SIF_PATH" 2>/dev/null \
        || stat -f '%z:%m' "$SIF_PATH" 2>/dev/null \
        || echo '?'
}

# Probing costs ~a minute, so remember the answer. Key it to the node AND the
# image: either one changing invalidates the result.
ROCM_MODE_CACHE="$MODEL_CACHE_DIR/.rocm-mode"
rocm_cache_key() { printf '%s|%s' "$(hostname -s 2>/dev/null || echo node)" "$(sif_stamp)"; }

# ── rocminfo shim ───────────────────────────────────────────────────────────
#
# On a node whose ROCm is newer than the image's, the container's rocminfo
# loads, reads the driver version, then fails:
#
#   ROCk module version 6.19.14.31400000 is loaded
#   hsa api call failure at: .../rocminfo.cc:357
#   Call returned HSA_STATUS_ERROR_INVALID_ARGUMENT
#
# ...while torch sees all 8 GPUs, because PyTorch's ROCm wheel bundles its own
# HIP runtime. So the image's ROCm *tools* are broken on this node and its
# *runtime* is fine. aiter only shells out to rocminfo to read the architecture
# out of its text — so give it text that is correct: a snapshot of the HOST's
# working rocminfo, replayed by a one-line script bound over the container's
# binary.
#
# This is exact rather than synthesised: it is real output for this node, so
# whatever aiter's parser expects, it gets. Nothing links against it and no
# host libraries are involved.

make_rocminfo_shim() {   # -> sets global shim_args
    shim_args=()
    [[ "$ROCMINFO_SHIM" == "off" ]] && return 0

    local target snap shim
    target="$(apptainer exec "$SIF_PATH" \
        sh -c 'ls -d /opt/rocm*/bin/rocminfo 2>/dev/null | head -n1' 2>/dev/null || true)"
    if [[ -z "$target" ]]; then
        [[ "$ROCMINFO_SHIM" == "force" ]] \
            && warn "ROCMINFO_SHIM=force but no rocminfo found in the image — skipping."
        return 0
    fi

    # Only step in when the container's own rocminfo is actually broken.
    if [[ "$ROCMINFO_SHIM" != "force" ]] \
       && apptainer exec --rocm "$SIF_PATH" "$target" >/dev/null 2>&1; then
        return 0
    fi

    if ! command -v rocminfo >/dev/null 2>&1; then
        warn "The container's rocminfo fails and the host has none to copy — aiter will not
  be able to detect the GPU architecture."
        return 0
    fi

    snap="$MODEL_CACHE_DIR/rocminfo-host.txt"
    if ! rocminfo > "$snap" 2>/dev/null || ! grep -q 'gfx' "$snap"; then
        rm -f "$snap"
        warn "The host's rocminfo did not produce usable output — cannot shim."
        return 0
    fi

    # Embed the snapshot in the script so the bind is self-contained.
    shim="$MODEL_CACHE_DIR/rocminfo-shim.sh"
    {
        printf '#!/bin/sh\ncat <<'\''QWEN38_ROCMINFO_EOF'\''\n'
        cat "$snap"
        printf 'QWEN38_ROCMINFO_EOF\n'
    } > "$shim"
    chmod +x "$shim"

    shim_args=(--bind "$shim":"$target")
    log "rocminfo shim: the image's rocminfo fails on this node; binding the host's output over $target"
}

set_rocm_mode_args() {   # $1 = mode -> sets global rocm_args
    case "$1" in
        rocm)    rocm_args=(--rocm) ;;
        devices) rocm_args=(--bind /dev/kfd:/dev/kfd --bind /dev/dri:/dev/dri) ;;
        *)       die "internal: unknown ROCm mode '$1'" ;;
    esac
}

# ROCm version string, best effort. /opt/rocm/.info/version is the canonical
# file; fall back to the versioned directory name.
ROCM_VER_CMD='cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm-* 2>/dev/null | sed -n "1s|.*/opt/rocm-||p"'
sif_rocm_ver()  { apptainer exec "$SIF_PATH" sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1; }
host_rocm_ver() {
    local v
    v="$(sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1 || true)"
    # Bunya installs ROCm from a module tree, not /opt, so fall back to the
    # version embedded in rocminfo's own path (/…/rocm/7.14.0/bin/rocminfo).
    [[ -z "$v" ]] && v="$(command -v rocminfo 2>/dev/null \
        | grep -o '[0-9][0-9]*\.[0-9][0-9]*\(\.[0-9][0-9]*\)\?' | tail -n1 || true)"
    [[ -z "$v" && -n "${ROCM_PATH:-}" ]] && v="${ROCM_PATH##*/}"
    printf '%s' "$v"
}

# Ask the container, under a given mode, the two questions that matter: can
# torch see the GPUs, and can aiter name the architecture? The second is the
# exact call that crashes — testing anything less is testing the wrong thing.
#
# aiter prints '[aiter] ...' banners on import, so answers go out as tagged
# lines and are grepped back rather than read positionally.
GPU_PROBE_PY='
n, gfx, err = -1, "", ""
try:
    import torch
    n = torch.cuda.device_count()
except BaseException as e:
    err = "torch: %s: %s" % (type(e).__name__, e)
if not err:
    try:
        from aiter.jit.utils.chip_info import get_gfx
        gfx = str(get_gfx() or "")
    except BaseException as e:
        err = "aiter: %s: %s" % (type(e).__name__, e)
print("QWEN38_DEVICES %d" % n)
print("QWEN38_GFX %s" % (gfx or "-"))
print("QWEN38_ERR %s" % (" ".join(err.split()) or "-"))
'

# gpu_probe <mode> [extra apptainer args ...]
# Sets PROBE_DEVICES / PROBE_GFX / PROBE_ERR. Returns 0 only if the container
# saw at least one GPU and aiter named the architecture.
gpu_probe() {
    local mode="$1"; shift
    local out
    set_rocm_mode_args "$mode"
    out="$(apptainer exec "${rocm_args[@]}" "$@" "$SIF_PATH" \
              python3 -c "$GPU_PROBE_PY" 2>&1 || true)"

    PROBE_DEVICES="$(sed -n 's/^QWEN38_DEVICES //p' <<<"$out" | tail -n1)"
    PROBE_GFX="$(sed -n 's/^QWEN38_GFX //p' <<<"$out" | tail -n1)"
    PROBE_ERR="$(sed -n 's/^QWEN38_ERR //p' <<<"$out" | tail -n1)"

    # No tagged output at all means python never ran (bad bind, missing device,
    # apptainer refused). Surface whatever it did say.
    if [[ -z "$PROBE_DEVICES" ]]; then
        PROBE_DEVICES="-1"
        PROBE_GFX="-"
        PROBE_ERR="$(tail -n 3 <<<"$out" | tr '\n' ' ')"
        PROBE_ERR="${PROBE_ERR:-container produced no output}"
    fi

    [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt 0 \
       && "$PROBE_GFX" == gfx* ]]
}

# ── gpucheck mode: can this image reach this node's GPUs? ───────────────────
# The diagnostic AND the permanent preflight — same code path, so what we debug
# with is what we run. Costs about a minute; the failure it replaces costs a
# crash 30 frames deep in aiter, or worse, a 1.4 TB weight load.

if [[ "$MODE" == "gpucheck" ]]; then
    log "Image:  $SIF_PATH"
    echo
    printf '  host ROCm       : %s\n' "$(host_rocm_ver || true)"
    printf '  container ROCm  : %s\n' "$(sif_rocm_ver || true)"
    host_gfx=""
    if command -v rocminfo >/dev/null 2>&1; then
        # grep -m1 would exit early and SIGPIPE rocminfo, which under
        # 'set -o pipefail' reads as a failed pipeline. Take the head instead.
        host_gfx="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -n1 || true)"
        printf '  host rocminfo       : %s\n' "${host_gfx:-ran, but printed no gfx line}"
    else
        printf '  host rocminfo       : not on PATH\n'
    fi
    for v in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do
        if [[ -n "${!v:-}" ]]; then
            note=""
            visible_devices_ok "${!v}" \
                || note="   <- NOT index-form; ROCr may fail to enumerate"
            printf '  %-20s: %s%s\n' "$v" "${!v}" "$note"
        fi
    done
    echo

    # Probe with the same aiter jit/ bind the server gets, when we already know
    # it — the .seeded marker records the in-image target. Testing a container
    # configured differently from the one we launch is how a read-only-.sif bug
    # survives several attempts.
    probe_extra=()
    jit_dir="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit}"
    if [[ -f "$jit_dir/.seeded" ]]; then
        jit_target="$(cut -d'|' -f2 "$jit_dir/.seeded" 2>/dev/null || true)"
        if [[ "$jit_target" == /* ]]; then
            probe_extra=(--bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR"
                         --bind "$jit_dir":"$jit_target")
            log "Probing with the aiter JIT bind: $jit_dir -> $jit_target"
        fi
    fi
    # Probe with the same GPU_ARCHS the server would get, so this mode can
    # actually verify the workaround it recommends.
    if [[ -n "$AITER_GPU_ARCHS" ]]; then
        probe_extra+=(--env "GPU_ARCHS=$AITER_GPU_ARCHS")
        log "Probing with GPU_ARCHS=$AITER_GPU_ARCHS"
    fi
    make_rocminfo_shim
    probe_extra+=(${shim_args[@]+"${shim_args[@]}"})
    echo

    # What does the image's rocminfo actually say? aiter only reports the exit
    # status, which hides the reason.
    printf '  in-container rocminfo:\n'
    apptainer exec --rocm "$SIF_PATH" sh -c \
        'rocminfo 2>&1 | head -n 12 || true' 2>&1 | sed 's/^/    /' || true
    echo

    gpucheck_ok=""
    best_mode=""; best_devices=0
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" ${probe_extra[@]+"${probe_extra[@]}"}; then
            printf '  %-8s : OK    devices=%s gfx=%s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX"
            [[ -z "$gpucheck_ok" ]] && gpucheck_ok="$m"
        else
            printf '  %-8s : FAIL  devices=%s gfx=%s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX"
            printf '             %s\n' "$PROBE_ERR"
        fi
        # Track the best result separately: "torch saw the GPUs but aiter could
        # not name the architecture" is a completely different problem from
        # "the container cannot reach the GPUs", and only the second one needs
        # a new image. Reporting both as one failure sends you down a
        # multi-hour rebuild for a broken rocminfo.
        if [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt "$best_devices" ]]; then
            best_devices="$PROBE_DEVICES"; best_mode="$m"
        fi
    done
    echo

    if [[ -n "$gpucheck_ok" ]]; then
        log "Verdict: use ROCM_MODE=$gpucheck_ok on this node."
        [[ "$gpucheck_ok" != "rocm" ]] && log \
            "  '--rocm' injects the host's ROCm libraries; '$gpucheck_ok' does not, which is
  what a host/container ROCm mismatch needs."
        [[ "$PROBE_DEVICES" != "$TP_SIZE" ]] && warn \
            "Container sees $PROBE_DEVICES GPU(s) but TP_SIZE=$TP_SIZE. Fix the allocation or TP_SIZE."
        # Record it so 'serve' does not pay for the probe again.
        printf '%s %s\n' "$(rocm_cache_key)" "$gpucheck_ok" > "$ROCM_MODE_CACHE" 2>/dev/null || true
        exit 0
    fi

    rm -f "$ROCM_MODE_CACHE" 2>/dev/null || true

    if [[ "$best_devices" -gt 0 ]]; then
        warn "Verdict: the GPUs ARE reachable ($best_devices via '$best_mode') — only aiter's
  architecture detection is broken. It shells out to the image's rocminfo, and
  that binary fails on this node even though HIP is fine. This does NOT need a
  new image."
        if [[ ${#shim_args[@]} -gt 0 ]]; then
            warn "  The rocminfo shim was already applied and aiter still could not read the
  architecture, so it is not parsing rocminfo the way we assumed. Send the
  output of:
      apptainer exec $SIF_PATH \\
          sed -n '1,90p' /sgl-workspace/aiter/aiter/jit/utils/chip_info.py"
        else
            warn "  No shim was applied. Set ROCMINFO_SHIM=force in $ENV_FILE to replay the
  host's rocminfo output inside the container, then re-run gpucheck."
        fi
        exit 1
    fi

    warn "Verdict: the container cannot reach the GPUs in any mode (torch saw none)."
    warn "  The node's kernel driver is newer than the container's ROCm and no bind
  fixes it — you need an image built for this node's ROCm. See the README
  'When the node's ROCm changes'.
    host ROCm $(host_rocm_ver || echo '?')  vs  container ROCm $(sif_rocm_ver || echo '?')"
    exit 1
fi

# ── loadstat mode: why was the last cold start slow? ────────────────────────

if [[ "$MODE" == "loadstat" ]]; then
    log "Weight loading report from $LOG_FILE"
    echo

    if [[ ! -r "$LOG_FILE" ]]; then
        warn "No server log at $LOG_FILE — start the server once, then re-run this."
    else
        if grep -qi "falling back to single-threaded" "$LOG_FILE"; then
            warn "SINGLE-THREADED weight loading was used. This is the sawtooth:"
            grep -i -m1 "falling back to single-threaded" "$LOG_FILE" | sed 's/^/    /'
            echo
            log "Fix: set WEIGHT_LOAD_THREADS=8 in qwen38.env and restart."
        else
            log "No single-threaded fallback warning found."
        fi

        echo
        log "Loader flags on the last launch (from the recorded argv):"
        grep -m1 -oE '\-\-load-format [^ ]+|\-\-model-loader-extra-config [^ ]+' "$LOG_FILE" \
            | sed 's/^/    /' || echo "    (none — upstream defaults)"
    fi

    # The server log is truncated on every launch; this history is not, so it is
    # the only place an A/B between runs survives.
    echo
    log "Time-to-ready history ($LOADTIMES_FILE):"
    if [[ -r "$LOADTIMES_FILE" ]]; then
        tail -10 "$LOADTIMES_FILE" | sed 's/^/    /'
        echo
        log "Compare COLD runs only. Host RAM is 1800 GB and the weights are ~1372 GB,"
        log "so a restart on the same node is served largely from page cache and will"
        log "look fast whatever the thread count is set to."
    else
        echo "    (nothing recorded yet — it is written when the ready-wait sees /health)"
    fi

    echo
    log "Load formats this image supports:"
    apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" \
        bash -c "sglang serve --help 2>&1 || python3 -m sglang.launch_server --help 2>&1" 2>/dev/null \
        | grep -A 12 -- '--load-format' | head -20 | sed 's/^/    /' \
        || warn "Could not read --load-format choices from the image."

    echo
    log "Bunya is GPFS, not Lustre — there is no 'lfs setstripe' here, and \$TMPDIR is"
    log "the same filesystem, so staging there buys nothing. The lever is thread count."
    exit 0
fi

# ── parsers mode: what can this image actually parse? ───────────────────────

if [[ "$MODE" == "parsers" ]]; then
    log "Parsers available in $SIF_PATH:"
    apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" \
        bash -c "sglang serve --help 2>&1 || python3 -m sglang.launch_server --help 2>&1" \
        | grep -A 6 -iE '\-\-(tool-call|reasoning)-parser' || \
        warn "Could not read parser choices from --help."
    echo
    log "Qwen3.8 uses TOOL_PARSER=qwen3_coder and REASONING_PARSER=qwen3."
    log "Reasoning CANNOT be disabled on this model — every reply opens with a"
    log "<think> block, so the reasoning parser is not optional in practice."
    log "Set both in qwen38.env from this list ('none' to omit)."
    exit 0
fi

# ── check mode: is the checkpoint reachable, and can this image load it? ────
# The cheap gate before committing to a ~1.4 TB download, and — while the
# canonical MXFP4 repo is not public — the only thing that tells you whether
# your configured MODEL_ID can be fetched at all.

if [[ "$MODE" == "check" ]]; then
    log "Image:   $SIF_PATH"
    log "Model:   $MODEL_ID"
    echo

    # Part 1: the candidate ladder. Pure HTTP against the hub, run inside the
    # container only so it uses the image's huggingface_hub and token handling.
    log "MXFP4 checkpoint candidates — reachability, size and declared quant_method:"
    echo
    apptainer exec \
        --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env MODEL_ID="$MODEL_ID" \
        --env MODEL_CANDIDATES="$MODEL_CANDIDATES" \
        "${PROBE_ENV[@]}" \
        "$SIF_PATH" python3 - <<'PY' || true
import json, os, urllib.request, urllib.error

token = os.environ.get("HF_TOKEN") or ""
current = os.environ["MODEL_ID"]
cands = os.environ["MODEL_CANDIDATES"].split()
if current not in cands:
    cands.insert(0, current)

# SGLang keys its MoE-runner auto-resolution off the checkpoint's own
# quant_method, so that string is the thing that decides whether you get the
# fused MXFP4 path or something slower. Map it to the SGLang name here.
SGLANG_QUANT = {
    "mxfp4":  "mxfp4        (MoE-only; the auto-resolution target)",
    "quark":  "quark        (AMD Quark; FP8/MXFP4/Int4FP8)",
    "fp8":    "fp8          (NOT MXFP4 — this is the 2.4 TB checkpoint)",
    "compressed-tensors": "compressed-tensors",
}

def api(path):
    req = urllib.request.Request("https://huggingface.co/api/" + path)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

for m in cands:
    mark = "->" if m == current else "  "
    try:
        d = api(f"models/{m}?blobs=true")
    except urllib.error.HTTPError as e:
        why = {401: "UNREACHABLE (401 — private, gated, or does not exist"
                    + (" — your HF_TOKEN did not open it)" if token else "; no HF_TOKEN was sent)"),
               403: "FORBIDDEN (403 — gated; accept the licence on the model page)",
               404: "NOT FOUND (404)"}.get(e.code, f"HTTP {e.code}")
        print(f"  {mark} {m}")
        print(f"       {why}")
        continue
    except Exception as e:
        print(f"  {mark} {m}\n       probe failed: {e}")
        continue

    sibs = d.get("siblings") or []
    total = sum(s.get("size") or 0 for s in sibs)
    shards = sum(1 for s in sibs if s["rfilename"].endswith(".safetensors"))
    cfg = d.get("config") or {}
    qm = ((cfg.get("quantization_config") or {}).get("quant_method")) or "?"
    print(f"  {mark} {m}")
    print(f"       OK   {total/1e9:8.1f} GB   {shards:4d} shards   "
          f"gated={d.get('gated')}   quant_method={qm}")
    print(f"            -> SGLang quantization: {SGLANG_QUANT.get(qm, qm + '   (unrecognised here)')}")
    archs = cfg.get("architectures") or []
    if archs:
        print(f"            -> architectures: {archs}")

print()
print("  '->' marks your configured MODEL_ID.")
PY

    echo
    # Part 2: can the image's SGLang actually load this architecture?
    # Don't let 'set -e' swallow the exit code: 1 = arch unsupported,
    # 2 = config unreadable, 3 = the probe itself broke. All meaningful.
    log "Can this image's SGLang load $MODEL_ID?"
    echo
    rc=0
    apptainer exec \
        --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env MODEL_ID="$MODEL_ID" \
        "${PROBE_ENV[@]}" \
        "$SIF_PATH" python3 - <<'PY' || rc=$?
import os, sys

model_id = os.environ["MODEL_ID"]

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    print(f"  config.json architectures : {archs}")
    text = getattr(cfg, "text_config", None)
    src = text if text is not None else cfg
    for attr in ("num_hidden_layers", "full_attention_interval", "num_experts",
                 "num_experts_per_tok", "max_position_embeddings",
                 "quantization_config"):
        if hasattr(src, attr):
            v = getattr(src, attr)
            if attr == "quantization_config" and isinstance(v, dict):
                v = v.get("quant_method", "?")
            print(f"  {attr:26}: {v}")
    lt = getattr(src, "layer_types", None)
    if lt:
        from collections import Counter
        c = Counter(lt)
        print(f"  {'layer_types':26}: {dict(c)}  ({len(lt)} layers)")
except Exception as e:
    print(f"  !! could not load config for {model_id}: {e}")
    print("     Gated, private, or no network from this node? The candidate table")
    print("     above says which of those it is.")
    sys.exit(2)

print()
from sglang.srt.models.registry import ModelRegistry  # type: ignore
known = set(getattr(ModelRegistry, "models", {}) or {})

missing = []
for a in archs:
    if a in known:
        print(f"  OK   SGLang registry knows '{a}'")
    else:
        print(f"  MISS SGLang registry does NOT know '{a}'")
        missing.append(a)

print(f"\n  ({len(known)} architectures registered in this image)")
qwen = sorted(a for a in known if "qwen3_5" in a.lower() or "qwen35" in a.lower())
print(f"  Qwen3.5-family architectures present: {qwen or '<none>'}")

# An empty registry is a BROKEN PROBE, not a negative answer, and the two look
# identical from here if you only test `arch in known`. Every model module
# failing to import — the aiter empty-AITER_JIT_DIR trap does exactly this —
# leaves len(known) == 0 and reports every architecture as unsupported. Say
# "could not tell" instead, and name the thing to look for.
if not known:
    print("\n  => INCONCLUSIVE: SGLang's registry came back EMPTY, so nothing was")
    print("     tested. Every model module failed to import — look for repeated")
    print("     'Ignore import error when loading sglang.srt.models.*' above.")
    print("     A trailing \": ''\" on those lines is the aiter AITER_JIT_DIR trap;")
    print("     see the README troubleshooting entry for FileNotFoundError: ''.")
    sys.exit(3)

if missing:
    print("\n  => This image CANNOT serve this model. Qwen3.8 landed with day-0")
    print("     support in v0.5.17, so a v0.5.17-rocm720-mi35x-* image should carry")
    print("     Qwen3_5MoeForCausalLM — check you are not on an older tag.")
    sys.exit(1)
print("\n  => This image can load this model. Safe to download the weights.")
PY
    case "$rc" in
        0) log "check passed." ;;
        1) warn "check FAILED: this image cannot serve $MODEL_ID." ;;
        2) # Do not advise "set MODEL_ID in the config file" when the config file
           # ALREADY sets it and an export is winning — that advice sends you to
           # re-do something you have already done. Diagnose the shadow instead.
           # ${SHADOWED[@]+...} so an empty array is safe under `set -u` on the
           # older bash some images still ship.
           if printf '%s\n' ${SHADOWED[@]+"${SHADOWED[@]}"} | grep -q '^MODEL_ID	'; then
               warn "check INCONCLUSIVE: could not read the model config — and the MODEL_ID
  it tried is NOT the one $ENV_FILE sets. An exported MODEL_ID is overriding
  the file (see the warning above). Run:
      unset MODEL_ID && ./serve-qwen38.sh check"
           else
               warn "check INCONCLUSIVE: could not read the model config.
  If MODEL_ID showed UNREACHABLE or NOT FOUND in the table above, set MODEL_ID in
  $ENV_FILE to one of the candidates that reported OK, set WEIGHTS_GB to match,
  then re-run check in a shell where MODEL_ID is not exported."
           fi ;;
        3) warn "check INCONCLUSIVE: the image's model registry would not import,
  so this says nothing about whether it can serve $MODEL_ID." ;;
        *) warn "check exited with status $rc." ;;
    esac
    exit "$rc"
fi

# ── Weights cache accounting ────────────────────────────────────────────────

# HF hub layout: models--org--name
weights_dir="$MODEL_CACHE_DIR/hub/models--${MODEL_ID//\//--}"
weights_cached=0
[[ -d "$weights_dir/snapshots" ]] && weights_cached=1

EST_GB="$WEIGHTS_GB"
NEED_GB=$(( WEIGHTS_GB + 150 ))

if [[ "$weights_cached" -eq 0 ]]; then
    [[ -n "${HF_TOKEN:-}" ]] \
        || die "No HF_TOKEN / HF_TOKEN_FILE set and weights for $MODEL_ID are not cached yet in $MODEL_CACHE_DIR."
    free_gb="$(df -Pk "$MODEL_CACHE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')"
    if [[ "${free_gb:-0}" -lt "$NEED_GB" ]]; then
        warn "Only ${free_gb} GB free in $MODEL_CACHE_DIR; $MODEL_ID is ~${EST_GB} GB (${NEED_GB} GB recommended with the .sif). Download will likely fail."
        warn "  Check your quota with 'rquota' before starting a multi-hour transfer."
    fi
    log "Weights not cached — first start downloads ~${EST_GB} GB. Run './serve-qwen38.sh download' first."
else
    log "Found cached weights for $MODEL_ID."
fi

# The DSpark draft is a SEPARATE checkpoint, and nothing above covers it. Without
# this check a missing draft surfaces as
#   RuntimeError: Cannot find any model weights with `RadixArk/Qwen3.8-2.4T-A95B-DSpark`
# raised from the draft worker — which the scheduler only builds AFTER the main
# model has finished loading. That is the whole 1.4 TB thrown away to learn a
# second repo was never fetched.
if [[ "$SPECULATIVE" == "dspark" && "$MODE" == "serve" ]]; then
    draft_dir="$MODEL_CACHE_DIR/hub/models--${DSPARK_MODEL//\//--}"
    if [[ ! -d "$draft_dir/snapshots" ]]; then
        die "SPECULATIVE=dspark but the draft model '$DSPARK_MODEL' is not cached in $MODEL_CACHE_DIR.
  The draft loads only after the main model finishes, so starting now would waste that whole load.
  Fetch it first:   ./serve-qwen38.sh download
  Or serve without speculative decoding:   unset SPECULATIVE"
    fi

    # Resolve the snapshot the SAME WAY the hub does — refs/<branch> -> sha —
    # not "newest directory by mtime". Those differ, and comparing the wrong one
    # validates a good snapshot while SGLang reads the one refs/main points at:
    # the preflight passes and the launch still dies. A cache can hold several
    # snapshots at once; only one of them is what a bare repo id resolves to.
    #
    # A 40-hex revision names a snapshot directly; anything else is a branch or
    # tag, which the cache stores as refs/<name> -> sha.
    draft_pin="tracking main"
    if [[ "$DSPARK_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
        draft_pin="pinned"
        draft_snapshot="$draft_dir/snapshots/$DSPARK_REVISION"
        [[ -d "$draft_snapshot" ]] \
            || die "DSPARK_REVISION=$DSPARK_REVISION is not in the cache.
  Fetch it first:   DSPARK_REVISION=$DSPARK_REVISION ./serve-qwen38.sh download
  Or track the branch instead:   DSPARK_REVISION=main"
    else
        [[ -n "$DSPARK_REVISION" ]] && draft_pin="tracking $DSPARK_REVISION"
        draft_ref="$draft_dir/refs/${DSPARK_REVISION:-main}"
        if [[ -r "$draft_ref" ]]; then
            draft_snapshot="$draft_dir/snapshots/$(<"$draft_ref")"
        else
            draft_snapshot="$(ls -1dt "$draft_dir"/snapshots/*/ 2>/dev/null | head -1)"
            draft_snapshot="${draft_snapshot%/}"
        fi
    fi

    # A present directory is not a usable checkpoint. SGLang resolves the draft's
    # config during ARGUMENT PARSING, before anything loads, and a config without
    # model_type surfaces as the unhelpful
    #   ValueError: Unrecognized model in RadixArk/Qwen3.8-2.4T-A95B-DSpark.
    #               Should have a `model_type` key in its config.json
    # Check it here instead, where we can say what to do about it.
    if [[ -z "$draft_snapshot" || ! -d "$draft_snapshot" ]]; then
        die "Could not resolve a draft snapshot under $draft_dir/snapshots.
  Re-fetch it:   rm -rf $draft_dir && ./serve-qwen38.sh download"
    fi
    # -e not -r: these are symlinks into blobs/, and a dangling one is exactly
    # what a half-finished download leaves behind.
    if [[ ! -e "$draft_snapshot/config.json" ]]; then
        die "The draft snapshot $draft_snapshot has no config.json (or it is a dangling symlink).
  The download is incomplete. Re-fetch:   rm -rf $draft_dir && ./serve-qwen38.sh download
  Or serve without speculative decoding:  unset SPECULATIVE"
    fi
    # Three fields out of one parse: model_type is the thing SGLang dies without,
    # and the rope type plus the draft's own trained context are what decide
    # whether this checkpoint can be trusted at the CONTEXT_LEN being served.
    # Read rope_parameters (transformers 5.x) or rope_scaling (4.x).
    draft_cfg="$(python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
rp = c.get('rope_parameters') or c.get('rope_scaling') or {}
print(c.get('model_type') or '')
print(rp.get('rope_type') or rp.get('type') or '')
print(rp.get('original_max_position_embeddings') or c.get('max_position_embeddings') or '')
" "$draft_snapshot/config.json" 2>/dev/null || true)"
    { read -r draft_model_type
      read -r DRAFT_ROPE_TYPE
      read -r DRAFT_TRAINED_CTX
    } <<<"$draft_cfg"

    if [[ -z "${draft_model_type:-}" ]]; then
        die "The draft config at $draft_snapshot/config.json is unparseable or has no 'model_type',
  so SGLang cannot resolve the speculative algorithm and dies during argument parsing.
  Either the download is truncated, or upstream changed the repo under you.

  Re-fetch it:                 rm -rf $draft_dir && ./serve-qwen38.sh download
  Pin a known-good revision:   DSPARK_REVISION=<sha> ./serve-qwen38.sh download
  Or serve without it:         unset SPECULATIVE"
    fi

    # Hand SGLang the resolved PATH, never the repo id. The repo id makes it
    # re-resolve through the hub, which is how it ends up reading a different
    # snapshot than the one checked here.
    log "Draft: $DSPARK_MODEL -> $(basename "$draft_snapshot") ($draft_pin)"
    DSPARK_MODEL="$draft_snapshot"
fi

# ── Download mode (no GPU required) ─────────────────────────────────────────

if [[ "$MODE" == "download" ]]; then
    log "Prefetching $MODEL_ID into $MODEL_CACHE_DIR (~${EST_GB} GB, no GPU required) ..."
    apptainer exec \
        --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        --env HF_HUB_ENABLE_HF_TRANSFER=1 \
        "$SIF_PATH" \
        bash -c "hf download '$MODEL_ID' || huggingface-cli download '$MODEL_ID'" \
        || die "Weights download failed. Re-run to resume.
  If it failed with 401/403, run './serve-qwen38.sh check' — the candidate table
  says whether this repo is reachable with your token at all."

    if [[ "$SPECULATIVE" == "dspark" ]]; then
        rev_arg=""
        [[ -n "$DSPARK_REVISION" ]] && rev_arg=" --revision '$DSPARK_REVISION'"
        log "Prefetching the DSpark draft model $DSPARK_MODEL${DSPARK_REVISION:+ @ $DSPARK_REVISION} ..."
        apptainer exec \
            --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
            --env HF_HOME="$MODEL_CACHE_DIR" \
            --env HF_TOKEN="${HF_TOKEN:-}" \
            --env HF_HUB_ENABLE_HF_TRANSFER=1 \
            "$SIF_PATH" \
            bash -c "hf download '$DSPARK_MODEL'$rev_arg || huggingface-cli download '$DSPARK_MODEL'$rev_arg" \
            || die "DSpark draft download failed."
    fi
    log "Download complete."
    exit 0
fi

# ── Serve-mode preflight (GPU node checks) ──────────────────────────────────

[[ -e /dev/kfd ]] || die "/dev/kfd not found — is this a ROCm GPU node? (Apptainer + --rocm needs it.)"
[[ -e /dev/dri ]] || die "/dev/dri not found — is this a ROCm GPU node?"

if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]*' || true)"
    case "$gfx" in
        gfx950) log "Detected gfx950 (MI350X/MI355X) — matches the configured MXFP4 image/model." ;;
        gfx942) warn "Detected gfx942 (MI300X/MI325X). CDNA3 has no hardware MX matmul, so it
        cannot run the MXFP4 checkpoint at all — MI300X serves FP8 across TWO nodes.
        See the README 'Running on other hardware'." ;;
        "")     warn "Could not detect GPU arch from rocminfo." ;;
        *)      warn "Detected $gfx — this recipe is validated for gfx950 (MI355X/MI350X)." ;;
    esac
fi

# The image ships its own ROCm and the node has its own. They do not have to
# match, but a MAJOR difference is the thing most likely to break passthrough —
# say so up front rather than letting it surface as an aiter import error.
host_rocm="$(host_rocm_ver || true)"
sif_rocm="$(sif_rocm_ver || true)"
if [[ -n "$host_rocm" && -n "$sif_rocm" ]]; then
    log "ROCm: node $host_rocm / container $sif_rocm"
    if [[ "$(cut -d. -f1,2 <<<"$host_rocm")" != "$(cut -d. -f1,2 <<<"$sif_rocm")" ]]; then
        warn "Node and container ROCm versions differ ($host_rocm vs $sif_rocm).
  '--rocm' injects the NODE's ROCm libraries into the container, which is what
  breaks first when they diverge. ROCM_MODE=$ROCM_MODE will sort it out; run
  './serve-qwen38.sh gpucheck' if startup fails in aiter."
    fi
fi

# --tp IS the total GPU count. --dp (with dp-attention) only subdivides those
# GPUs for attention — it does NOT multiply the count — so DP must divide TP.
GPUS_USED="$TP_SIZE"
if [[ "${DP_SIZE:-1}" -gt 1 && $(( TP_SIZE % DP_SIZE )) -ne 0 ]]; then
    die "DP_SIZE=$DP_SIZE must divide TP_SIZE=$TP_SIZE (dp-attention splits the $TP_SIZE GPUs into $DP_SIZE groups)."
fi

GPU_VIS="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
alloc_count=""
if [[ -n "$GPU_VIS" ]]; then
    alloc_count="$(awk -F, '{print NF}' <<<"$GPU_VIS")"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    alloc_count="$SLURM_GPUS_ON_NODE"
fi
[[ -n "$GPU_VIS" ]] && log "Allocated GPUs: [$GPU_VIS]"
if [[ -n "$alloc_count" && "$alloc_count" -gt 0 && "$GPUS_USED" -ne "$alloc_count" ]]; then
    warn "TP_SIZE=$TP_SIZE uses $GPUS_USED GPU(s) but $alloc_count are allocated. Qwen3.8 MXFP4 needs all 8 (~1.4 TB of weights); set TP_SIZE=$alloc_count."
fi

# Port free?
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    exec 3>&- 3<&- || true
    die "Port $PORT is already in use on this node (another server running? try './serve-qwen38.sh status')."
fi

if server_running; then
    die "A server is already recorded running (pid $(<"$PID_FILE")). Use './serve-qwen38.sh status' or 'stop' first."
fi

# ── API key ─────────────────────────────────────────────────────────────────

API_KEY_FILE="$MODEL_CACHE_DIR/qwen38-api-key"
if [[ -z "${QWEN38_API_KEY:-}" ]]; then
    if [[ -r "$API_KEY_FILE" ]]; then
        QWEN38_API_KEY="$(<"$API_KEY_FILE")"
        log "Using API key from $API_KEY_FILE"
    else
        QWEN38_API_KEY="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        (umask 077 && printf '%s' "$QWEN38_API_KEY" > "$API_KEY_FILE")
        log "Generated new API key and saved it to $API_KEY_FILE"
    fi
fi

# ── aiter JIT directory bind ────────────────────────────────────────────────
# The MXFP4 MoE JIT-compiles FlyDSL kernels at CUDA-graph capture time and
# writes them INSIDE the image, under aiter/jit/. An Apptainer .sif is
# read-only, so that write dies with:
#
#   OSError: [Errno 30] Read-only file system:
#     '/sgl-workspace/aiter/aiter/jit/flydsl_cache/launch_hgemm_kernel_*/*.lock'
#
# ...and it dies *after* the ~1.4 TB weight load, which is an expensive way to
# find out. Fix: give the container a writable aiter/jit.
#
# We bind the whole jit/ directory, not just jit/flydsl_cache, because the
# image ships prebuilt modules there (module_rmsnorm_quant.so and friends) and
# aiter writes build artefacts beside the FlyDSL cache. Binding an empty dir
# over it would hide the prebuilt kernels, so seed the scratch copy from the
# image once, then bind it back at the SAME path (compiled artefacts can embed
# absolute paths) and reuse it on every later run.

AITER_JIT_DIR="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit}"

# Ask the image where aiter lives. Use find_spec, NOT 'import aiter': importing
# it needs a GPU (this exec has no --rocm) and prints '[aiter] import [...]'
# banners to stdout that would corrupt the captured path.
detected_jit="$(apptainer exec "$SIF_PATH" python3 - <<'PY' 2>/dev/null | tail -n 1
import importlib.util, os
spec = importlib.util.find_spec("aiter")
origin = getattr(spec, "origin", None) if spec is not None else None
print(os.path.join(os.path.dirname(origin), "jit") if origin else "")
PY
)"

if [[ -n "${AITER_JIT_TARGET:-}" ]]; then
    # A manual override must name a directory INSIDE the image. Without this
    # check a host path here binds somewhere harmless, the real path stays
    # read-only, and you find out after the full weight load.
    apptainer exec "$SIF_PATH" test -d "$AITER_JIT_TARGET" 2>/dev/null \
        || die "AITER_JIT_TARGET='$AITER_JIT_TARGET' does not exist inside the image.
  It must be a path in the CONTAINER (e.g. /sgl-workspace/aiter/aiter/jit),
  not a host directory. Detected value: ${detected_jit:-<detection failed>}
  Unset it in $ENV_FILE to use auto-detection."
    [[ -n "$detected_jit" && "$AITER_JIT_TARGET" != "$detected_jit" ]] \
        && warn "AITER_JIT_TARGET='$AITER_JIT_TARGET' overrides detected '$detected_jit'."
else
    AITER_JIT_TARGET="$detected_jit"
fi

cache_bind=()
if [[ "${AITER_JIT_TARGET:-}" != /* ]]; then
    warn "Could not locate aiter's jit/ directory in the image (got: '${AITER_JIT_TARGET:-}')."
    warn "  Startup will likely die at CUDA-graph capture with"
    warn "  \"Read-only file system: .../aiter/jit/flydsl_cache/...\"."
    warn "  Set AITER_JIT_TARGET in qwen38.env to the jit/ directory from that path."
else
    mkdir -p "$AITER_JIT_DIR" || die "Cannot create $AITER_JIT_DIR"

    # The seed marker records WHICH image it came from. A jit/ dir seeded from
    # one image and bound over another hides that image's prebuilt kernels
    # behind stale ones — the 'module_*.so: undefined symbol' crash. The seed is
    # derived data, so re-deriving it is always safe.
    seed_stamp="$SGLANG_IMAGE|$AITER_JIT_TARGET|$(sif_stamp)"
    if [[ "$(cat "$AITER_JIT_DIR/.seeded" 2>/dev/null || true)" != "$seed_stamp" ]]; then
        if [[ -e "$AITER_JIT_DIR/.seeded" ]]; then
            log "Image changed since $AITER_JIT_DIR was seeded — re-seeding."
            # Guard the rm: this must never be able to point at scratch itself.
            case "$AITER_JIT_DIR" in
                ""|/|"$HOME"|"$MODEL_CACHE_DIR")
                    die "Refusing to clear AITER_JIT_DIR='$AITER_JIT_DIR' — set it to a directory of its own." ;;
            esac
            rm -rf "${AITER_JIT_DIR:?}"/* "${AITER_JIT_DIR:?}"/.[!.]* 2>/dev/null || true
        else
            log "Seeding writable aiter JIT dir from the image (one-off copy) ..."
        fi
        apptainer exec --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" "$SIF_PATH" \
            cp -a "$AITER_JIT_TARGET/." "$AITER_JIT_DIR/" \
            || die "Failed to copy $AITER_JIT_TARGET out of the image into $AITER_JIT_DIR"
        printf '%s\n' "$seed_stamp" > "$AITER_JIT_DIR/.seeded"
        log "Seeded $AITER_JIT_DIR ($(du -sh "$AITER_JIT_DIR" 2>/dev/null | cut -f1 || echo '?'))"
    fi

    cache_bind=(--bind "$AITER_JIT_DIR":"$AITER_JIT_TARGET")
    log "aiter JIT dir: $AITER_JIT_DIR -> $AITER_JIT_TARGET"
fi
# The writability preflight lives just before the launch, so it can run against
# the exact argv the server gets — see "Preflight" below.

# ── Build the launch command ────────────────────────────────────────────────

# Ask the image what flags it actually has, rather than assuming. Finding out
# the other way costs a full 1.4 TB load before the server dies on an unknown
# argument.
#
# Memoised: several callers below ask about different flags, and shelling into
# the image once per question is a second each for the same answer. Empty means
# the probe itself failed, which is not the same as "the flag is absent" — every
# caller has to tell those two apart.
IMAGE_HELP=""
IMAGE_HELP_PROBED=0
image_help() {
    if (( ! IMAGE_HELP_PROBED )); then
        IMAGE_HELP_PROBED=1
        IMAGE_HELP="$(apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" \
            bash -c "sglang serve --help 2>&1 || python3 -m sglang.launch_server --help 2>&1" 2>/dev/null || true)"
    fi
    printf '%s' "$IMAGE_HELP"
}

# has_flag <--flag>: 0 = present, 1 = absent, 2 = could not tell.
has_flag() {
    local h; h="$(image_help)"
    [[ -z "$h" ]] && return 2
    grep -q -- "$1" <<<"$h"
}

# Which HIP sampling bindings does this image have? A speculative verify step
# reaches for top_k_renorm_prob / top_p_renorm_prob, and on ROCm those are bound
# in dflash_utils.py by an is_hip() branch that arrived in two halves (sglang
# #32621 for top_p, #32641 for top_k). An image built before those has NEITHER,
# both names are None, and calling one is
#   TypeError: 'NoneType' object is not callable
# inside the scheduler's event loop. That kills the SERVER, not the request
# (sglang #32569).
#
# THIS MATTERS MORE FOR QWEN3.8 THAN IT DID FOR KIMI K3. Qwen's own recommended
# sampling for this model is temperature=1.0, top_p=0.95, top_k=20 — it sends
# BOTH of the affected parameters and a non-zero temperature by default. On K3
# you had to go out of your way to trip this; here the documented settings do it.
#
# Grep for the alias names rather than infer from the image tag: the tag is a
# build date, the aliases are the thing that actually has to be there. Textual,
# not an import — importing sglang.srt.speculative drags in torch, which is slow
# and unhappy outside an allocation. Empty output means "could not tell", which
# is not the same as "absent"; the caller distinguishes them.
#
# The same probe answers a second, worse question — see the `v`/`t` markers below
# and sglang #33694. Markers: k, p (renorm kernels), v (the HIP branch claims
# non-greedy verify), t (#33694 applied), probed (the file was read at all).
RENORM_BINDINGS=""
RENORM_PROBED=0
renorm_bindings() {
    if (( ! RENORM_PROBED )); then
        RENORM_PROBED=1
        RENORM_BINDINGS="$(apptainer exec "$SIF_PATH" bash -c '
            f=/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py
            # Search only image-owned roots. A bare `find /` here would walk the
            # bound MODEL_CACHE_DIR — 1.4 TB of weights — to find a 30 KB file.
            [[ -r "$f" ]] || f=$(find /sgl-workspace /opt /usr/lib /usr/local \
                -name dflash_utils.py -path "*sglang/srt/speculative*" 2>/dev/null | head -1)
            [[ -n "$f" && -r "$f" ]] || exit 0
            grep -q top_k_renorm_probs_triton "$f" && echo k
            grep -q top_p_renorm_probs_triton "$f" && echo p
            # The SECOND landmine, and a much commoner trigger — sglang #33694.
            # An `elif is_hip():` branch set _DFLASH_SAMPLING_VERIFY_AVAILABLE
            # = True without ever binding tree_speculative_sampling_target_only,
            # and that kernel does not exist on ROCm at all. Any request with
            # temperature > 0 then took the non-greedy verify path and died on
            # the unbound name. Grep the BRANCH, not the file: the call site is
            # present either way, so a bare grep for the symbol proves nothing.
            # #33694 fixed it by binding the name to None inside the branch.
            hip_branch=$(awk "/^elif is_hip\(\):/ {h=1; next} h && /^[^ \t]/ {h=0} h" "$f")
            if [[ -n "$hip_branch" ]]; then
                grep -q "_DFLASH_SAMPLING_VERIFY_AVAILABLE[[:space:]]*=[[:space:]]*True" \
                    <<<"$hip_branch" && echo v
                grep -q "tree_speculative_sampling_target_only" <<<"$hip_branch" && echo t
            fi
            echo probed
        ' 2>/dev/null || true)"
    fi
    printf '%s' "$RENORM_BINDINGS"
}

# Markers are one per line and matched WHOLE-LINE on purpose. Substring tests
# here are a trap: `probed` contains a `p`, so a *p* glob reports "top_p is fine"
# for the image where nothing is fine.
renorm_has() { grep -qxF "$1" <<<"$(renorm_bindings)"; }

# shellcheck disable=SC2206  # intentional word splitting of the configured launcher
launcher=($LAUNCH_CMD)
# shellcheck disable=SC2206  # intentional word splitting of user-provided extra args
extra_args=($EXTRA_ENGINE_ARGS)

dp_flag=()
[[ "${DP_SIZE:-1}" -gt 1 ]] && dp_flag=(--dp "$DP_SIZE" --enable-dp-attention)

ctx_flag=()
[[ -n "$CONTEXT_LEN" ]] && ctx_flag=(--context-length "$CONTEXT_LEN")

# EMPTY = no flag. The verified MI355X cell passes no --attention-backend, and
# on gfx950 the model hook chooses for itself. This is deliberately NOT defaulted
# to 'triton' the way the K3 recipe was.
attn_flag=()
[[ -n "$ATTENTION_BACKEND" ]] && attn_flag=(--attention-backend "$ATTENTION_BACKEND")

# EMPTY = no flag, which is what the cookbook prescribes: the MoE runner resolves
# from the checkpoint's own quant_method. Set this only to override a checkpoint
# whose declared method SGLang picks wrongly.
quant_flag=()
[[ -n "$QUANTIZATION" ]] && quant_flag=(--quantization "$QUANTIZATION")

cgraph_flag=()
[[ -n "$CUDA_GRAPH_MAX_BS_DECODE" ]] && cgraph_flag=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE")

kv_flag=()
[[ -n "$KV_CACHE_DTYPE" ]] && kv_flag=(--kv-cache-dtype "$KV_CACHE_DTYPE")

page_flag=()
[[ -n "$PAGE_SIZE" ]] && page_flag=(--page-size "$PAGE_SIZE")

radix_flag=()
[[ "$DISABLE_RADIX_CACHE" == "1" ]] && radix_flag=(--disable-radix-cache)

parser_flags=()
[[ "$TOOL_PARSER" != "none" && -n "$TOOL_PARSER" ]] \
    && parser_flags+=(--tool-call-parser "$TOOL_PARSER")
[[ "$REASONING_PARSER" != "none" && -n "$REASONING_PARSER" ]] \
    && parser_flags+=(--reasoning-parser "$REASONING_PARSER")

# ── Speculative decoding ────────────────────────────────────────────────────

spec_flags=()
if [[ -n "$SPECULATIVE" ]]; then
    case "$SPECULATIVE" in
        nextn)
            spec_flags=(--speculative-algorithm NEXTN)
            log "NEXTN speculative decoding ENABLED (MTP weights ship inside the checkpoint — no draft model)."
            ;;
        dspark)
            spec_flags=(--speculative-algorithm DSPARK
                        --speculative-draft-model-path "$DSPARK_MODEL")
            [[ -n "$DSPARK_BLOCK_SIZE" ]] \
                && spec_flags+=(--speculative-dspark-block-size "$DSPARK_BLOCK_SIZE")
            log "DSpark speculative decoding ENABLED (draft: $DSPARK_MODEL)."
            warn "  DSpark is NOT part of the verified MI355X matrix — upstream offers it only
  on GB300 FP8/NVFP4/BF16 and B300 NVFP4. Nothing here is measured. Treat any
  number you get from it as new information, not as a confirmation."
            ;;
    esac

    # The 3/1/4 preset fills itself in for NEXTN; these exist so you can move off
    # it deliberately. Empty = leave SGLang's choice alone.
    [[ -n "$SPEC_NUM_STEPS" ]]        && spec_flags+=(--speculative-num-steps "$SPEC_NUM_STEPS")
    [[ -n "$SPEC_EAGLE_TOPK" ]]       && spec_flags+=(--speculative-eagle-topk "$SPEC_EAGLE_TOPK")
    [[ -n "$SPEC_NUM_DRAFT_TOKENS" ]] && spec_flags+=(--speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS")

    # THE MRR-48 TRAP. A speculative cell with no --max-running-requests takes 48
    # from the speculative hook rather than a memory-derived ceiling. Nothing
    # errors; you simply serve a fraction of what the node can hold, and the
    # throughput number that comes out of the bench is wrong for a reason no log
    # line mentions. Upstream documents this on the cookbook page.
    if [[ -z "$MAX_RUNNING_REQUESTS" ]]; then
        warn "  MAX_RUNNING_REQUESTS is unset with speculative decoding ON. The speculative
  hook then pins --max-running-requests to 48 instead of deriving it from
  memory — so concurrency is capped at 48 whatever the GDN state pool could
  actually admit. Set MAX_RUNNING_REQUESTS in $ENV_FILE to serve more."
    fi

    # ReplaySSM spec-verify: replaces the recurrent verify's per-draft full-state
    # snapshots with a per-slot raw-input ring, so the GDN state pool admits more
    # concurrent requests at a given --mamba-full-memory-ratio. Verify output is
    # bitwise unchanged — it is a memory tradeoff, not an accuracy one.
    #
    # NOTE THE FLAG NAME: --enable-linear-replayssm-spec. There is a separate
    # --enable-linear-replayssm (buffered DECODE, not verify) which is mutually
    # exclusive and requires --mamba-radix-cache-strategy no_buffer. Do not
    # "simplify" one into the other.
    if [[ "$REPLAYSSM_SPEC" == "1" ]]; then
        # Capture the status explicitly rather than reading $? in an elif —
        # has_flag has three answers (present / absent / could-not-tell) and
        # they must not collapse into two.
        rss_rc=0; has_flag '--enable-linear-replayssm-spec' || rss_rc=$?
        if (( rss_rc == 0 )); then
            spec_flags+=(--enable-linear-replayssm-spec)
            log "  ReplaySSM spec-verify ENABLED (--enable-linear-replayssm-spec)."
        elif (( rss_rc == 2 )); then
            warn "  Could not read --help from the image; passing --enable-linear-replayssm-spec unverified."
            spec_flags+=(--enable-linear-replayssm-spec)
        else
            die "This image has no --enable-linear-replayssm-spec. It is present in mainline
  server_args.py, so an image that lacks it predates the feature — pull a newer
  v0.5.17-rocm720-mi35x-* tag into a SECOND SIF_PATH and test there, or set
  REPLAYSSM_SPEC=0."
        fi
    fi

    # The sampling landmine. This does NOT fail at startup, which is what makes
    # it worth a warning here: the server comes up healthy and serves correctly
    # until the first request that trips it, then the SCHEDULER dies and takes
    # every concurrent request with it.
    if ! renorm_has probed; then
        warn "  Could not read the image's sampling bindings; cannot tell whether"
        warn "  top_p/top_k requests are safe with speculative decoding on (sglang #32569)."
    elif renorm_has k && renorm_has p; then
        log "  Sampling bindings OK: this image has the HIP top_k and top_p renorm kernels."
    else
        warn "  This image is missing HIP renorm kernels the speculative verify step needs."
        if renorm_has p; then
            warn "  top_p works here; a request setting TOP_K raises NameError and KILLS"
            warn "  THE SERVER — not just that request."
        else
            warn "  A request setting TOP_P (<1.0) or TOP_K raises"
            warn "  \"TypeError: 'NoneType' object is not callable\" and KILLS THE SERVER —"
            warn "  not just that request. One such request poisons its whole batch."
        fi
        warn "  READ THIS TWICE FOR QWEN3.8: Qwen's own recommended sampling is"
        warn "  temperature=1.0, top_p=0.95, top_k=20 — the documented settings send"
        warn "  BOTH affected parameters. Unlike K3, you do not have to go looking"
        warn "  for this bug; a normal client finds it on the first request."
        warn "  Real fix: a newer v0.5.17-rocm720-mi35x-* image, tested in a SECOND"
        warn "  SIF_PATH first. Zero-risk: unset SPECULATIVE."
    fi

    # The second landmine, and the one a real client is far likelier to step on:
    # top_p/top_k are optional, temperature > 0 is what every chat client sends —
    # and it is Qwen's recommended default here.
    if renorm_has v && ! renorm_has t; then
        warn "  This image's is_hip() branch claims non-greedy verify is available but"
        warn "  never binds tree_speculative_sampling_target_only — the kernel does not"
        warn "  exist on ROCm at all. A request with TEMPERATURE > 0 then raises"
        warn "  \"NameError: tree_speculative_sampling_target_only\" and KILLS THE SERVER"
        warn "  (sglang #33694, fixed 6 Aug 2026). Qwen3.8's recommended temperature is"
        warn "  1.0, so this fires on essentially every real request."
    fi

    # A draft has its own trained context, and past it the accept rate does not
    # degrade — it collapses. This draft declares 262,144, which matches the
    # model's native window, so unlike K3's 4k draft there is no built-in cliff.
    # Warn only if that stops being true.
    if [[ "$SPECULATIVE" == "dspark" && "${DRAFT_ROPE_TYPE:-}" == "default" \
          && "${DRAFT_TRAINED_CTX:-}" =~ ^[0-9]+$ ]]; then
        # Serving past the draft's trained window is the risk. An empty
        # CONTEXT_LEN means "the model's own 262,144", which is >= any draft
        # window we would be warning about, so it counts as past it too.
        if [[ -z "$CONTEXT_LEN" ]] \
           || { [[ "$CONTEXT_LEN" =~ ^[0-9]+$ ]] && (( CONTEXT_LEN > DRAFT_TRAINED_CTX )); }; then
            warn "  This draft has UNSCALED RoPE (\"rope_type\": \"default\") and a trained"
            warn "  window of $DRAFT_TRAINED_CTX tokens, while you are serving ctx=${CONTEXT_LEN:-model max}."
            warn "  Past the draft's trained window the accept rate does not degrade, it"
            warn "  COLLAPSES, and speculative decoding becomes a net LOSS. Measure with"
            warn "  './bench-qwen38.sh longcontext' before trusting it in an agentic session."
        fi
    fi
fi

# ── GDN state pool ──────────────────────────────────────────────────────────
#
# 69 of Qwen3.8's 92 layers are Gated DeltaNet, and their recurrent state lives
# in its own pool. That pool — not KV — is usually what caps concurrency. Every
# one of these is empty by default, so the argv is unchanged until you set one.
mamba_flags=()
[[ -n "$MAMBA_FULL_MEMORY_RATIO" ]] && mamba_flags+=(--mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO")
[[ -n "$MAX_MAMBA_CACHE_SIZE" ]]    && mamba_flags+=(--max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE")
[[ -n "$MAMBA_SSM_DTYPE" ]]         && mamba_flags+=(--mamba-ssm-dtype "$MAMBA_SSM_DTYPE")
[[ -n "$MAMBA_RADIX_STRATEGY" ]]    && mamba_flags+=(--mamba-radix-cache-strategy "$MAMBA_RADIX_STRATEGY")
[[ -n "$LINEAR_ATTN_BACKEND" ]]     && mamba_flags+=(--linear-attn-backend "$LINEAR_ATTN_BACKEND")
[[ "$INT8_MAMBA_CHECKPOINT" == "1" ]] && mamba_flags+=(--enable-int8-mamba-checkpoint)

if (( ${#mamba_flags[@]} )); then
    unknown_mamba=()
    for f in "${mamba_flags[@]}"; do
        [[ "$f" == --* ]] || continue
        grep -q -- "$f" <<<"$(image_help)" || unknown_mamba+=("$f")
    done
    if [[ -n "$(image_help)" ]] && (( ${#unknown_mamba[@]} )); then
        die "This image does not offer: ${unknown_mamba[*]}
  These are hybrid linear-attention flags; an older image will not have them.
  Unset the matching variables in qwen38.env, or use a newer image."
    fi
    log "GDN state pool: ${mamba_flags[*]}"
fi

# --max-mamba-cache-size has to match the slots-per-request ratio in force, or it
# silently clamps max_running_requests to a fraction of the target. State slots
# per request: --disable-radix-cache 1 | no_buffer 3 | extra_buffer 5 (this
# model's 'auto'), or 4 where PP disables the overlap scheduler.
if [[ -n "$MAX_MAMBA_CACHE_SIZE" ]]; then
    if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then slots=1
    elif [[ "$MAMBA_RADIX_STRATEGY" == "no_buffer" ]]; then slots=3
    elif [[ "$MAMBA_RADIX_STRATEGY" == "extra_buffer_lazy" ]]; then slots=4
    else slots=5
    fi
    log "  MAX_MAMBA_CACHE_SIZE=$MAX_MAMBA_CACHE_SIZE at $slots slot(s)/request => about $(( MAX_MAMBA_CACHE_SIZE / slots )) concurrent requests."
    if [[ -n "$MAX_RUNNING_REQUESTS" && $(( MAX_MAMBA_CACHE_SIZE / slots )) -lt "$MAX_RUNNING_REQUESTS" ]]; then
        warn "  That is BELOW MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS — the state pool, not the
  scheduler, will be what limits you. Raise MAX_MAMBA_CACHE_SIZE to at least
  $(( MAX_RUNNING_REQUESTS * slots )), or leave it empty and let
  --mamba-full-memory-ratio size the pool."
    fi
fi

perf_flags=()
[[ -n "$CHUNKED_PREFILL_SIZE" ]] && perf_flags+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
[[ -n "$MAX_RUNNING_REQUESTS" ]] && perf_flags+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
if [[ -n "$SCHEDULE_POLICY" ]]; then
    perf_flags+=(--schedule-policy "$SCHEDULE_POLICY")
    [[ "$DISABLE_RADIX_CACHE" == "1" ]] \
        && warn "SCHEDULE_POLICY=$SCHEDULE_POLICY has little effect while DISABLE_RADIX_CACHE=1 (prefix reuse is off)."
fi

# ── aiter ───────────────────────────────────────────────────────────────────
# SGLANG_USE_AITER=1 is the ONE env var the verified MI355X cell sets. This is
# where the gfx950 MXFP4 MoE speed comes from.
aiter_env=()

# GPU_ARCHS lets aiter skip its rocminfo shell-out when that call is the only
# broken part of the stack. It is NOT a fix for a GPU the container cannot
# reach: if torch sees no devices this changes nothing. Deliberately outside the
# ENABLE_AITER gate — sglang imports aiter during module import regardless, so
# ENABLE_AITER=0 does not avoid the detection path.
if [[ -n "$AITER_GPU_ARCHS" ]]; then
    aiter_env+=(--env "GPU_ARCHS=$AITER_GPU_ARCHS")
    log "AITER_GPU_ARCHS=$AITER_GPU_ARCHS — aiter will skip rocminfo architecture detection."
fi

if [[ "$ENABLE_AITER" == "1" ]]; then
    aiter_env+=(--env SGLANG_USE_AITER=1)
    # AITER_FLYDSL_FORCE is what routes gemms through the FlyDSL JIT compiler,
    # which writes into aiter/jit/flydsl_cache inside the read-only .sif. Set
    # FLYDSL_FORCE=0 to fall back to aiter's prebuilt gemm path: slower, but it
    # compiles nothing and so cannot hit the read-only failure.
    #
    # NOTE this is NOT part of the upstream MI355X cell, which sets only
    # SGLANG_USE_AITER=1. It is carried over from the K3 recipe because the
    # read-only-.sif failure it guards is a property of Apptainer, not of the
    # model. If you are chasing a numerics difference against upstream, this is
    # the first variable to drop.
    if [[ "$FLYDSL_FORCE" == "1" ]]; then
        aiter_env+=(--env AITER_FLYDSL_FORCE=1)
        log "AITER_FLYDSL_FORCE=1 — gemms routed through the FlyDSL JIT compiler."
        log "  This is NOT in upstream's MI355X cell. A/B it against FLYDSL_FORCE=0"
        log "  and keep whichever the bench actually prefers on these shapes."
    fi

    # The rest of the ladder. Each is a plain env var an older image simply
    # ignores, which is the safe direction — there is nothing to capability-probe.
    if [[ "$ROCM_MULTI_STREAM" == "1" ]]; then
        aiter_env+=(--env SGLANG_ROCM_USE_MULTI_STREAM=1)
        # Upstream's own note: "Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue
        # serialization." Without it the dual streams serialise on the hardware
        # queue and the feature is a no-op at best.
        if [[ -z "$GPU_MAX_HW_QUEUES" ]]; then
            GPU_MAX_HW_QUEUES=5
            log "ROCM_MULTI_STREAM=1 — setting GPU_MAX_HW_QUEUES=5 (required, else the"
            log "  two MoE streams serialise on one hardware queue and it buys nothing)."
        elif (( GPU_MAX_HW_QUEUES < 5 )); then
            warn "ROCM_MULTI_STREAM=1 needs GPU_MAX_HW_QUEUES>=5, got $GPU_MAX_HW_QUEUES.
  The shared-expert and routed-expert streams will serialise on one hardware
  queue, so dual-stream MoE will not help. Raise it or set ROCM_MULTI_STREAM=0."
        fi
        log "Dual-stream MoE ENABLED (shared vs routed experts). Qwen3.8 runs 1 shared"
        log "  + 10 routed per token, which is the shape this targets. UNMEASURED here."
    fi
    [[ -n "$GPU_MAX_HW_QUEUES" ]] && aiter_env+=(--env "GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES")

    if [[ -n "$AITER_KV_LAYOUT" ]]; then
        aiter_env+=(--env "SGLANG_AITER_KV_CACHE_LAYOUT=$AITER_KV_LAYOUT")
        log "AITER KV cache layout: $AITER_KV_LAYOUT (default is nhd)."
        [[ "$AITER_KV_LAYOUT" == "vectorized_5d" ]] && log \
            "  vectorized_5d is the SHUFFLE layout pa_decode_gluon consumes natively,
  so full-attention decode avoids runtime permutes. Qwen3.8 has 23 such
  layers (64 Q over 4 KV heads). UNMEASURED here."
    fi

    if [[ "$AITER_FP8_PER_TOKEN" == "1" ]]; then
        aiter_env+=(--env SGLANG_USE_AITER_FP8_PER_TOKEN=1)
        log "AITER per-token FP8 ENABLED — this checkpoint is hybrid (MXFP4 experts,"
        log "  FP8 attention/dense), so it is the FP8 half this touches. UNMEASURED."
    fi
else
    warn "ENABLE_AITER=0 — SGLANG_USE_AITER is NOT set. This drops the one environment
  variable the verified MI355X cell requires; expect the MXFP4 MoE to run far
  below its tuned speed. Use this only to isolate an aiter crash."
fi

# Forward the SLURM GPU-visibility vars into the container (Apptainer inherits
# host env by default, but be explicit) so ROCm sees exactly the allocated GPUs.
#
# ...but only if the container's ROCr can parse them. A newer host stack can
# hand out UUID-form lists ("GPU-a1b2..."), and an older ROCr given one of those
# fails to enumerate ANY agent — which presents as a dead driver rather than as
# a bad variable. SLURM's cgroup already limits which devices are visible, so
# dropping an unparseable value is safe.
gpu_env=()
for v in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do
    val="${!v:-}"
    [[ -z "$val" ]] && continue
    if visible_devices_ok "$val"; then
        gpu_env+=(--env "$v=$val")
    else
        warn "$v='$val' is not an index list — NOT forwarding it into the container.
  The container's ROCm may not parse this form, and an unparseable value stops
  it enumerating any GPU at all. The SLURM allocation still constrains what is
  visible. Export a plain index list (e.g. 0,1,2,3,4,5,6,7) to override."
    fi
done

# ── Weight loading ──────────────────────────────────────────────────────────
#
# SGLang silently falls back to SINGLE-THREADED weight loading whenever
# checkpoint prefetch is on and the user has not asked for threads explicitly:
#
#   "--weight-loader-prefetch-checkpoints is enabled; falling back to
#    single-threaded weight loading to avoid I/O oversubscription with the
#    prefetch threads. Set enable_multithread_load=true in
#    --model-loader-extra-config to keep multi-threaded loading."
#
# One sequential reader against GPFS is what makes a cold start sawtooth: a
# burst while a shard streams, a dip while it is converted and copied to HBM,
# then the next shard. Naming either key in the JSON suppresses that fallback,
# which is the whole reason this flag is set by default.
#
# Bunya is GPFS, not Lustre, so there is no 'lfs setstripe' to reach for and
# $TMPDIR is the same filesystem — the fix has to be client-side parallelism.
load_flags=()
loader_cfg=""

if (( WEIGHT_LOAD_THREADS > 0 )); then
    loader_cfg="\"enable_multithread_load\":true,\"num_threads\":$WEIGHT_LOAD_THREADS"
fi

if [[ "$LOAD_FORMAT" == "presharded" && -n "$PRESHARDED_PATH" ]]; then
    # presharded takes its target root in the SAME extra-config object, so both
    # settings have to be merged into one flag rather than passed twice.
    [[ -n "$loader_cfg" ]] && loader_cfg+=","
    loader_cfg+="\"presharded_path\":\"$PRESHARDED_PATH\""
    # The speculative draft is a separate model and gets its own root. Without
    # it the draft dumps to <draft_model_path>/presharded — inside the HF cache,
    # which is exactly the read-only mount upstream warns against.
    if [[ "$SPECULATIVE" == "dspark" ]]; then
        loader_cfg+=",\"draft_presharded_path\":\"$PRESHARDED_PATH/dspark\""
    fi
fi

if [[ -n "$loader_cfg" || -n "$LOAD_FORMAT" ]]; then
    help_text="$(image_help)"

    if [[ -z "$help_text" ]]; then
        warn "Could not read --help from the image; passing the weight-loading flags unverified."
    else
        if [[ -n "$loader_cfg" ]] && ! grep -q -- '--model-loader-extra-config' <<<"$help_text"; then
            warn "This image has no --model-loader-extra-config, so multi-threaded weight
  loading cannot be requested and the cold start will stay single-threaded.
  Set WEIGHT_LOAD_THREADS=0 in qwen38.env to silence this."
            loader_cfg=""
        fi
        if [[ -n "$LOAD_FORMAT" ]] && ! grep -q -- "$LOAD_FORMAT" <<<"$help_text"; then
            die "This image's --load-format does not offer '$LOAD_FORMAT'.
  Run './serve-qwen38.sh loadstat' to see what it does support, then set
  LOAD_FORMAT in qwen38.env accordingly (empty = the image's default)."
        fi
    fi
fi

[[ -n "$LOAD_FORMAT" ]] && load_flags+=(--load-format "$LOAD_FORMAT")
[[ -n "$loader_cfg" ]] && load_flags+=(--model-loader-extra-config "{$loader_cfg}")

if [[ -n "$loader_cfg" ]]; then
    log "Weight loading: ${WEIGHT_LOAD_THREADS} threads${LOAD_FORMAT:+, format=$LOAD_FORMAT}"
fi

cmd=("${launcher[@]}"
     --model-path "$MODEL_ID"
     --served-model-name "$SERVED_MODEL_NAME"
     --trust-remote-code
     --tp-size "$TP_SIZE"
     ${dp_flag[@]+"${dp_flag[@]}"}
     ${attn_flag[@]+"${attn_flag[@]}"}
     ${quant_flag[@]+"${quant_flag[@]}"}
     --mem-fraction-static "$MEM_FRACTION"
     ${cgraph_flag[@]+"${cgraph_flag[@]}"}
     --host 0.0.0.0 --port "$PORT"
     --api-key "$QWEN38_API_KEY"
     ${radix_flag[@]+"${radix_flag[@]}"}
     ${ctx_flag[@]+"${ctx_flag[@]}"}
     ${kv_flag[@]+"${kv_flag[@]}"}
     ${page_flag[@]+"${page_flag[@]}"}
     ${parser_flags[@]+"${parser_flags[@]}"}
     ${spec_flags[@]+"${spec_flags[@]}"}
     ${mamba_flags[@]+"${mamba_flags[@]}"}
     ${perf_flags[@]+"${perf_flags[@]}"}
     ${load_flags[@]+"${load_flags[@]}"}
     ${extra_args[@]+"${extra_args[@]}"})

# ── Launch ──────────────────────────────────────────────────────────────────

log "Starting SGLang: $MODEL_ID  (TP=$TP_SIZE -> $GPUS_USED GPUs, ctx=${CONTEXT_LEN:-model max (262144)}, port=$PORT)"
log "Image: $SIF_PATH"
log "Logs:  $LOG_FILE"

# Assemble the container argv ONCE. Everything below — the preflight and the
# launch itself — uses this same array, so the configuration we test is by
# construction the configuration we run. (Testing a separately-built command is
# how a preflight passes while the real launch is missing a bind.)
make_rocminfo_shim

# presharded writes a second, per-rank copy of the checkpoint. Refuse to start a
# dump that cannot finish — the same reasoning as the weights-cache accounting
# above, and the same failure mode if skipped: hours lost, then no space.
presharded_bind=()
if [[ "$LOAD_FORMAT" == "presharded" && -n "$PRESHARDED_PATH" ]]; then
    mkdir -p "$PRESHARDED_PATH" 2>/dev/null || true
    [[ "$SPECULATIVE" == "dspark" ]] && mkdir -p "$PRESHARDED_PATH/dspark" 2>/dev/null || true
    [[ -d "$PRESHARDED_PATH" && -w "$PRESHARDED_PATH" ]] \
        || die "PRESHARDED_PATH '$PRESHARDED_PATH' does not exist or is not writable."
    ps_free_gb="$(df -Pk "$PRESHARDED_PATH" | awk 'NR==2 {print int($4/1024/1024)}')"
    if [[ "${ps_free_gb:-0}" -lt "$WEIGHTS_GB" ]]; then
        warn "Only ${ps_free_gb} GB free at $PRESHARDED_PATH; the presharded dump is up to ${WEIGHTS_GB} GB
  (less with dedup, but do not count on it). The dump happens AFTER a full load."
    fi
    # Only needs its own bind when it sits outside the cache dir already bound.
    [[ "$PRESHARDED_PATH" != "$MODEL_CACHE_DIR"/* ]] \
        && presharded_bind=(--bind "$PRESHARDED_PATH":"$PRESHARDED_PATH")
fi

prefetch_env=()
[[ -n "$PREFETCH_BLOCK_SIZE_MB" ]] \
    && prefetch_env=(--env SGLANG_PREFETCH_BLOCK_SIZE_MB="$PREFETCH_BLOCK_SIZE_MB")

base_args=(--bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR"
    ${cache_bind[@]+"${cache_bind[@]}"}
    ${shim_args[@]+"${shim_args[@]}"}
    ${presharded_bind[@]+"${presharded_bind[@]}"}
    --env HF_HOME="$MODEL_CACHE_DIR"
    --env HF_TOKEN="${HF_TOKEN:-}"
    --env HF_HUB_ENABLE_HF_TRANSFER=1
    --env SGLANG_SET_CPU_AFFINITY="$SET_CPU_AFFINITY"
    ${prefetch_env[@]+"${prefetch_env[@]}"}
    ${aiter_env[@]+"${aiter_env[@]}"}
    ${gpu_env[@]+"${gpu_env[@]}"})

# ── Pick the GPU passthrough mode ───────────────────────────────────────────
# Probed against base_args, so the configuration we test is the one we launch.

chosen_mode=""
cache_key="$(rocm_cache_key)"

if [[ "$ROCM_MODE" != "auto" ]]; then
    log "ROCM_MODE=$ROCM_MODE (explicit) — verifying it reaches the GPUs ..."
    if gpu_probe "$ROCM_MODE" "${base_args[@]}"; then
        chosen_mode="$ROCM_MODE"
    else
        die "ROCM_MODE=$ROCM_MODE cannot reach this node's GPUs.
  devices seen : $PROBE_DEVICES        gfx: $PROBE_GFX
  container    : $PROBE_ERR
  Run './serve-qwen38.sh gpucheck' — it tries every mode and names the cause.
  Or set ROCM_MODE=auto to let this script choose."
    fi
elif [[ "$(cut -d' ' -f1 "$ROCM_MODE_CACHE" 2>/dev/null || true)" == "$cache_key" ]]; then
    chosen_mode="$(cut -d' ' -f2 "$ROCM_MODE_CACHE" 2>/dev/null || true)"
    log "GPU passthrough: $chosen_mode (remembered for this node+image; delete $ROCM_MODE_CACHE to re-probe)"
else
    log "Probing GPU passthrough modes (~1 min; the answer is cached per node+image) ..."
    probe_best_devices=0
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" "${base_args[@]}"; then
            log "  $m: OK — $PROBE_DEVICES GPU(s), $PROBE_GFX"
            chosen_mode="$m"
            break
        fi
        warn "  $m: devices=$PROBE_DEVICES gfx=$PROBE_GFX — $PROBE_ERR"
        [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt "$probe_best_devices" ]] \
            && probe_best_devices="$PROBE_DEVICES"
    done
    [[ -n "$chosen_mode" ]] \
        && printf '%s %s\n' "$cache_key" "$chosen_mode" > "$ROCM_MODE_CACHE" 2>/dev/null || true
fi

if [[ -z "$chosen_mode" ]]; then
    # Torch seeing the GPUs while aiter cannot name the architecture is a
    # broken rocminfo, not a broken container — and needs a one-line fix
    # rather than a new image. Do not conflate the two.
    if [[ "${probe_best_devices:-0}" -gt 0 ]]; then
        die "The GPUs are reachable ($probe_best_devices seen) but aiter cannot determine the
  architecture — the image's rocminfo fails on this node even though HIP works.
  This does NOT need a new image.
  rocminfo shim applied: ${shim_args[*]:-<none>}
  Run './serve-qwen38.sh gpucheck' for the full report."
    fi
    die "No GPU passthrough mode works on this node — the container cannot reach the GPUs.
  Tried: ${ROCM_MODES[*]}
  node ROCm $(host_rocm_ver || echo '?')  vs  container ROCm $(sif_rocm_ver || echo '?')
  torch saw 0 devices in every mode, so the node's kernel driver is newer than
  the container's ROCm and no bind fixes it — you need an image built for this
  node's ROCm. Run './serve-qwen38.sh gpucheck' for the full report, and see
  the README 'When the node's ROCm changes'."
fi

if [[ "${PROBE_DEVICES:-}" =~ ^[0-9]+$ && "$PROBE_DEVICES" -lt "$GPUS_USED" ]]; then
    warn "Container sees $PROBE_DEVICES GPU(s) but TP_SIZE=$TP_SIZE needs $GPUS_USED. Startup will fail."
fi

set_rocm_mode_args "$chosen_mode"
log "GPU passthrough: $chosen_mode (${rocm_args[*]})"

apptainer_args=(exec "${rocm_args[@]}" "${base_args[@]}")

# ── Preflight: aiter's JIT dir must be writable *as the server will see it* ──
# The MXFP4 MoE compiles FlyDSL kernels at CUDA-graph capture, which happens
# after the ~1.4 TB weight load. Catch a read-only path in seconds instead.
if [[ "${AITER_JIT_TARGET:-}" == /* ]]; then
    probe="$AITER_JIT_TARGET/flydsl_cache/.write-test.$$"
    if ! probe_err="$(apptainer "${apptainer_args[@]}" "$SIF_PATH" \
            sh -c "mkdir -p '$AITER_JIT_TARGET/flydsl_cache' \
                   && touch '$probe' && rm -f '$probe'" 2>&1)"; then
        die "aiter's JIT dir is NOT writable inside the container.
  Bind attempted : ${cache_bind[*]:-<none — detection failed>}
  Path tested    : $probe
  Container said : ${probe_err:-<no output>}
  Startup would die at CUDA-graph capture after the full weight load.
  Fix the bind, or set AITER_JIT_TARGET in qwen38.env to the jit/ directory
  shown in the 'Read-only file system' traceback."
    fi
    log "Preflight OK: $AITER_JIT_TARGET is writable in the container."
else
    warn "No aiter JIT bind — startup will likely die at CUDA-graph capture."
fi

# The log is world-readable on scratch and gets pasted into bug reports, so
# never write the HF token or the API key into it.
redact_args() {
    local out=() prev="" a
    for a in "$@"; do
        if [[ "$prev" == "--api-key" ]]; then out+=("<redacted>")
        elif [[ "$a" == HF_TOKEN=* ]]; then   out+=("HF_TOKEN=<redacted>")
        else out+=("$a"); fi
        prev="$a"
    done
    printf '%s' "${out[*]}"
}

: > "$LOG_FILE"
{
    printf '### serve-qwen38.sh  %s\n' "$(date)"
    printf '### image=%s\n' "$SGLANG_IMAGE"
    printf '### apptainer: apptainer %s %s\n' "$(redact_args "${apptainer_args[@]}")" "$SIF_PATH"
    printf '### command: %s\n\n' "$(redact_args "${cmd[@]}")"
} >> "$LOG_FILE"

apptainer "${apptainer_args[@]}" \
    "$SIF_PATH" \
    "${cmd[@]}" \
    >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

cleanup() {
    log "Shutting down server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -f 'sglang.launch_server' 2>/dev/null || true
    pkill -f 'sglang serve'         2>/dev/null || true
    rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Wait for readiness ──────────────────────────────────────────────────────

log "Waiting for the server to become healthy (timeout ${READY_TIMEOUT}s)."
log "A cold start reads ~${WEIGHTS_GB} GB off scratch and JIT-compiles MXFP4 kernels — be patient."
log "Follow detailed progress in another shell with: tail -f $LOG_FILE"

start_ts="$(date +%s)"
while true; do
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready_secs=$(( $(date +%s) - start_ts ))
        # Effective GB/s over the whole startup, not just the read: it includes
        # conversion, H2D and graph capture. It is a comparable number between
        # runs, which is the point.
        ready_gbs="$(awk -v g="$WEIGHTS_GB" -v s="$ready_secs" \
            'BEGIN {printf "%.2f", (s > 0 ? g / s : 0)}')"
        log "Ready in ${ready_secs}s (${ready_gbs} GB/s effective over ${WEIGHTS_GB} GB of weights)"

        # $LOG_FILE is truncated on every launch, so the comparison you actually
        # want — this run against the last one — would be gone. Keep a small
        # append-only history instead. This is what makes an A/B possible.
        printf '%s\t%s\tthreads=%s\tformat=%s\tspec=%s\t%ss\t%s GB/s\n' \
            "$(date -Is)" "$(hostname -s 2>/dev/null || hostname)" \
            "$WEIGHT_LOAD_THREADS" "${LOAD_FORMAT:-auto}" "${SPECULATIVE:-none}" \
            "$ready_secs" "$ready_gbs" >> "$LOADTIMES_FILE" 2>/dev/null || true

        if grep -qi "falling back to single-threaded" "$LOG_FILE" 2>/dev/null; then
            warn "SGLang fell back to SINGLE-THREADED weight loading — that is the sawtooth.
  Set WEIGHT_LOAD_THREADS=8 in qwen38.env. See './serve-qwen38.sh loadstat'."
        fi
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        tail -n 60 "$LOG_FILE" 2>&1 || true
        rm -f "$PID_FILE"
        die "Server process exited during startup. Last 60 log lines above ($LOG_FILE).
     Unknown architecture?   ./serve-qwen38.sh check
     Unknown parser name?    ./serve-qwen38.sh parsers
     KV/GDN pool OOM?        set CONTEXT_LEN=131072 in qwen38.env and retry
                             (do NOT lower MEM_FRACTION below 0.9 first — see README)
     Speculative TypeError?  unset SPECULATIVE (sglang #32569 / #33694)"
    fi
    if (( $(date +%s) - start_ts > READY_TIMEOUT )); then
        die "Server did not become healthy within ${READY_TIMEOUT}s. Check: tail -f $LOG_FILE"
    fi
    sleep 10
done

# ── Connection banner ───────────────────────────────────────────────────────

NODE_HOST="$(hostname -s 2>/dev/null || hostname)"
JOBID="${SLURM_JOB_ID:-<jobid>}"
cat <<EOF

============================================================================
  $SERVED_MODEL_NAME is up and serving on Bunya's MI355X.

  Model:       $MODEL_ID
  Layout:      TP=$TP_SIZE, ctx=${CONTEXT_LEN:-model max (262144)}, spec=${SPECULATIVE:-off}
  Node:        $NODE_HOST     (job $JOBID)
  Endpoint:    http://$NODE_HOST:$PORT/v1   (OpenAI-compatible)
  Model name:  $SERVED_MODEL_NAME
  API key:     $API_KEY_FILE
               export QWEN38_API_KEY="\$(cat $API_KEY_FILE)"

  Smoke test (from this node) — READ THE REPLY, coherence is the real test.
  Qwen3.8 always reasons, so the answer arrives in reasoning_content + content:
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \\
      -H "Authorization: Bearer \$QWEN38_API_KEY" \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"$SERVED_MODEL_NAME","reasoning_effort":"low","messages":[{"role":"user","content":"Say hello in one sentence."}]}'

  Prove the parsers round-trip (tool calls AND reasoning):
    ./serve-qwen38.sh toolcheck

  Second shell into this job (to run a client alongside the server):
    srun --overlap --jobid $JOBID --pty /bin/bash -l

  From your laptop (tunnel through the login node, then use localhost):
    ssh -N -L $PORT:$NODE_HOST:$PORT \${USER}@bunya1.rcc.uq.edu.au

  opencode:   ./opencode-setup.sh --host $NODE_HOST --port $PORT
              (or --host localhost when tunnelling), then pick
              '$SERVED_MODEL_NAME' via /models.

  qwen-code:  ./qwencode-setup.sh --host $NODE_HOST --port $PORT
              (or --host localhost when tunnelling).

  Benchmark:  ./bench-qwen38.sh sweep
              (no published MI355X figures exist yet — these are the first)

  Stop with Ctrl-C, 'scancel $JOBID', or './serve-qwen38.sh stop'.
============================================================================

EOF

if [[ "$DETACH" -eq 1 ]]; then
    # Disarm the cleanup traps: the server keeps running in the background
    # (until './serve-qwen38.sh stop' or the SLURM job/allocation ends).
    trap - EXIT INT TERM
    disown "$SERVER_PID" 2>/dev/null || true
    log "Detached. You have your shell back — the server keeps running on this node."
    log "  Logs:  tail -f $LOG_FILE"
    log "  Stop:  ./serve-qwen38.sh stop"
    exit 0
fi

# Stay attached: keeps the SLURM job alive and tears the server down on
# Ctrl-C / scancel via the traps above.
log "Attached. Ctrl-C (or scancel) stops the server. Streaming logs:"
tail -f "$LOG_FILE" &
TAIL_PID=$!
# Wait on the server; when it exits, stop tailing.
wait "$SERVER_PID"
kill "$TAIL_PID" 2>/dev/null || true
