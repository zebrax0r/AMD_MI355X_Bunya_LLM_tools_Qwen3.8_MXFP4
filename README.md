# Qwen3.8 on Bunya's AMD MI355X — native MXFP4 serving with SGLang

One-click SGLang serving of **Qwen3.8-2.4T-A95B** in **native MXFP4** on a single
8×MI355X node of UQ's Bunya supercomputer, with an OpenAI-compatible endpoint,
working tool calls and reasoning, a benchmark harness, and client wiring for
opencode and Qwen Code.

This is the sibling of
[`AMD_MI355X_Bunya_LLM_tools_KimiK3`](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_KimiK3).
Same machinery, same discipline, different model. Where that repo learned
something the hard way about Apptainer, ROCm passthrough or aiter, that knowledge
is carried over here verbatim rather than re-derived — it is a property of the
environment, not of the model.

---

## Status — what is verified, what is measured, what is neither

Three different words, and this README keeps them apart on purpose.

| | |
|---|---|
| **Verified by upstream** | SGLang marks the `mi355x / mxfp4 / balanced / single-node` cookbook cell `verified: true` — they have run it and it is correct. The launch recipe in this repo is that cell, unchanged apart from three documented deviations. |
| **Measured here** | **Nothing yet.** This repo was written on 14 Aug 2026, two days after the model shipped. No Bunya run has happened. Every performance number below is upstream's, attributed, or absent. |
| **Neither** | DSpark speculative decoding on MI355X, the GDN state-pool knobs, `presharded` loading, and the two non-canonical MXFP4 checkpoints. All are plumbed, all are off or empty by default, none are claimed to work. |

Upstream publishes **no throughput figures at all** for the MI355X cell — every
benchmark in the Qwen3.8 day-0 material is on NVIDIA GB300. So there is nothing
to compare a first run against except your own later runs. `bench-qwen38.sh`
records the node and the date with every result for exactly that reason.

### The model, in one table

Read off the checkpoint's own `config.json`.

| | |
|---|---|
| Parameters | 2.4T total, 95B active per token |
| Layers | 92 — **69 Gated DeltaNet** (linear attention) + **23 gated full attention**, 3:1 (`full_attention_interval: 4`) |
| MoE | 512 experts, top-10 routed + 1 shared, expert intermediate 2048 |
| Hidden / vocab | 8192 / 248,320 |
| Full attention | 64 query heads over 4 KV heads, head dim 256, output gating (`swish`) |
| GDN | 128 value heads, 16 QK heads, head dim 128, conv kernel 4; state dtype fp32 |
| MTP | `mtp_num_hidden_layers: 1` — ships in the checkpoint, so NEXTN needs no draft |
| Context | 262,144 native (`rope_type: default`), extensible to 1,010,000 with scaling |
| Architecture | `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` |

The MXFP4 checkpoint is **hybrid**: MXFP4 experts, FP8 attention and dense
layers. That is what the `FP8-MXFP4` in the canonical repo name means.

### Provenance

* Recipe, flags and the configuration guidance: SGLang's
  [Qwen3.8 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8),
  `hw=mi355x, variant=default, quant=mxfp4, strategy=balanced, nodes=single`.
* Architecture figures: the checkpoint's own `config.json`, cross-checked
  against the cookbook's model introduction.
* Checkpoint sizes and reachability: the HuggingFace API, queried 14 Aug 2026.
* Everything about Apptainer, ROCm passthrough, the aiter JIT directory and
  GPFS weight loading: measured on bun159/160/161 during the Kimi K3 work,
  Jul–Aug 2026.

---

## The recipe

Upstream's verified MI355X cell is exactly this:

```
SGLANG_USE_AITER=1 sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4 \
  --tp-size 8 \
  --mem-fraction-static 0.9 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host <HOST> --port <PORT>
```

in `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260812`.

**Read what is not there.** No `--attention-backend`, no `--moe-runner-backend`,
no `--mamba-ssm-dtype`, no `--context-length`, no speculative flags, no
`--kv-cache-dtype`. Every one of those is a deliberate omission, not an
oversight, and this repo keeps them omitted:

* `--attention-backend` — `trtllm_mha` is SM100-only. On gfx950 the model hook
  chooses for itself. `ATTENTION_BACKEND` defaults to **empty** here, which
  emits no flag. (The Kimi K3 sibling hardcoded `triton`; do not copy that
  across.)
* `--moe-runner-backend` — upstream: *"Leave `--moe-runner-backend` unset and the
  runner resolves from the checkpoint's own `quant_method`."* The only exception
  on the whole page is the NVFP4 wide-EP tier, which is not us.
* `--mamba-ssm-dtype bfloat16` — load-bearing **on SM100 only**, where it gates
  FlashInfer's GDN decode default. There is no such default to unlock on gfx950,
  so here it is a memory knob, not a correctness one.

### Our deviations from upstream

Three, all environmental:

1. **Binds `0.0.0.0`, not `127.0.0.1`.** Upstream's recipe is a single-machine
   demo. We need SSH tunnels and off-node clients to reach the endpoint.
2. **Adds `--api-key` and `--served-model-name`.** The key is the direct
   consequence of (1) — it is the only thing gating the port. It is generated
   once and persisted to `$MODEL_CACHE_DIR/qwen38-api-key`.
3. **Forces `SGLANG_SET_CPU_AFFINITY=0`.** Affinity pinning computes
   `total_pcores = psutil.cpu_count(logical=False)` — the **whole node's** cores,
   ignoring the cgroup — then hands rank *i* the slice `[i·n, (i+1)·n)`. Under a
   SLURM cgroup that owns only a subset, high ranks name CPUs we do not have and
   startup dies with `CPU number N is not eligible`. Upstream has no cgroup to
   trip over.

   **This costs less than it looks like it does.** SGLang's own default for
   `SGLANG_SET_CPU_AFFINITY` is already `False` (`srt/environ.py`), so we only
   diverge if the *image* turns it on — check with
   `apptainer exec $SIF_PATH env | grep -iE 'AFFINITY|NUMA'`. And **NUMA binding
   is a separate mechanism that still runs**: `scheduler.py` gates it on
   `SGLANG_NUMA_BIND_V2`, which defaults **true** and uses the cgroup-aware
   `numactl` path in `numa_utils.py` — it probes whether the binding can be
   applied before applying it, and warns-and-continues if not. So we lose
   per-rank *core partitioning*, not NUMA memory locality. (That path needs
   `numactl` inside the image; if it is missing, binding is skipped with a
   warning. Worth confirming once.)

   If you allocate the whole node's CPUs (`--exclusive`), the cgroup owns every
   core, the arithmetic above lines up, and `SET_CPU_AFFINITY=1` becomes
   available — that is the way to get per-rank isolation back.

Not a deviation any more, but worth recording: **`AITER_FLYDSL_FORCE` now
defaults to `0`** (it was `1` until 14 Aug 2026). It is ours, not upstream's, and
appears nowhere in SGLang mainline — aiter reads it. It was the measured fast path
on Kimi K3's MoE shapes; Qwen3.8's are different, and forcing the JIT path when
there is no tuned config for these shapes is precisely how you land on the slow
heuristic fallback. So the out-of-box run is now upstream's cell verbatim. Turning
it back on is the first rung of the [tuning ladder](#tuning-ladder).

---

## The canonical MXFP4 checkpoint is not public yet

This is the first thing that will stop you, so it is the first thing documented.

`Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4` is the id the verified cell names and the
cookbook links to. As of **14 Aug 2026** it is not fetchable: anonymously it
returns **HTTP 401** (model page, API and `resolve/main/config.json` all refuse),
and **with a token that lacks access it returns 404** — observed on Bunya the same
day. The BF16 and FP8 repos of the same model are public and return 200. Either
code means the same thing for you: it is private or gated and you cannot pull it.

Do not read 404 as "I typed the name wrong". `check` prints whichever code you
actually got, and distinguishes "no `HF_TOKEN` was sent" from "your `HF_TOKEN` did
not open it", because those have different fixes.

Two public MXFP4 checkpoints exist:

| Repo | Size | Shards | `quant_method` | Base | Notes |
|---|---:|---:|---|---|---|
| `Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4` | ? | ? | ? | — | **The validated one. 401/404 today.** |
| `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` | 1372 GB | 213 | `quark` | Qwen …-FP8 | AMD's own Quark MXFP4. Smallest, so the most KV/GDN headroom. |
| `Inferact/Qwen3.8-2.4T-A95B-MXFP4` | 1597 GB | 101 | `mxfp4` | Qwen …-A95B | Declares the exact string SGLang's auto-resolution keys on. Community repo. |

SGLang's `QUANTIZATION_CHOICES` carries `mxfp4`, `quark` and `quark_mxfp4`, so
both alternatives are servable in principle. **Neither is what upstream
validated.** A quantisation that loads but resolves to the wrong kernel produces
fluent nonsense, not an error — so whichever you serve, read the first replies
for coherence rather than assuming.

### What the script does about it

`./serve-qwen38.sh check` probes every candidate with your token *before* you
commit to a ~1.4 TB transfer, and prints reachability, size, shard count and the
declared `quant_method` for each:

```
  -> Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4
       UNREACHABLE (401 — private, gated, or does not exist — your HF_TOKEN did not open it)
     amd/Qwen3.8-2.4T-A95B-Quark-MXFP4
       OK     1372.5 GB    213 shards   gated=False   quant_method=quark
            -> SGLang quantization: quark        (AMD Quark; FP8/MXFP4/Int4FP8)
            -> architectures: ['Qwen3_5MoeForCausalLM']
     Inferact/Qwen3.8-2.4T-A95B-MXFP4
       OK     1597.1 GB    101 shards   gated=False   quant_method=mxfp4
            -> SGLang quantization: mxfp4        (MoE-only; the auto-resolution target)
            -> architectures: ['Qwen3_5MoeForCausalLM']
```

(That is real output, taken 14 Aug 2026 with no token. It distinguishes "no
`HF_TOKEN` was sent" from "your `HF_TOKEN` did not open it", because those have
different fixes.)

`->` marks your configured `MODEL_ID`. Set `MODEL_ID` in `qwen38.env` to one that
reports OK, set `WEIGHTS_GB` to match, and re-run `check`.

**Run `check` before `download`.** Skipping it is how you spend hours discovering
that the repo still 401s (or 404s) for your token.

---

## Memory budget — and why `--mem-fraction-static 0.9` must not be lowered

`0.9` looks reckless. It is not, and upstream says so explicitly: *"`--mem-fraction-static`
looks aggressive on purpose. With the aiter backend above 8192 context SGLang
multiplies it by `0.85` before allocating, so MI355X's `0.9` lands at ≈0.765.
Don't 'fix' these downward."*

The arithmetic:

```
8 × 288 GB                     = 2304 GB HBM on the node
0.765 × 2304                   = 1763 GB static allocation
      − ~1372 GB of MXFP4 weights
                               = ~390 GB for KV cache + GDN state pool + CUDA graphs
```

Lowering it to a "safer" `0.85` gives an effective 0.7225, which costs about
100 GB — **a quarter of your entire serving headroom**. If allocation fails at
startup, set `CONTEXT_LEN=131072` *first*, so you learn which knob mattered.

For context on why single-node MXFP4 is the whole point: at 2.4T parameters BF16
is ≈4.8 TB and FP8 ≈2.4 TB. FP8 fits no single node anywhere — not even a B300,
whose 8 × 288 GB misses by a hair — so every FP8 recipe upstream publishes is
multi-node. Single-node means FP4, and on AMD that means MXFP4 on gfx950.
MI300X is CDNA3, has no hardware MX matmul at all, and needs two nodes for FP8.

---

## GDN state is the scarce resource, not KV

Qwen3.8 is a **hybrid**: of its 92 layers, **69 are Gated DeltaNet** linear
attention and **23 are gated full attention**, in a strict 3:1 pattern
(`full_attention_interval: 4`).

* **GDN layers** hold a fixed-size recurrent state per request — 128 value heads
  and 16 QK heads at head dimension 128, with a causal conv. Memory is *O(1)* per
  layer regardless of context length, while compute stays *O(N)*.
* **Full-attention layers** hold an ordinary KV cache: 64 query heads over 4 KV
  heads at head dimension 256, with output gating.

Upstream is explicit that **the GDN state pool, not KV, is usually what caps
concurrency**, and that a request's cost depends on the caching strategy:

| Strategy | GDN state slots per request |
|---|---:|
| `--disable-radix-cache` | 1 |
| `no_buffer` | 3 |
| `extra_buffer_lazy` | 4 |
| `extra_buffer` — this model's `auto` | 5 |

Two traps follow, and `serve-qwen38.sh` catches both:

**`extra_buffer` needs the radix cache ON.** `mamba_extra_buffer_of()` requires
`disable_radix_cache` to be false. Set `DISABLE_RADIX_CACHE=1` and the strategy
goes *inert* — the budget silently drops from 5 slots to 1. Nothing in the log
says so. That is a 5× change in concurrency headroom from a flag that looks
unrelated.

**`--max-mamba-cache-size` is in slots, not requests.** It has to match the ratio
in force or it silently clamps `max_running_requests` to a fraction of your
target. Upstream's worked example: GB300 Low Latency pins `80`, which is exactly
its 16 concurrent requests × 5 slots. `MAX_MAMBA_CACHE_SIZE` in `qwen38.env`
makes the script do that arithmetic and warn when it undercuts
`MAX_RUNNING_REQUESTS`. Prefer `MAMBA_FULL_MEMORY_RATIO` unless you are pinning a
tuned capacity set — that is what every cookbook cell does bar two.

Note the checkpoint declares `mamba_ssm_dtype: float32`, so `MAMBA_SSM_DTYPE=bfloat16`
halves the state pool's footprint and buys concurrency directly. Untested here.

---

## Reasoning and tool calling

### Reasoning cannot be turned off

Qwen3.8 **always** reasons. Every response opens with a `<think>…</think>` block;
there is no non-thinking mode. The `qwen3` reasoning parser splits that block
into `reasoning_content`, leaving `content` as the answer alone.

This makes `REASONING_PARSER` effectively non-optional. Get it wrong and the
server still returns HTTP 200 — every reply just arrives with raw `<think>` tags
glued to the front of `content`. That is not an error anywhere in the stack; it
simply makes agentic clients behave strangely. `./serve-qwen38.sh toolcheck`
asserts it directly.

Depth is tunable per request:

```python
resp = client.chat.completions.create(
    model="qwen3.8",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    reasoning_effort="xhigh",     # xhigh (default) | medium | low
)
print("Reasoning:", resp.choices[0].message.reasoning_content)
print("Answer:",    resp.choices[0].message.content)
```

`preserve_thinking` carries reasoning from earlier turns into context and is on
by default.

Two practical consequences:

* **Benchmark output lengths measure thinking, not answering.** At `xhigh` most
  of a 512-token budget is `<think>`. That is a real property of the model, not
  a benchmark artefact — but do not compare these tok/s against a non-reasoning
  model's without saying so.
* **Client output limits must be generous.** Reasoning tokens come out of the
  same generation budget, so a limit set too low truncates the model mid-thought
  and the client sees an empty answer rather than an error. Qwen suggests
  allowing 262,144 tokens of reasoning and 131,072 for the final response in
  agentic work; the opencode template pins output at 131072 accordingly.

### Tool calling

`--tool-call-parser qwen3_coder` surfaces structured calls via
`message.tool_calls`. Prove the whole loop closes with:

```bash
./serve-qwen38.sh toolcheck
```

It checks reasoning splitting, then a two-turn tool round trip where the "tool"
returns a value the model cannot guess — and asserts that value appears in the
final answer. A model emitting a plausible `tool_calls` object proves the parser
serialises; it does not prove the loop closes, and agentic clients need both.

### Sampling

Qwen's recommended generation settings for this model: `temperature=1.0`,
`top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`,
`repetition_penalty=1.0`. Raising `presence_penalty` toward 2 curbs runaway
repetition at some risk of language mixing.

**Note that this sends both `top_p` and `top_k` and a non-zero temperature.**
That matters if you enable speculative decoding — see below.

---

## Speculative decoding

**Off by default.** The verified cell uses none, and a first run should have one
variable in it, not three. Two options are plumbed:

### `SPECULATIVE=nextn` — the checkpoint's own MTP head

Qwen3.8 ships multi-token-prediction weights inside the checkpoint
(`mtp_num_hidden_layers: 1`), so NEXTN needs **no draft model** and the 3/1/4
preset fills in automatically. This is what upstream's NVIDIA cells use, and it
is the cheaper of the two to try because there is nothing extra to fetch.

**The MRR-48 trap.** A speculative cell with no `--max-running-requests` gets
**48** from the speculative hook rather than a memory-derived ceiling. Nothing
errors. You simply serve a fraction of what the node can hold, and the throughput
number you benchmark is wrong for a reason no log line mentions. Set
`MAX_RUNNING_REQUESTS` explicitly; the script warns whenever you have not.

### `SPECULATIVE=dspark` — the trained draft model

`RadixArk/Qwen3.8-2.4T-A95B-DSpark`, 6.6 GB, 5 layers, `block_size: 7`,
`max_position_embeddings: 262144`.

**This is not part of the verified matrix on MI355X.** Upstream offers the DSpark
chip only on GB300 FP8/NVFP4/BF16 and B300 NVFP4, because *"everything else on
the page is either pipelined or wide-EP"* — and DSpark's `_handle_dspark`
rejects those outright rather than degrading. MI355X TP8 is neither pipelined nor
wide-EP, so it should clear the constraints. It is simply unvalidated. Treat any
number you get as new information, not as confirmation.

One thing is better here than on the K3 sibling: that draft was trained at 4,096
tokens with unscaled RoPE, and serving it at 100k collapsed the accept rate from
7.25 to 1.39 — turning DSpark into a 2.93× net *loss*. This draft declares a
262,144 window, matching the model's own native context, so there is no built-in
cliff. `serve-qwen38.sh` still checks the draft's rope type and trained window at
launch and warns if that stops being true.

### The ROCm sampling landmine — and it is worse here than on Kimi K3

The speculative verify step calls `top_k_renorm_prob` / `top_p_renorm_prob`. On
ROCm those are bound by an `is_hip()` branch that landed in two halves — sglang
[#32621](https://github.com/sgl-project/sglang/pull/32621) (top_p, 28 Jul 2026)
and [#32641](https://github.com/sgl-project/sglang/pull/32641) (top_k, 31 Jul).
An image predating those has **neither**: both names are `None`, and the first
request that sets either raises inside the scheduler's event loop and **kills the
server**, not just that request — taking every concurrent request with it
([#32569](https://github.com/sgl-project/sglang/issues/32569)).

There is a second one with a commoner trigger:
[#33694](https://github.com/sgl-project/sglang/pull/33694) (fixed 6 Aug 2026). An
`elif is_hip():` branch set `_DFLASH_SAMPLING_VERIFY_AVAILABLE = True` without
ever binding `tree_speculative_sampling_target_only`, a kernel that does not
exist on ROCm at all. Any request with `temperature > 0` then raised `NameError`
and killed the scheduler.

**Why this is sharper for Qwen3.8:** its documented recommended sampling is
`temperature=1.0, top_p=0.95, top_k=20`. The defaults send *both* affected
parameters *and* a non-zero temperature. On K3 you had to go out of your way to
trip this; here a normal client finds it on the first request.

`serve-qwen38.sh` greps the image for the Triton aliases and for that binding,
and warns at launch, so you cannot meet either unknowingly. The pinned
`v0.5.17-…-20260812` image postdates all three fixes and should be clean — read
the launch output and confirm rather than assuming.

### ReplaySSM

`REPLAYSSM_SPEC=1` adds `--enable-linear-replayssm-spec`. A GDN layer's recurrent
state overwrites itself every token, so speculative verify has to be rewindable.
Snapshotting the whole K×V state per draft step costs 64 KiB per request, layer
and head at K=V=128, times γ+1 steps — scratch taken out of the same budget as
the persistent state pool. ReplaySSM stores each draft step's raw inputs instead
(a few hundred bytes) and replays the accepted prefix with one fold kernel. The
fold is a verbatim clone of the verify recurrence, so the rebuilt state is
bit-identical: **no accuracy tradeoff, only a memory one**, and draft-step scratch
shrinks by roughly two orders of magnitude.

**Read the flag name.** There is a separate `--enable-linear-replayssm` (buffered
*decode*, not verify) which is mutually exclusive and requires
`--mamba-radix-cache-strategy no_buffer`. Do not "simplify" one into the other.
Constraint: linear-chain only — `--speculative-eagle-topk` must be unset or 1.
The script enforces that and probes `--help` before passing the flag.

---

## Bunya specifics you need to know

* **The MI355X nodes are `bun159`, `bun160`, `bun161`** — 8 × gfx950 each, 288 GB
  HBM per GPU.
* **They live in `admin_test`, not `gpu_rocm`.** The working SLURM recipe is
  `--partition=admin_test --account=a_rcc --qos=sdf --gres=gpu:mi355x:8`.
  Confirm with `sinfo -o "%P %.10l %G %f" | grep mi355x`.
* **Apptainer exists only on compute nodes**, never the login nodes. Every mode
  except a bare `stop`/`status` must run inside `salloc`/`sbatch`.
* **Work from `/scratch`, not `/home`** (tight quota) and not RDM. Budget ~1.5 TB:
  ~1372 GB of weights, ~24 GB for the `.sif`, 6.6 GB if you fetch the DSpark
  draft. Check with `rquota` and `df -h`.
* **Bunya is GPFS, not Lustre.** There is no `lfs setstripe` to reach for, and
  `$TMPDIR` is the same filesystem, so staging there buys nothing. The only
  weight-loading lever is client-side parallelism — see below.
* **The nodes have run ROCm 7.14 while this image ships 7.2.0.** That mismatch is
  handled automatically; see "When the node's ROCm changes".
* `--mem=1800G` in the sbatch file is **host RAM** for staging the weight load,
  not HBM.

---

## Prerequisites

* An account on Bunya with access to `admin_test` / `a_rcc` / `qos=sdf`.
* ~1.5 TB of free `/scratch`.
* A HuggingFace token — needed for the first download, and needed by `check` to
  tell "gated" from "does not exist".
* Outbound internet from the compute node (for the image pull and the weights).
* `jq` on whatever machine runs the client setup scripts, if you already have a
  config to merge into. Both scripts degrade gracefully without it.

---

## Repository contents

| File | Purpose |
|---|---|
| `serve-qwen38.sh` | The core script: pull, check, gpucheck, parsers, download, serve, toolcheck, loadstat, stop, status |
| `serve-qwen38.sbatch` | SLURM batch wrapper (MI355X recipe) |
| `qwen38-env.example` | Config template — copy to `qwen38.env` and edit |
| `bench-qwen38.sh` | tok/s / TTFT / TPOT at 1024/512, or `longcontext` at 100k |
| `opencode-setup.sh` | Writes/merges the opencode provider config on any machine |
| `opencode.qwen38.json` | The provider template `opencode-setup.sh` fills in |
| `qwencode-setup.sh` | Writes/merges the Qwen Code provider config **and** `~/.qwen/.env` |
| `qwencode.qwen38.json` | The provider template `qwencode-setup.sh` fills in |

Secrets never live in the repo: `qwen38.env` (your HF token), the generated API
key and the `.sif` are gitignored / stored under `$MODEL_CACHE_DIR` on scratch.

---

## Walkthrough

### Step 0 — Get the code onto Bunya

```bash
cd /scratch/user/$USER
git clone <this repo> qwen38 && cd qwen38
```

### Step 1 — Configure

```bash
cp qwen38-env.example qwen38.env
$EDITOR qwen38.env     # MODEL_CACHE_DIR and HF_TOKEN are the two required ones
```

`qwen38.env` is gitignored. Everything uses `${VAR:-default}`, so anything you
`export` beforehand (or pass via `sbatch --export`) wins.

### Step 2 — Allocate an MI355X node

```bash
salloc --partition=admin_test --account=a_rcc --qos=sdf \
       --nodes=1 --gres=gpu:mi355x:8 --cpus-per-task=192 \
       --mem=1800G --time=8:00:00
```

### Step 3 — Build the image, then **check before you download**

```bash
./serve-qwen38.sh pull       # ~24 GB, one-time
./serve-qwen38.sh check      # THE GATE — reachability + architecture
./serve-qwen38.sh gpucheck   # ~1 min; can the container reach the GPUs?
./serve-qwen38.sh parsers    # is qwen3_coder / qwen3 actually in this image?
```

`check` is the cheap gate on a 1.4 TB commitment. It answers two separate
questions — *is the checkpoint reachable with my token* and *can this image's
SGLang load this architecture* — and reports them separately, because they have
completely different fixes.

Then, only once `check` is happy:

```bash
./serve-qwen38.sh download   # ~1.4 TB. No GPU needed; resumable.
```

### Step 4 — Serve

```bash
./serve-qwen38.sh serve            # stays attached, streams the log
./serve-qwen38.sh serve --detach   # returns your shell, server keeps running
```

or as a batch job:

```bash
mkdir -p logs && sbatch serve-qwen38.sbatch
grep -A 30 'is up and serving' logs/qwen38-<jobid>.out
```

A cold start reads ~1.4 TB off GPFS and JIT-compiles MXFP4 kernels. Be patient;
`READY_TIMEOUT` defaults to 4 hours.

### Step 5 — Verify, and actually read the output

```bash
export QWEN38_API_KEY="$(cat $MODEL_CACHE_DIR/qwen38-api-key)"
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Authorization: Bearer $QWEN38_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8","reasoning_effort":"low",
       "messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

**Read the reply.** Coherence is the real test here, more than usual: two of the
three MXFP4 checkpoints are not the one upstream validated, and a mis-resolved
quantisation produces fluent nonsense rather than an error.

### Step 5b — Prove the parsers round-trip

```bash
./serve-qwen38.sh toolcheck
```

Expect three reasoning checks and five tool-calling checks to pass. This is pure
HTTP, so it works through a tunnel from anywhere.

### Step 6 — Connect a client

```bash
# terminal 1 (<node> = the hostname the serve banner printed):
ssh -N -L 30000:<node>:30000 $USER@bunya1.rcc.uq.edu.au

# terminal 2:
./opencode-setup.sh --host localhost --port 30000
export QWEN38_API_KEY="<the key>"
opencode        # then /models -> 'Qwen3.8 (Bunya MI355X)'
```

or, for Qwen's own CLI:

```bash
./qwencode-setup.sh --host localhost --port 30000
qwen
```

`qwencode-setup.sh` writes **both** `~/.qwen/settings.json` and `~/.qwen/.env`.
The three environment variables (`OPENAI_BASE_URL`, `OPENAI_MODEL`,
`OPENAI_API_KEY`) are the documented, stable interface and outrank the settings
file; the `modelProviders` schema is newer and has been described more than one
way upstream. If qwen ever authenticates against Qwen's cloud instead of your
endpoint, the settings block is not being read — fall back to
`./qwencode-setup.sh --env-only`.

Anything else that speaks OpenAI works directly:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="<key>")
```

### Step 7 — Shut down

`Ctrl-C` if attached, otherwise `./serve-qwen38.sh stop` or `scancel <jobid>`.
The script traps SIGTERM and tears the container down cleanly.

---

## Performance tuning

Nothing here is measured on Bunya yet. The order below is chosen so that each
step has one variable in it.

1. **Baseline.** `./bench-qwen38.sh sweep` with everything at its default. Set
   `BENCH_REPEATS=3` — a single run has no error bar, and on the K3 sibling
   bun161 and bun159 differed by 5–10% on identical configs.
2. **The agentic shape.** `./bench-qwen38.sh longcontext` — 100k in, 512 out,
   c=1. This is what opencode and Qwen Code actually do every turn, and it is a
   different regime from 1024/512, not the same regime scaled. Two thirds of the
   layers being GDN should make Qwen3.8 degrade more gently with length than an
   all-attention model; whether it does on gfx950 is exactly what this measures.
3. **`SPECULATIVE=nextn`**, with `MAX_RUNNING_REQUESTS` pinned. Re-run both
   shapes. On the K3 sibling the sign of the speculative win *flipped* with
   context — a 4.7× TPOT win at 1k, a 1.7× loss at 100k — so a short-context win
   does not carry. Watch `accept len` in the server log: below ~2 at any context
   means the draft is out of its depth, which is a different failure from the
   step simply costing too much.
4. **GDN state pool**, one knob at a time: `MAMBA_FULL_MEMORY_RATIO` first (it is
   the lever upstream actually uses), then `MAMBA_SSM_DTYPE=bfloat16`, then
   `MAMBA_RADIX_STRATEGY`. Watch quality, not just throughput, on the dtype one.
5. **`DISABLE_RADIX_CACHE`** — but read the GDN section first. On K3, caching on
   was better on throughput and per-token latency everywhere, at some cost in
   TTFT under load, and the benchmark *understates* the gain for agentic clients
   because random prompts share no prefixes at all.

`bench-qwen38.sh` records `MODEL_ID`, image tag, every relevant flag, the tuning
ladder's state, the hostname and the date into each result file. Compare with
`ls -t $MODEL_CACHE_DIR/bench`.

### Are the defaults already optimal?

Mostly yes, and that was checked against SGLang's source rather than assumed.
**Every knob this repo leaves empty resolves to SGLang's own default** — which is
exactly what upstream's verified cell does, since it passes none of them either:

| Knob | SGLang default | Ours |
|---|---|---|
| `chunked_prefill_size` | `None` | empty |
| `cuda_graph_max_bs_decode` | `None` | empty |
| `max_mamba_cache_size` | `None` | empty |
| `mamba_ssm_dtype` | `None` | empty |
| `mamba_radix_cache_strategy` | `"auto"` | empty |
| `mamba_full_memory_ratio` | `0.9` | empty |
| `max_running_requests` | `None` | empty |

So there is nothing being left on the table by omission. The two things that
*were* worth changing are covered above: `FLYDSL_FORCE` now defaults off, and the
`SET_CPU_AFFINITY` framing was overstated.

**Dropping the K3 recipe's `SGLANG_AITER_K3_OPT` provably costs Qwen3.8 nothing.**
That variable still exists in mainline `mxfp4.py`, but it has exactly one effect —
`_inter_align = 128 if _aiter_k3_opt else 256`, applied as
`round_up(intermediate_size_per_partition, _inter_align)`:

```
Qwen3.8   moe_intermediate 2048 / tp8 = 256  ->  align128 = 256, align256 = 256   delta    0
Kimi K3   moe_intermediate 3072 / tp8 = 384  ->  align128 = 384, align256 = 512   delta +128
```

Qwen3.8 is unpadded either way, so K3's +33% memory saving simply does not arise
here. Same for the hidden dimension: `round_up(8192, 256) = 8192`, no pad.

<a name="tuning-ladder"></a>
### Tuning ladder — untested knobs that look matched to this model

None of these are in upstream's verified cell. All are SGLang defaults-off, all
are **off by default here**, and none are measured on Bunya. Treat each as a
hypothesis with a benchmark attached: turn on **one**, run `./bench-qwen38.sh
sweep` with `BENCH_REPEATS=3`, keep it only if it beats the error bar.

| Rung | Variable | Why it plausibly helps *this* model |
|---|---|---|
| 1 | `FLYDSL_FORCE=1` | Was the measured fast path on K3's MXFP4 MoE on the same gfx950. May or may not carry to Qwen3.8's shapes — that is the whole question. |
| 2 | `SPECULATIVE=nextn` | MTP ships in the checkpoint, so it is free. Pin `MAX_RUNNING_REQUESTS` or it silently becomes 48. |
| 3 | `ROCM_MULTI_STREAM=1` | Dual-stream MoE: the shared expert and the routed experts stop serialising. Qwen3.8 routes **10 experts + 1 shared** per token, which is exactly the shape this targets. Needs `GPU_MAX_HW_QUEUES>=5`; the script sets 5 for you. |
| 4 | `AITER_KV_LAYOUT=vectorized_5d` | The SHUFFLE layout `pa_decode_gluon` and aiter's CK FmhaBatchPrefill consume natively, so full-attention decode avoids runtime permutes. Qwen3.8 has 23 such layers. |
| 5 | `AITER_FP8_PER_TOKEN=1` | The checkpoint is hybrid — MXFP4 experts, **FP8** attention and dense. This touches that FP8 half. |

**One knob that is not on this ladder, and must not be added to it:**
`SGLANG_USE_AITER_MOE_GU_ITLV`. It reads like a performance switch and is not
one. SGLang's `mxfp4.py` uses `gate_up_interleaved` to select a **weight-layout
transform** matched to how the checkpoint physically stores `w13` — the
non-interleaved branch is commented *"e.g. K3 Latent MoE"*. Flipping it gives you
**wrong output, not slower output**, and wrong output from this model looks like
fluent, confident nonsense rather than an error.

---

## Weight loading — why a cold start sawtooths

SGLang silently falls back to **single-threaded** weight loading whenever
checkpoint prefetch is on and you have not asked for threads explicitly:

```
--weight-loader-prefetch-checkpoints is enabled; falling back to single-threaded
weight loading to avoid I/O oversubscription with the prefetch threads.
```

One sequential reader against GPFS is what makes a cold start sawtooth: a burst
while a shard streams, a dip while it is converted and copied to HBM, then the
next shard. `WEIGHT_LOAD_THREADS` (default 8) names `num_threads` in
`--model-loader-extra-config`, which suppresses that fallback. Upstream reports
~3×.

Memory: the buffered loader holds ~(threads + 2) shards at once. The AMD
checkpoint's shards are ~6.5 GB (1372 GB / 213), so 8 threads is ~65 GB against
`--mem=1800G`. The Inferact checkpoint has 101 much larger shards — check your
`--mem` before raising the thread count on that one. Past ~16 the GPFS client,
not the thread count, is normally the limit.

`./serve-qwen38.sh loadstat` reports whether the last cold start hit the fallback,
what loader flags were used, and an append-only time-to-ready history. Compare
**cold** runs only: host RAM is 1800 GB and the weights are ~1372 GB, so a restart
on the same node is served largely from page cache and looks fast whatever the
thread count says.

For an allocation you restart inside, `LOAD_FORMAT=presharded` is the big win —
the first run dumps a per-rank, already-quantised checkpoint and every later run
reads only its own 1/8 and skips re-quantisation. It costs one slow load plus up
to another ~1.4 TB of scratch. Untested here.

---

## The node is ROCm 7.14, the image is 7.2 — and that is fine

The image ships its own complete ROCm. The host's reaches into it through exactly
three channels, and all three have broken a run on these nodes:

1. **`--rocm`** binds the host's ROCm libraries into `/.singularity.d/libs` and
   *prepends* that to `LD_LIBRARY_PATH`, so the container's binaries run against
   the host's `libhsa`/`libamdhip`. When the two versions diverge, the
   container's own `rocminfo` starts exiting 1 and aiter dies on import with
   `Get GPU arch from rocminfo failed`.
2. **The kernel driver**, via `/dev/kfd`. A KFD ioctl ABI break is not fixable
   from userspace — that one genuinely needs a new image.
3. **Inherited environment.** A `*_VISIBLE_DEVICES` value the container's ROCr
   cannot parse — SLURM on a newer stack can hand out UUID-form lists like
   `GPU-a1b2…` — takes down agent enumeration for the *whole* container, which
   presents exactly like a dead driver. The script only forwards index-form
   lists and says so when it drops one.

So the script does not assume; it **probes**. `ROCM_MODE=auto` tries `--rocm`,
then a plain `/dev/kfd` + `/dev/dri` bind (which leaves the container's ROCm
userspace intact end to end), and caches the answer per node+image in
`$MODEL_CACHE_DIR/.rocm-mode`.

`./serve-qwen38.sh gpucheck` runs the same code path as a standalone diagnostic —
so what you debug with is what you run — and, crucially, **distinguishes two
failures that look identical**:

* *torch saw 0 devices in every mode* → the container really cannot reach the
  GPUs. Needs a new image.
* *torch saw 8 devices but aiter could not name the architecture* → only the
  image's `rocminfo` is broken. **Does not need a new image.** The fix is the
  `ROCMINFO_SHIM`, which snapshots the host's working `rocminfo` output and binds
  a one-line script replaying it over the container's binary. It is real output
  for this node, so whatever aiter's parser expects, it gets, and nothing links
  against it.

Conflating those two sends you into a multi-hour image rebuild for a one-line fix.

---

## The read-only `.sif` problem

The MXFP4 MoE JIT-compiles FlyDSL kernels at CUDA-graph capture time and writes
them **inside** the image, under `aiter/jit/`. An Apptainer `.sif` is read-only,
so that write dies with:

```
OSError: [Errno 30] Read-only file system:
  '/sgl-workspace/aiter/aiter/jit/flydsl_cache/launch_hgemm_kernel_*/*.lock'
```

…and it dies **after** the full ~1.4 TB weight load.

The fix: seed a scratch copy of the image's whole `jit/` directory (not just
`flydsl_cache` — the image ships prebuilt `module_*.so` there that an empty bind
would hide), then bind it back at the *same* path, because compiled artefacts can
embed absolute paths. Compiled kernels then persist across runs.

The seed marker records which image it came from, so changing `SGLANG_IMAGE`
re-seeds automatically. A `jit/` seeded from one image and bound over another is
how you get `module_*.so: undefined symbol` at graph capture.

And the preflight that makes this cheap: just before launch, the script writes a
test file into `flydsl_cache` **through the exact same `apptainer exec` argv the
server will use**. Testing a separately-built command is how a preflight passes
while the real launch is missing a bind.

---

## Upstream drift — how to re-check

```bash
# newest mi35x image tags (do NOT substitute the mi30x stream — different arch AND ROCm)
curl -s 'https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags?page_size=100' \
  | python3 -c 'import json,sys; [print(t["name"], t["last_updated"][:10]) for t in json.load(sys.stdin)["results"] if "mi35x" in t["name"]]'

# has the canonical MXFP4 checkpoint gone public?
curl -s -o /dev/null -w '%{http_code}\n' https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8-MXFP4

# have the weights changed under you?
curl -s https://huggingface.co/api/models/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["sha"], d["lastModified"])'

# has the DSpark draft been rewritten? (pin DSPARK_REVISION if it starts moving)
curl -s https://huggingface.co/api/models/RadixArk/Qwen3.8-2.4T-A95B-DSpark \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["sha"], d["lastModified"])'

# does a candidate image carry the HIP renorm kernels? (the speculative landmine)
apptainer exec <candidate.sif> grep -c 'top_._renorm_probs_triton' \
  /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py
```

Migrate to a new image into a **second** `SIF_PATH` so the working one is
preserved, then re-verify `check` → `parsers` → `toolcheck` → a full bench before
switching over. A new mainline image can move the attention, sampling and
speculative paths all at once.

---

## Running on other hardware

* **MI350X** — identical. Same gfx950, same 288 GB, same `mi35x` image, and
  upstream emits the same command. Nothing in this repo needs changing.
* **MI300X** — will not work. CDNA3 (gfx942) has no hardware MX matmul, so it
  cannot run an MXFP4 checkpoint at all. Upstream's MI300X recipe is **FP8 across
  two nodes** with `--kv-cache-dtype fp8_e4m3`, `--page-size 16`,
  `--disable-custom-all-reduce` on every rank, `--pp-size 2`, and the
  `mi30x`/ROCm-7.00 image. That is a different recipe, not a tweak to this one.
  `serve-qwen38.sh` detects gfx942 and says so.
* **NVIDIA** — use the cookbook directly. NVFP4 on B300/GB300, and the launch
  image is `lmsysorg/sglang:qwen38`.

---

## Script reference

```
./serve-qwen38.sh [serve]        start serving (default), stays attached
./serve-qwen38.sh serve --detach start serving, wait until healthy, return the shell
./serve-qwen38.sh pull           build the .sif from the container image (one-time)
./serve-qwen38.sh check          checkpoint reachable? image can load it? (BEFORE download)
./serve-qwen38.sh gpucheck       can this image reach this node's GPUs? (~1 min)
./serve-qwen38.sh parsers        list tool-call/reasoning parsers this image supports
./serve-qwen38.sh toolcheck      reasoning + two-turn tool round trip vs a running server
./serve-qwen38.sh loadstat       why the last cold start was slow + time-to-ready history
./serve-qwen38.sh download       prefetch weights (+ DSpark draft if enabled)
./serve-qwen38.sh stop           stop the server
./serve-qwen38.sh status         server state + health check + /v1/models

./opencode-setup.sh  [--host H] [--port P] [--model M] [--context N]
                     [--api-key K] [--embed-key] [--config PATH]

./qwencode-setup.sh  [--host H] [--port P] [--model M] [--context N]
                     [--api-key K] [--no-key] [--env-only] [--config PATH]

./bench-qwen38.sh [sweep]        1024/512 tok/s at concurrency 2/8/32
./bench-qwen38.sh latency        single-stream latency only
./bench-qwen38.sh throughput     saturate at BENCH_MAX_CONCURRENCY
./bench-qwen38.sh longcontext    100k/512 at c=1 — the agentic shape
```

Handy extras:

```bash
tail -f $MODEL_CACHE_DIR/qwen38-server.log          # follow startup / requests
head -4 $MODEL_CACHE_DIR/qwen38-server.log          # the exact launch command used
srun --overlap --jobid <jobid> --pty /bin/bash -l   # second shell on the serving node
```

---

## Configuration reference

All knobs live in `qwen38.env` (copied from `qwen38-env.example`). Anything you
`export` beforehand takes precedence. The file itself carries the full reasoning
for each default; this table is the index.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CACHE_DIR` | *(required)* | Scratch path for HF cache + `.sif` + API key + log (~1.5 TB) |
| `HF_TOKEN` / `HF_TOKEN_FILE` | *(required)* | HuggingFace token, inline or from a file |
| `MODEL_ID` | `Qwen/…-FP8-MXFP4` | The validated repo — **unavailable (401/404)**, see above |
| `MODEL_CANDIDATES` | *(3 repos)* | What `check` probes |
| `SERVED_MODEL_NAME` | `qwen3.8` | Name clients use in the `model` field |
| `WEIGHTS_GB` | `1372` | Sizing for the space preflight and the GB/s readout |
| `SGLANG_IMAGE` | `…rocm720-mi35x-20260812` | Pinned gfx950 image. Never the `mi30x` one |
| `SIF_PATH` | `$MODEL_CACHE_DIR/qwen38-mi355x.sif` | Where the Apptainer image is stored |
| `AITER_JIT_DIR` / `_TARGET` | *(auto)* | Writable copy of aiter's `jit/`, bound over the in-image one |
| `QWEN38_API_KEY` | *(auto-generated)* | Bearer key; saved to `$MODEL_CACHE_DIR/qwen38-api-key` |
| `PORT` | `30000` | Endpoint port on the node |
| `TP_SIZE` | `8` | Tensor parallel = **total GPU count** |
| `DP_SIZE` | `1` | dp-attention groups; must divide `TP_SIZE` |
| `CONTEXT_LEN` | *(empty)* | Empty = model max (262,144). Set `131072` if pools won't allocate |
| `MEM_FRACTION` | `0.9` | **Do not lower.** aiter scales it by 0.85 → ≈0.765 |
| `ATTENTION_BACKEND` | *(empty)* | Empty = no flag, as upstream |
| `QUANTIZATION` | *(empty)* | Empty = resolve from the checkpoint's `quant_method` |
| `TOOL_PARSER` | `qwen3_coder` | Tool-call parser |
| `REASONING_PARSER` | `qwen3` | Not optional in practice — reasoning cannot be disabled |
| `DISABLE_RADIX_CACHE` | `0` | Also load-bearing for the GDN `extra_buffer` strategy |
| `SPECULATIVE` | *(empty)* | `nextn` (in-checkpoint MTP) or `dspark` (separate draft) |
| `MAX_RUNNING_REQUESTS` | *(empty)* | **Pin it when speculative is on** — else it silently becomes 48 |
| `REPLAYSSM_SPEC` | `0` | `--enable-linear-replayssm-spec`; linear-chain only |
| `MAMBA_FULL_MEMORY_RATIO` | *(empty)* | GDN state pool vs KV pool — upstream's preferred lever |
| `MAX_MAMBA_CACHE_SIZE` | *(empty)* | Hard cap in **slots**, not requests |
| `MAMBA_RADIX_STRATEGY` | *(empty)* | auto → `extra_buffer` (5 slots/request) |
| `MAMBA_SSM_DTYPE` | *(empty)* | Checkpoint default is fp32; `bfloat16` halves the pool |
| `INT8_MAMBA_CHECKPOINT` | `0` | ~2× cached-prefix capacity at fixed memory. Untested |
| `ROCM_MODE` | `auto` | `rocm` / `devices` GPU passthrough; probed and cached |
| `ROCMINFO_SHIM` | `auto` | Replays the host's `rocminfo` when the image's is broken |
| `ENABLE_AITER` | `1` | Sets `SGLANG_USE_AITER=1` — the cell's one required env var |
| `FLYDSL_FORCE` | `0` | `AITER_FLYDSL_FORCE`. Ours, not upstream's. Rung 1 of the tuning ladder |
| `SET_CPU_AFFINITY` | `0` | Matches SGLang's own default. NUMA binding still runs — see above |
| `WEIGHT_LOAD_THREADS` | `8` | Suppresses SGLang's single-threaded fallback |
| `LOAD_FORMAT` / `PRESHARDED_PATH` | *(empty)* | `presharded` for allocations you restart in |
| `EXTRA_ENGINE_ARGS` | *(empty)* | Appended verbatim, e.g. `--ep-size 8` |

---

## Troubleshooting

**I edited `qwen38.env` and nothing changed.**
You sourced it before editing. Every line is `export VAR="${VAR:-default}"` so
that shell exports win; once `source qwen38.env` has exported the old value,
`${VAR:-<your new value>}` sees it and your edit is discarded — silently, in that
shell, forever. Hit on bun161 on 14 Aug 2026, where an edited `MODEL_ID` had no
effect and `check` then advised setting the very thing that had just been set.

The scripts now detect this and print the offending variable, both values, and
the fix:

```
[qwen38 WARN] An exported variable is OVERRIDING /…/qwen38.env:
    MODEL_ID    file: amd/Qwen3.8-2.4T-A95B-Quark-MXFP4    in use: Qwen/…-FP8-MXFP4
  … Clear them:
      unset MODEL_ID
```

**You do not need to source `qwen38.env` at all** — every script reads it itself.
Source it only in a shell that wants `$MODEL_CACHE_DIR` for its own commands, and
remember that doing so pins those values in that shell.

**`check` says UNREACHABLE (401) or NOT FOUND (404) for `Qwen/…-FP8-MXFP4`.**
Expected as of 14 Aug 2026 — 401 anonymously, 404 with a token that lacks access;
both mean you cannot pull it. Pick a candidate that reports OK, set `MODEL_ID` and
`WEIGHTS_GB` in `qwen38.env`, and re-run `check` **in a shell where `MODEL_ID` is
not exported** (see the entry above).

**`check` reports "0 architectures registered" / INCONCLUSIVE.**
Not a "no". Every model module failed to import. Look for repeated
`Ignore import error when loading sglang.srt.models.*` — a trailing `: ''` on
those lines is the aiter `AITER_JIT_DIR` trap below.

**`FileNotFoundError: [Errno 2] No such file or directory: ''`**
aiter's `get_user_jit_dir()` branches on `"AITER_JIT_DIR" in os.environ` rather
than on the variable having a *value*, then calls `os.makedirs("")`. Sourcing
`qwen38.env` (which sets it empty) is enough to poison any import of aiter inside
the container. Both scripts resolve it to a real path before every probe; if you
hit it in your own command, pass `--env AITER_JIT_DIR=/tmp/aiter-jit`.

**`OSError: [Errno 30] Read-only file system: …/aiter/jit/flydsl_cache/…`**
The aiter JIT bind is missing or wrong. `serve-qwen38.sh` preflights this in
seconds; if it still happens, set `AITER_JIT_TARGET` to the `jit/` directory
shown in the traceback.

**`module_*.so: undefined symbol` at graph capture.**
A `jit/` dir seeded from a different image. Delete `$AITER_JIT_DIR` and restart —
the script re-seeds automatically when `SGLANG_IMAGE` changes, but a hand-edited
path can escape that.

**`RuntimeError: Get GPU arch from rocminfo failed`**
Run `./serve-qwen38.sh gpucheck`. If torch sees 8 devices, this is a broken
`rocminfo`, not a broken container — set `ROCMINFO_SHIM=force`.

**`CPU number N is not eligible`**
`SET_CPU_AFFINITY` must stay `0` unless you allocate the whole node's CPUs.

**Replies contain raw `<think>` tags.**
`REASONING_PARSER` is wrong. Check `./serve-qwen38.sh parsers`; it should be
`qwen3`. `./serve-qwen38.sh toolcheck` asserts this.

**`toolcheck` turn 1 fails with `finish_reason=length`.**
The model never finished thinking, which looks like a parser failure but is not.
Qwen3.8 reasons at `xhigh` by default. Raise `max_tokens`, or send
`reasoning_effort: "low"`.

**Server dies mid-session with `TypeError: 'NoneType' object is not callable` or
`NameError: tree_speculative_sampling_target_only`.**
The speculative sampling landmine. See that section. Immediate mitigation: unset
`SPECULATIVE`.

**Throughput plateaus at 48 concurrent requests.**
The MRR-48 trap. Pin `MAX_RUNNING_REQUESTS`.

**KV / GDN pool allocation fails at startup.**
Set `CONTEXT_LEN=131072` **first**. Do not lower `MEM_FRACTION` below 0.9 — read
the memory budget section for why.

---

## Sources

* [SGLang Qwen3.8 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8)
  — the verified MI355X MXFP4 cell, the configuration tips, and the DSpark /
  ReplaySSM guidance.
* [SGLang and Miles Add Day-0 Support for Qwen3.8](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support)
  — architecture, and the NVIDIA benchmarks.
* [Day 0 Support for Qwen 3.8 on AMD Instinct GPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-8-on-amd-instinct-gpus.html)
* [`Qwen/Qwen3.8-2.4T-A95B`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) ·
  [FP8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) ·
  [`amd/…-Quark-MXFP4`](https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4) ·
  [`RadixArk/…-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-2.4T-A95B-DSpark)
* [`AMD_MI355X_Bunya_LLM_tools_KimiK3`](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_KimiK3)
  — where the Apptainer, ROCm-passthrough, aiter and GPFS findings were measured.

---

## License

MIT — see [LICENSE](LICENSE).
