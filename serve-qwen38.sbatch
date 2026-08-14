#!/bin/bash --login
#SBATCH --job-name=qwen38-mi355x
#SBATCH --partition=admin_test
#SBATCH --account=a_rcc
#SBATCH --qos=sdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1800G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qwen38-%j.out
#SBATCH --error=logs/qwen38-%j.out
#
# Batch wrapper around serve-qwen38.sh for Bunya's MI355X nodes (bun159/160/161).
# Submit from /scratch (not /home, not RDM) with:
#
#   mkdir -p logs && sbatch serve-qwen38.sbatch
#
# The partition/account/qos/gres above are the discovered MI355X recipe:
#   --partition=admin_test  (the MI355X nodes live here, not gpu_rocm)
#   --account=a_rcc  --qos=sdf  --gres=gpu:mi355x:8
# The MI355X nodes are bun159/bun160/bun161; the gres lands you on a free one.
# Add e.g. '#SBATCH --nodelist=bun161' if you want to pin a specific node.
# Confirm the current recipe with:  sinfo -o "%P %.10l %G %f" | grep mi355x
#
# ALL EIGHT GPUs ARE REQUIRED. The MXFP4 checkpoint is ~1.4 TB against 8 x 288 GB
# of HBM, so there is no smaller layout that fits — TP8 on one node is the whole
# recipe.
#
# Walltime note: a cold start reads ~1.4 TB of weights off scratch and
# JIT-compiles MXFP4 kernels on first run, so budget a substantial slice of the
# allocation for startup alone. READY_TIMEOUT in qwen38.env (default 14400s)
# must stay below this walltime.
#
# Memory: --mem=1800G is host RAM for staging the weight load, not HBM.
#
# The connection banner (node, endpoint URL, API key location, tunnel command)
# is printed into the job log. Once the job is RUNNING:
#
#   grep -A 30 'is up and serving' logs/qwen38-<jobid>.out
#
# The server runs until walltime or 'scancel <jobid>' (the script traps SIGTERM
# and stops the container cleanly).
#
# BEFORE YOUR FIRST SUBMIT, in an interactive allocation:
#   ./serve-qwen38.sh pull        # build the .sif
#   ./serve-qwen38.sh check       # IS THE CHECKPOINT EVEN REACHABLE? (see README)
#   ./serve-qwen38.sh download    # ~1.4 TB
# Skipping 'check' is how you spend hours discovering the canonical MXFP4 repo
# still 401s for your token.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[qwen38-sbatch] Job $SLURM_JOB_ID on $(hostname) at $(date)"
exec ./serve-qwen38.sh serve
