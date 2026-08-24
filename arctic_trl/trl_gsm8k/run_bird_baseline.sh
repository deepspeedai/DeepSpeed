#!/usr/bin/env bash
# Native TRL async-GRPO BIRD baseline launcher: external `vllm serve` on 4 GPUs + accelerate-DDP trainer on 4 GPUs.
#
# TRL+Arctic must run disaggregated, so the baseline is disaggregated too for a fair comparison: trainer DDP on
# GPUs 0-3 (4 processes) and `vllm serve` (TP=4) on GPUs 4-7. This script owns the vllm-serve lifecycle (the trainer
# python must NOT launch it -- under `accelerate launch` there are N copies of that process). Teardown kills the
# server process group on exit.
#
# Env overrides: MODEL, BIRD_TRAIN_PARQUET, MAX_STEPS, NUM_PROMPTS, NUM_GEN, PER_DEVICE_BSZ, MAX_COMPLETION_LEN,
# MAX_MODEL_LEN, LR, SEED, METRICS_OUT, PORT, TRAINER_GPUS, SERVER_GPUS.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
HOST="127.0.0.1"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SEED="${SEED:-42}"
TRAINER_GPUS="${TRAINER_GPUS:-0,1,2,3}"
SERVER_GPUS="${SERVER_GPUS:-4,5,6,7}"
NUM_TRAINERS="$(awk -F, '{print NF}' <<<"${TRAINER_GPUS}")"
SERVER_TP="$(awk -F, '{print NF}' <<<"${SERVER_GPUS}")"
METRICS_OUT="${METRICS_OUT:-${HERE}/metrics_bird_baseline.json}"
SERVER_LOG="${HERE}/_server_bird_baseline.log"
BASE_URL="http://${HOST}:${PORT}"

# Spawned rollout child (AsyncRolloutWorker) pickles reward_funcs by module path -> bird_task must be importable.
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

echo "== BIRD baseline: vllm serve (TP=${SERVER_TP}) on GPUs ${SERVER_GPUS} + accelerate DDP (${NUM_TRAINERS}) on GPUs ${TRAINER_GPUS} =="
hostname; date -Iseconds

# Kill ONLY the given pid's subtree (never a process group): the autorun runner is this script's process-group
# leader, so a `kill -- -<pgid>` here would take the runner down with us. Recurse over child pids instead.
kill_tree() {
  local pid="$1" sig="$2" c
  for c in $(pgrep -P "${pid}" 2>/dev/null); do kill_tree "${c}" "${sig}"; done
  kill "-${sig}" "${pid}" 2>/dev/null || true
}

# ---- launch external vllm serve on the server GPUs (plain background child; kill by pid subtree on exit) ----
CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" VLLM_SERVER_DEV_MODE=1 \
  vllm serve "${MODEL}" \
    --host "${HOST}" --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${SERVER_TP}" \
    --seed "${SEED}" \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[base] vllm serve pid=${SERVER_PID}; log -> ${SERVER_LOG}"

cleanup() {
  echo "[base] terminating vllm serve pid=${SERVER_PID} (subtree) ..."
  kill_tree "${SERVER_PID}" TERM
  for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
  kill_tree "${SERVER_PID}" KILL
  echo "[base] ---- vllm serve log tail ----"; tail -40 "${SERVER_LOG}" 2>/dev/null; echo "[base] ---- end ----"
}
trap cleanup EXIT

# ---- wait for server health ----
echo "[base] waiting for ${BASE_URL}/health ..."
HEALTHY=0
for _ in $(seq 1 900); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "[base] vllm serve exited early"; exit 1; fi
  if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  sleep 2
done
[ "${HEALTHY}" = "1" ] || { echo "[base] server never healthy"; exit 1; }
echo "[base] vllm server healthy"; date -Iseconds

# ---- accelerate-DDP trainer on the trainer GPUs (connects to the external server) ----
# Run from HERE (trl_gsm8k): torchrun puts the launch cwd on sys.path[0], and the repo root contains a `trl/`
# source-checkout dir that would shadow the editable-installed `trl` package (ImportError: __version__). HERE has
# no `trl/`, so `import trl` resolves to the installed package; and bird_task stays importable via cwd.
cd "${HERE}"
CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" accelerate launch \
  --num_processes "${NUM_TRAINERS}" --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no \
  run_qwen3_1.7b_bird_grpo_baseline.py \
    --model "${MODEL}" \
    --vllm-base-url "${BASE_URL}" \
    --max-steps "${MAX_STEPS:-30}" \
    --num-generations "${NUM_GEN:-8}" \
    --per-device-bsz "${PER_DEVICE_BSZ:-8}" \
    --max-completion-length "${MAX_COMPLETION_LEN:-1024}" \
    --num-prompts "${NUM_PROMPTS:-128}" \
    --learning-rate "${LR:-1e-6}" \
    --seed "${SEED}" \
    --metrics-out "${METRICS_OUT}"
rc=$?
echo "[base] accelerate launch rc=${rc}"; date -Iseconds
exit ${rc}
