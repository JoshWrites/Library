#!/usr/bin/env bash
# E2 -- Cold-load timing for Qwen2.5-Coder GGUFs on the 5700 XT (Vulkan)
#
# Question: how long does it take to load each Qwen2.5-Coder size from
# page-cached GGUF into VRAM, including Vulkan shader compile? This sets
# the swap-penalty-per-direction budget for the swap-in design.
#
# Method: stop the production sidecars (so the 5700 XT is clean), then
# launch llama-server with each model 3 times, timing process-start to
# first /v1/models 200 response. Issue one inference request per load to
# confirm the model serves and capture token/sec.
#
# Output: bench/coder-load-times-5700xt.csv with model,run,load_ms,first_token_ms,tok_per_sec.
#
# IMPORTANT: this stops llama-secondary.service and llama-embed.service
# during the run. They are restarted on exit (success or failure) via trap.

set -uo pipefail

BENCH_DIR="$(dirname "$(readlink -f "$0")")"
OUT_CSV="${BENCH_DIR}/coder-load-times-5700xt.csv"
LOG="${BENCH_DIR}/e2-run.log"
LLAMA_BIN="/home/levine/src/llama.cpp/llama-b8799-vulkan/llama-server"
TEST_PORT=11500
RUNS_PER_MODEL=3

declare -A MODELS=(
  [qwen2.5-coder-1.5b]="/home/levine/models/qwen2.5-coder-1.5b/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
  [qwen2.5-coder-3b]="/home/levine/models/qwen2.5-coder-3b/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
  [qwen2.5-coder-7b]="/home/levine/models/qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
)

# Stable iteration order (smallest -> largest)
MODEL_ORDER=(qwen2.5-coder-1.5b qwen2.5-coder-3b qwen2.5-coder-7b)

: > "$LOG"
echo "model,run,load_ms,first_token_ms,tokens_generated,tok_per_sec,exit_status" > "$OUT_CSV"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

restore_sidecars() {
  log "Restoring production sidecars..."
  systemctl start llama-secondary.service llama-embed.service 2>>"$LOG" || \
    log "WARN: failed to restart sidecars (may need manual systemctl start)"
  sleep 3
  for svc in llama-secondary.service llama-embed.service; do
    if systemctl is-active --quiet "$svc"; then
      log "  $svc: active"
    else
      log "  $svc: NOT ACTIVE -- manual intervention required"
    fi
  done
}

cleanup() {
  local rc=$?
  log "Cleanup: killing any test llama-server on port $TEST_PORT"
  pkill -f "llama-server.*--port $TEST_PORT" 2>/dev/null || true
  sleep 1
  restore_sidecars
  exit $rc
}
trap cleanup EXIT INT TERM

# --- Sanity checks ---
if [[ ! -x "$LLAMA_BIN" ]]; then
  log "ERROR: llama-server not found at $LLAMA_BIN"
  exit 1
fi
for m in "${MODEL_ORDER[@]}"; do
  if [[ ! -f "${MODELS[$m]}" ]]; then
    log "ERROR: missing model file: ${MODELS[$m]}"
    exit 1
  fi
done

log "E2 start. Bin: $LLAMA_BIN, port: $TEST_PORT, runs/model: $RUNS_PER_MODEL"

# --- Stop production sidecars ---
log "Stopping llama-secondary and llama-embed (will restore on exit)..."
systemctl stop llama-secondary.service llama-embed.service 2>>"$LOG"
sleep 3
for svc in llama-secondary.service llama-embed.service; do
  if systemctl is-active --quiet "$svc"; then
    log "ERROR: $svc still active after stop. Aborting."
    exit 1
  fi
done

# --- Pre-warm page cache for each GGUF (so disk I/O isn't measured) ---
log "Pre-warming page cache for all GGUFs..."
for m in "${MODEL_ORDER[@]}"; do
  cat "${MODELS[$m]}" > /dev/null
done

run_one() {
  local model_name="$1"
  local model_path="$2"
  local run_idx="$3"

  log "[$model_name run $run_idx] launching llama-server..."

  local server_log="${BENCH_DIR}/e2-${model_name}-run${run_idx}.log"
  local t_start t_ready

  t_start=$(date +%s.%N)
  "$LLAMA_BIN" \
    -m "$model_path" \
    --device Vulkan1 -ngl 99 \
    -c 4096 \
    --host 127.0.0.1 --port "$TEST_PORT" \
    > "$server_log" 2>&1 &
  local server_pid=$!

  # Poll /v1/models until 200, with a 120 sec hard timeout
  local ready=0
  for _ in $(seq 1 240); do
    if curl -fsS --max-time 1 "http://127.0.0.1:${TEST_PORT}/v1/models" > /dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "[$model_name run $run_idx] server died during load. Last 20 log lines:"
      tail -20 "$server_log" | tee -a "$LOG"
      echo "$model_name,$run_idx,,,,,died" >> "$OUT_CSV"
      return 1
    fi
    sleep 0.5
  done
  t_ready=$(date +%s.%N)

  if [[ $ready -eq 0 ]]; then
    log "[$model_name run $run_idx] TIMEOUT after 120s waiting for /v1/models"
    kill "$server_pid" 2>/dev/null || true
    echo "$model_name,$run_idx,,,,,timeout" >> "$OUT_CSV"
    return 1
  fi

  local load_ms
  load_ms=$(python3 -c "print(int(($t_ready - $t_start) * 1000))")
  log "[$model_name run $run_idx] ready in ${load_ms}ms"

  # Inference probe: short fixed prompt, 64 tokens out, measure first-token + total
  local probe_resp
  local t_inf_start t_inf_end
  t_inf_start=$(date +%s.%N)
  probe_resp=$(curl -fsS --max-time 60 \
    -X POST "http://127.0.0.1:${TEST_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number."}],"max_tokens":64,"temperature":0,"stream":false}' \
    2>>"$LOG")
  t_inf_end=$(date +%s.%N)

  local tokens
  tokens=$(echo "$probe_resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)

  local inf_ms tok_per_sec
  inf_ms=$(python3 -c "print(int(($t_inf_end - $t_inf_start) * 1000))")
  if [[ "$tokens" -gt 0 ]]; then
    tok_per_sec=$(python3 -c "print(round($tokens / ($t_inf_end - $t_inf_start), 2))")
  else
    tok_per_sec="0"
  fi

  log "[$model_name run $run_idx] inference: ${tokens} tok in ${inf_ms}ms = ${tok_per_sec} tok/s"

  echo "$model_name,$run_idx,$load_ms,$inf_ms,$tokens,$tok_per_sec,ok" >> "$OUT_CSV"

  # Shutdown
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  sleep 2
}

# --- Run the matrix ---
for model_name in "${MODEL_ORDER[@]}"; do
  for run in $(seq 1 "$RUNS_PER_MODEL"); do
    run_one "$model_name" "${MODELS[$model_name]}" "$run" || \
      log "[$model_name run $run] failed; continuing"
  done
done

# --- Summary ---
log "E2 complete. Computing summary..."
python3 <<PY | tee -a "$LOG"
import csv
from collections import defaultdict
runs = defaultdict(list)
with open("$OUT_CSV") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["exit_status"] != "ok":
            continue
        runs[row["model"]].append({
            "load_ms": int(row["load_ms"]),
            "tok_per_sec": float(row["tok_per_sec"]),
            "tokens": int(row["tokens_generated"]),
        })
print("\n--- E2 summary ---")
for model, rs in runs.items():
    if not rs:
        print(f"{model}: no successful runs")
        continue
    loads = [r['load_ms'] for r in rs]
    tps = [r['tok_per_sec'] for r in rs]
    print(f"{model}: n={len(rs)} successful runs")
    print(f"  load_ms   min={min(loads)}  max={max(loads)}  mean={sum(loads)/len(loads):.0f}")
    print(f"  tok_per_s min={min(tps):.1f}  max={max(tps):.1f}  mean={sum(tps)/len(tps):.1f}")
PY

log "E2 done. Sidecars will be restored by trap."
