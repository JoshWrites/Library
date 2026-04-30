#!/usr/bin/env bash
# E3 — RDNA1 Vulkan stability soak (2 hours)
#
# Question: are there crash-class bugs on Qwen2.5-Coder shapes when
# co-resident with the existing sidecars on the 5700 XT?
#
# Method: run Qwen2.5-Coder-3B Q4_K_M on port 11500 (Vulkan, co-resident
# with llama-secondary :11435 and llama-embed :11437). Drive synthetic
# edit-prediction traffic (200-tok prefix, 60-tok completion, every 2s)
# in parallel with low-rate ingest-shaped embed calls (every 30s).
#
# Output:
#   bench/e3-soak.log               — main run log
#   bench/e3-server.log             — llama-server stderr
#   bench/e3-edits.ndjson           — one line per edit-prediction request
#   bench/e3-embeds.ndjson          — one line per embed request
#   bench/e3-vram.csv               — VRAM samples every 60s
#   bench/e3-summary.txt            — one-line verdict at the end
#
# This script is designed to be launched in the background with
# `nohup` or `run_in_background`. It cleans up the test llama-server
# on exit. The production sidecars are NEVER stopped.

set -uo pipefail

BENCH_DIR="$(dirname "$(readlink -f "$0")")"
LLAMA_BIN="/home/levine/src/llama.cpp/llama-b8799-vulkan/llama-server"
MODEL="/home/levine/models/qwen2.5-coder-3b/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
TEST_PORT=11500

DURATION_SEC=7200          # 2 hours
EDIT_INTERVAL_SEC=2
EMBED_INTERVAL_SEC=30
VRAM_INTERVAL_SEC=60

LOG="${BENCH_DIR}/e3-soak.log"
SERVER_LOG="${BENCH_DIR}/e3-server.log"
EDITS_LOG="${BENCH_DIR}/e3-edits.ndjson"
EMBEDS_LOG="${BENCH_DIR}/e3-embeds.ndjson"
VRAM_CSV="${BENCH_DIR}/e3-vram.csv"
SUMMARY="${BENCH_DIR}/e3-summary.txt"
DMESG_BEFORE="${BENCH_DIR}/e3-dmesg-before.log"
DMESG_AFTER="${BENCH_DIR}/e3-dmesg-after.log"

: > "$LOG"
: > "$EDITS_LOG"
: > "$EMBEDS_LOG"
echo "epoch,vram_used_bytes,vram_free_bytes" > "$VRAM_CSV"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# --- Cleanup trap ---
SERVER_PID=""
EDIT_PID=""
EMBED_PID=""
VRAM_PID=""

cleanup() {
  local rc=$?
  log "Cleanup: tearing down workers and test server..."
  for pid in "$EDIT_PID" "$EMBED_PID" "$VRAM_PID"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -f "llama-server.*--port $TEST_PORT" 2>/dev/null || true
  log "Cleanup done. Exit code: $rc"
  exit $rc
}
trap cleanup EXIT INT TERM

# --- Sanity ---
[[ -x "$LLAMA_BIN" ]] || { log "ERROR: $LLAMA_BIN not executable"; exit 1; }
[[ -f "$MODEL" ]]    || { log "ERROR: $MODEL not found"; exit 1; }

for svc in llama-secondary.service llama-embed.service; do
  systemctl is-active --quiet "$svc" || { log "ERROR: $svc not active. Aborting."; exit 1; }
done

log "E3 start. Duration: ${DURATION_SEC}s. Model: $MODEL. Port: $TEST_PORT"

# --- Capture dmesg state before ---
dmesg --time-format iso 2>/dev/null | tail -200 > "$DMESG_BEFORE" || true

# --- Build code-prefix corpus from library/ source tree ---
log "Building code-prefix corpus from /home/levine/Documents/Repos/Library/library/..."
CORPUS_FILE="${BENCH_DIR}/e3-corpus.txt"
find /home/levine/Documents/Repos/Library/library -name '*.py' -type f -exec cat {} + > "$CORPUS_FILE" 2>/dev/null
CORPUS_BYTES=$(stat -c %s "$CORPUS_FILE")
log "Corpus: $CORPUS_BYTES bytes"
[[ "$CORPUS_BYTES" -lt 5000 ]] && { log "ERROR: corpus too small"; exit 1; }

# --- Launch test llama-server (co-resident with existing sidecars) ---
log "Launching Qwen2.5-Coder-3B on port $TEST_PORT (Vulkan1, ngl=99, c=4096)..."
"$LLAMA_BIN" \
  -m "$MODEL" \
  --device Vulkan1 -ngl 99 -c 4096 \
  --host 127.0.0.1 --port "$TEST_PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait up to 60 sec for server to be ready
ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 1 "http://127.0.0.1:${TEST_PORT}/v1/models" > /dev/null 2>&1; then
    ready=1; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "ERROR: server died during startup. Last 30 lines:"
    tail -30 "$SERVER_LOG" | tee -a "$LOG"
    exit 1
  fi
  sleep 0.5
done
[[ $ready -eq 1 ]] || { log "ERROR: server not ready after 60s"; exit 1; }
log "Test server ready."

# --- VRAM sampler ---
vram_loop() {
  while true; do
    local json total used free epoch
    json=$(rocm-smi --showmeminfo vram -d 0 --json 2>/dev/null)
    total=$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); c=next(iter(d.values())); print(c['VRAM Total Memory (B)'])" 2>/dev/null || echo 0)
    used=$(echo "$json"  | python3 -c "import json,sys; d=json.load(sys.stdin); c=next(iter(d.values())); print(c['VRAM Total Used Memory (B)'])" 2>/dev/null || echo 0)
    free=$(( total - used ))
    epoch=$(date +%s)
    echo "$epoch,$used,$free" >> "$VRAM_CSV"
    sleep "$VRAM_INTERVAL_SEC"
  done
}

# --- Edit-prediction worker ---
edit_loop() {
  local end=$1
  python3 - "$CORPUS_FILE" "$EDITS_LOG" "$TEST_PORT" "$end" "$EDIT_INTERVAL_SEC" <<'PY'
import sys, json, time, random, urllib.request

corpus_path, log_path, port, end_epoch, interval = sys.argv[1:]
end_epoch = float(end_epoch); interval = float(interval); port = int(port)

with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
    corpus = f.read()

# Pick character offsets that yield ~200-token prefixes (rough: 4 chars/tok).
# Slice ~800 chars to approximate 200 tokens.
PREFIX_CHARS = 800

n = 0
while time.time() < end_epoch:
    start = random.randint(0, max(0, len(corpus) - PREFIX_CHARS - 1))
    prefix = corpus[start:start+PREFIX_CHARS]
    body = json.dumps({
        "messages": [
            {"role": "user", "content": "Continue this Python code naturally:\n\n" + prefix}
        ],
        "max_tokens": 60,
        "temperature": 0.2,
        "stream": False
    }).encode()
    t0 = time.time()
    rec = {"t": t0, "n": n}
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            rec["status"] = resp.status
            rec["tokens"] = tokens
            rec["dur_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:
        rec["status"] = -1
        rec["error"] = repr(e)[:200]
        rec["dur_ms"] = int((time.time() - t0) * 1000)
    with open(log_path, "a") as g:
        g.write(json.dumps(rec) + "\n")
    n += 1
    # Sleep the remainder of the interval if we have time
    elapsed = time.time() - t0
    sleep_for = max(0, interval - elapsed)
    time.sleep(sleep_for)
PY
}

# --- Embed worker (background ingest cadence) ---
embed_loop() {
  local end=$1
  python3 - "$EMBEDS_LOG" "$end" "$EMBED_INTERVAL_SEC" <<'PY'
import sys, json, time, urllib.request

log_path, end_epoch, interval = sys.argv[1:]
end_epoch = float(end_epoch); interval = float(interval)

texts = ["the quick brown fox jumps over the lazy dog " * 8 for _ in range(16)]
body = json.dumps({"input": texts, "model": "e5"}).encode()

n = 0
while time.time() < end_epoch:
    t0 = time.time()
    rec = {"t": t0, "n": n}
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11437/v1/embeddings",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            rec["status"] = resp.status
            rec["dur_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:
        rec["status"] = -1
        rec["error"] = repr(e)[:200]
        rec["dur_ms"] = int((time.time() - t0) * 1000)
    with open(log_path, "a") as g:
        g.write(json.dumps(rec) + "\n")
    n += 1
    elapsed = time.time() - t0
    sleep_for = max(0, interval - elapsed)
    time.sleep(sleep_for)
PY
}

# --- Launch workers ---
END_EPOCH=$(($(date +%s) + DURATION_SEC))
log "End epoch: $END_EPOCH ($(date -d @$END_EPOCH -Iseconds))"

vram_loop &
VRAM_PID=$!

edit_loop "$END_EPOCH" &
EDIT_PID=$!

embed_loop "$END_EPOCH" &
EMBED_PID=$!

log "Workers up. PIDs: server=$SERVER_PID edit=$EDIT_PID embed=$EMBED_PID vram=$VRAM_PID"
log "Soak running. Will report periodically."

# --- Periodic progress prints (every 15 min) ---
PROGRESS_INTERVAL=900
NEXT_PROGRESS=$(($(date +%s) + PROGRESS_INTERVAL))

while [[ $(date +%s) -lt $END_EPOCH ]]; do
  sleep 30

  # Check server still alive
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "ERROR: test server died at $(date -Iseconds). See $SERVER_LOG"
    break
  fi

  # Periodic progress
  if [[ $(date +%s) -ge $NEXT_PROGRESS ]]; then
    NEXT_PROGRESS=$(( $(date +%s) + PROGRESS_INTERVAL ))
    edits_n=$(wc -l < "$EDITS_LOG")
    embeds_n=$(wc -l < "$EMBEDS_LOG")
    edits_fail=$(grep -c '"status": -1\|"status": [45]' "$EDITS_LOG" 2>/dev/null || echo 0)
    embeds_fail=$(grep -c '"status": -1\|"status": [45]' "$EMBEDS_LOG" 2>/dev/null || echo 0)
    last_vram=$(tail -1 "$VRAM_CSV" | cut -d, -f2)
    last_vram_gb=$(python3 -c "print(round(${last_vram:-0}/1e9, 2))")
    log "Progress: edits=$edits_n (fail=$edits_fail) embeds=$embeds_n (fail=$embeds_fail) vram=${last_vram_gb}GB"
  fi
done

log "Duration window closed. Stopping workers..."
kill "$EDIT_PID" "$EMBED_PID" "$VRAM_PID" 2>/dev/null || true
wait "$EDIT_PID" "$EMBED_PID" "$VRAM_PID" 2>/dev/null || true

# --- Capture dmesg state after ---
dmesg --time-format iso 2>/dev/null | tail -200 > "$DMESG_AFTER" || true

# --- Summary ---
log "Computing summary..."
python3 <<PY | tee "$SUMMARY" | tee -a "$LOG"
import json, csv
from pathlib import Path

bench = Path("$BENCH_DIR")
edits = [json.loads(l) for l in (bench/"e3-edits.ndjson").read_text().splitlines() if l.strip()]
embeds = [json.loads(l) for l in (bench/"e3-embeds.ndjson").read_text().splitlines() if l.strip()]

def summarize(name, rows):
    n = len(rows)
    fail = sum(1 for r in rows if r.get("status") in (-1,) or (isinstance(r.get("status"), int) and r["status"] >= 400))
    durs = [r["dur_ms"] for r in rows if r.get("status") == 200]
    durs.sort()
    p50 = durs[len(durs)//2] if durs else 0
    p95 = durs[int(len(durs)*0.95)] if durs else 0
    print(f"{name}: n={n} ok={n-fail} fail={fail} p50={p50}ms p95={p95}ms")

print("--- E3 summary ---")
summarize("edit-prediction", edits)
summarize("embed", embeds)

# VRAM trend
vram_rows = list(csv.DictReader((bench/"e3-vram.csv").open()))
if vram_rows:
    vals = [int(r["vram_used_bytes"]) for r in vram_rows]
    print(f"VRAM used: start={vals[0]/1e9:.2f}GB  end={vals[-1]/1e9:.2f}GB  max={max(vals)/1e9:.2f}GB  delta={(vals[-1]-vals[0])/1e6:+.0f}MB")

# Server log scan
server_log = (bench/"e3-server.log").read_text()
err_lines = [l for l in server_log.splitlines() if any(k in l.lower() for k in ("error","crash","abort","oom","vk_error","fail"))]
print(f"Server stderr error-class lines: {len(err_lines)}")
for l in err_lines[:5]:
    print(f"  {l[:200]}")
if len(err_lines) > 5:
    print(f"  ... and {len(err_lines)-5} more")

# dmesg diff
before = set((bench/"e3-dmesg-before.log").read_text().splitlines())
after  = set((bench/"e3-dmesg-after.log").read_text().splitlines())
new_lines = [l for l in after - before if any(k in l.lower() for k in ("amdgpu","sdma","vulkan","gpu fault","ring","reset"))]
print(f"New GPU-relevant dmesg lines: {len(new_lines)}")
for l in new_lines[:5]:
    print(f"  {l[:200]}")
PY

log "E3 complete."
