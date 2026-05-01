#!/usr/bin/env bash
# E1 -- VRAM headroom audit for the 5700 XT (gfx1010, GPU0 in rocm-smi enumeration)
#
# Question: how much VRAM is free on the 5700 XT with the two existing
# sidecars (llama-secondary :11435, llama-embed :11437) warm and serving?
#
# Output: bench/vram-baseline-5700xt.csv with min/max/mean per phase.

set -uo pipefail

BENCH_DIR="$(dirname "$(readlink -f "$0")")"
OUT_CSV="${BENCH_DIR}/vram-baseline-5700xt.csv"
LOG="${BENCH_DIR}/e1-run.log"
GPU=0  # rocm-smi index for the 5700 XT (verified via --showproductname)

: > "$LOG"
echo "phase,sample,vram_used_bytes,vram_total_bytes,vram_used_pct" > "$OUT_CSV"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

sample_vram() {
  # rocm-smi --showmeminfo vram --json gives us {"card0":{"VRAM Total Memory (B)":..., "VRAM Total Used Memory (B)":...}}
  local phase="$1"
  local n_samples="$2"
  for i in $(seq 1 "$n_samples"); do
    local json
    json=$(rocm-smi --showmeminfo vram -d "$GPU" --json 2>/dev/null)
    local total used
    total=$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); c=next(iter(d.values())); print(c['VRAM Total Memory (B)'])" 2>/dev/null)
    used=$(echo "$json"  | python3 -c "import json,sys; d=json.load(sys.stdin); c=next(iter(d.values())); print(c['VRAM Total Used Memory (B)'])" 2>/dev/null)
    if [[ -z "$total" || -z "$used" ]]; then
      log "WARN: failed to parse rocm-smi sample $i in phase $phase"
      continue
    fi
    local pct
    pct=$(python3 -c "print(round($used/$total*100, 2))")
    echo "$phase,$i,$used,$total,$pct" >> "$OUT_CSV"
    sleep 1
  done
}

log "E1 start. GPU index=$GPU (5700 XT). Output: $OUT_CSV"

# --- Confirm both sidecars are running ---
for svc in llama-secondary.service llama-embed.service; do
  if ! systemctl is-active --quiet "$svc"; then
    log "ERROR: $svc is not active. Aborting."
    exit 1
  fi
done

# --- Phase 1: cold idle (no recent requests) ---
log "Phase 1: cold idle baseline (10 samples)"
sample_vram "cold_idle" 10

# --- Phase 2: warm both sidecars with a single request each ---
log "Phase 2: warming sidecars (1 embed + 1 summarize call)"
curl -fsS -X POST http://127.0.0.1:11437/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":"hello world","model":"e5"}' > /dev/null 2>&1 \
  || log "WARN: embed warm-up call failed"

curl -fsS -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":5}' > /dev/null 2>&1 \
  || log "WARN: summarize warm-up call failed"

sleep 2
log "Phase 2: warm-idle samples (10)"
sample_vram "warm_idle" 10

# --- Phase 3: synthetic embed load (16-batch, like Library uses) ---
log "Phase 3: synthetic embed load (16-batch in background)"

embed_load_loop() {
  local payload
  payload=$(python3 -c "
import json
texts = ['the quick brown fox jumps over the lazy dog ' * 8 for _ in range(16)]
print(json.dumps({'input': texts, 'model': 'e5'}))
")
  for _ in $(seq 1 30); do
    curl -fsS -X POST http://127.0.0.1:11437/v1/embeddings \
      -H 'content-type: application/json' \
      -d "$payload" > /dev/null 2>&1 || true
  done
}

embed_load_loop &
LOAD_PID=$!

sleep 1  # let the first batch hit
log "Phase 3: under-load samples (10)"
sample_vram "under_load" 10

wait "$LOAD_PID" 2>/dev/null || true
log "Phase 3: done. Embed loop exited."

# --- Summary ---
log "E1 complete. Computing summary..."
python3 <<PY | tee -a "$LOG"
import csv
from collections import defaultdict
phases = defaultdict(list)
with open("$OUT_CSV") as f:
    r = csv.DictReader(f)
    for row in r:
        phases[row["phase"]].append((int(row["vram_used_bytes"]), int(row["vram_total_bytes"])))
print("\n--- E1 summary ---")
for phase, rows in phases.items():
    used = [u for u,_ in rows]
    total = rows[0][1]
    free = [total - u for u in used]
    print(f"{phase}: n={len(rows)}")
    print(f"  used  min={min(used)/1e9:.2f}GB  max={max(used)/1e9:.2f}GB  mean={sum(used)/len(used)/1e9:.2f}GB")
    print(f"  free  min={min(free)/1e9:.2f}GB  max={max(free)/1e9:.2f}GB  mean={sum(free)/len(free)/1e9:.2f}GB")
print(f"\nVRAM total: {total/1e9:.2f}GB")
PY

log "E1 done."
