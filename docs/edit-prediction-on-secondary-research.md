# Edit prediction on the secondary GPU — research → experiment → spec

Date: 2026-04-28 (research) → 2026-04-29 (V1 deployed)
Status: **V1 deployed.** Shape A (Qwen2.5-Coder-3B co-resident, no llama-swap) shipped via `llama-coder.service` on :11438. E1/E2/E3 complete. E5 (subjective quality eyeball) still TODO.
Adjacent doc: `audio-ingest-research.md` (same hardware envelope, different workload)

## V1 results (added 2026-04-29)

**E1 — VRAM headroom audit** (`bench/vram-baseline-5700xt.csv`):
- 8.57 GB total, 5.72 GB used by existing two sidecars warm, **2.85 GB free**.
- VRAM steady under embed load (KV caches pre-allocated, no growth).

**E2 — Cold-load times for Qwen2.5-Coder Vulkan** (`bench/coder-load-times-5700xt.csv`):
| Model | Load (page-cache warm) | tok/s |
|---|---|---|
| Qwen2.5-Coder-1.5B Q4_K_M | ~1.03 s | 150 |
| Qwen2.5-Coder-3B Q4_K_M | ~1.52 s | 99 |
| Qwen2.5-Coder-7B Q4_K_M | ~2.62 s | 57 |

**E3 — 2-hour stability soak** (`bench/e3-summary.txt`):
- Qwen2.5-Coder-3B + llama-secondary + llama-embed all co-resident on 5700 XT.
- Edit-prediction workload: every 2s, 200-tok prefix, 60-tok completion.
- Embed workload (concurrent): 16-batch every 30s.
- **3,600 / 3,600 edits + 240 / 240 embeds, zero failures.** p50/p95 = 777ms / 814ms (edit), 388ms / 583ms (embed).
- VRAM start = end = max = 8.12 GB. Zero drift, zero leak.
- Zero GPU-class dmesg events. RDNA1 Vulkan is production-grade for this workload.

**Decision:** Shape A confirmed. 3B fits co-resident with the existing two sidecars and meets latency budget without swap orchestration. llama-swap unnecessary for V1. 7B remains an option if E5 reveals 3B quality is insufficient — would force Shape B (sole-resident with eviction).

**Deployed config:**
- `/etc/systemd/system/llama-coder.service` — Qwen2.5-Coder-3B-Instruct Q4_K_M, Vulkan1, -ngl 99, -c 4096, port 11438.
- Ports registry updated (`Workstation/docs/ports-registry.md:39`).
- No Library code changes (this is a parallel sidecar, not a Library client).
- Zed client config: point `open_ai_compatible_api` at `http://127.0.0.1:11438/v1`, model `qwen2.5-coder-3b-instruct-q4_k_m.gguf`.

## Goal

Add an always-warm code-completion endpoint on the 5700 XT so Zed (and
any other OpenAI-compatible-edit-prediction client) gets local inference
without paying Baseten/Zed's hosted Zeta service. Library's existing
embed and summarize sidecars must continue to work, but they can become
swap-in workloads — the card's default resident becomes the coder model.

Constraint: **all-local, latency-sensitive on edit prediction, tolerant
on embed/summarize**.

## Hardware envelope (recap)

- **GPU1 — RX 7900 XTX, 24 GB, RDNA3, ROCm 7.2.1.** Primary chat model
  (`llama-primary` / `llama-second-opinion` on :11434). Untouched by
  this work.
- **GPU0 — RX 5700 XT, 8 GB, RDNA1 (gfx1010), ROCm 7.2.1.** Today hosts
  `llama-secondary` (Qwen3-4B summarizer, :11435) and `llama-embed`
  (multilingual-e5-large Q8_0, :11437) — verified co-resident under
  concurrent load on 2026-04-22 per ports registry. (Confirmed
  2026-04-28 against `systemctl status llama-embed.service`: ExecStart
  references `multilingual-e5-large-Q8_0.gguf`. The
  `fetch-mxbai-embed.sh` script in second-opinion is historical — :11437
  ran mxbai before migrating to multilingual-e5-large for better
  non-English coverage.)
- CPU: Ryzen 9 5950X.

RDNA1 reality (also covered in audio-ingest-research.md): ROCm PyTorch
on gfx1010 is broken since PyTorch ≥ 2.0. **Vulkan via llama.cpp is
the only mature path.** Anything that requires a PyTorch-on-ROCm wheel
is off the table.

## What "Zeta-equivalent" actually means

Zed's Zeta is a fine-tune of Qwen2.5-Coder-7B with three-stage training
(SFT → DPO → speculative decoding) specialized for *next-edit*
prediction (not just line-completion). Inference is stateless per
request — the editor packages a code window + recent-edit history and
sends it as one prompt. No server-side session state.

Realistic local options at the relevant sizes:

| Model | Params | Q4_K_M weights | Notes |
|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | ~1.0 GB | Tab-complete only |
| Qwen2.5-Coder-3B-Instruct | 3B | ~1.9 GB | Sweet spot for 8 GB co-residency |
| Qwen2.5-Coder-7B-Instruct | 7B | ~4.5 GB | Sole-resident only; closest to Zeta's base |
| Zeta-7B (Zed-published GGUF, if it exists) | 7B | ~4.5 GB | Adds next-edit fine-tune behavior |

There is **no Zed-published Zeta-3B**. "Zeta-3B" in shorthand below
means "Qwen2.5-Coder-3B as the coder slot" — the closest small model
in the same lineage, not a Zed artifact.

## Engine matrix (RDNA1-viable only)

| Stack | Backend | RDNA1 / Vulkan | VRAM (Q4_K_M) | Notes |
|---|---|---|---|---|
| llama.cpp Vulkan + Qwen2.5-Coder-3B | llama-server | Works (already proven on this card) | ~1.9 GB + KV | Drop-in next to existing :11435/:11437 |
| llama.cpp Vulkan + Qwen2.5-Coder-7B | llama-server | Same | ~4.5 GB + KV | Sole resident only |
| llama.cpp Vulkan + Zeta-7B GGUF | llama-server | Same | ~4.5 GB + KV | Conditional on a published GGUF |
| llama-swap proxy in front of all three | Go binary, OpenAI-compat | Trivial | 0 (proxy itself) | Lets us declare co-resident groups + TTLs |

## The architectural shift

Today the 5700 XT runs two co-resident sidecars under fixed systemd
units. Library posts to fixed ports and assumes the right model is up.

The proposed shift:

1. **Default resident = coder model**, always warm, served via
   llama-swap on a new front-door port.
2. **Embed and summarize become swap-in workloads.** Library's calls
   still go to the same OpenAI-compatible endpoints — but llama-swap
   intercepts, evicts the coder, loads embed (or summarize), serves,
   idles for TTL, reloads coder.
3. **Edit prediction is the latency-sensitive workload**, so it gets
   the warm slot. Embed/summarize tolerate cold starts (embed is
   batch-y, summarize is multi-second already).

This inverts the current "embed + summarize co-resident, no coder"
layout. We give up co-residency of embed and summarize to make room
for a third workload, and we lean on llama-swap's TTL knob to keep
ingest runs from thrashing.

### Concurrency edge

llama-swap serializes — one model resident at a time on this card. If
ingest is running and you start typing in Zed, edit prediction stalls
until the current embed/summarize finishes. Solo work: fine.
Multi-user (anny editing while ingest runs): occasional stalls. Worth
naming because it's a behavioral change, not just a config change.

## Resolved: what's actually on :11437

The ports registry briefly listed mxbai-embed-large as the running
embed model. Ground truth on 2026-04-28 from `systemctl status
llama-embed.service`:

```
ExecStart: llama-server -m /home/levine/models/multilingual-e5-large/multilingual-e5-large-Q8_0.gguf
           --device Vulkan1 -ngl 99 --embedding --pooling mean -c 512 ...
```

Library docs were correct. `Workstation/docs/ports-registry.md:38`
was stale and has been updated. The `fetch-mxbai-embed.sh` script in
the second-opinion repo is historical: :11437 ran mxbai-embed-large
first, then migrated to multilingual-e5-large for better non-English
coverage (both are 1024-dim, so the swap was dimensionally compatible
with whatever vector index already existed). The V1 spec below
targets the current model.

## Open questions to resolve via experiment

1. How much VRAM is actually free on the 5700 XT with the two current
   sidecars warm? (Driver overhead eats some of the nominal 8 GB.)
2. What's the cold-load time for Qwen2.5-Coder-3B and -7B over Vulkan
   on this card? (This is the swap penalty per direction.)
3. Does Qwen2.5-Coder-3B as edit-prediction backend feel good enough,
   or do we need to commit to 7B?
4. Does llama-swap's group/TTL behavior actually keep coder resident
   between Library calls without manual tuning?
5. Are there RDNA1-specific Vulkan crashes on Qwen2.5-Coder shapes
   (analogous to whisper.cpp issue #3611's fillBuffer crash)?

## Experiments

Ordered by **unattended-friendliness first**. Each experiment defines
what to measure, how to run it without supervision, and the decision
it unblocks.

### E1 — VRAM headroom audit (unattended, ~5 min)

**Question:** How much VRAM is free on the 5700 XT with both current
sidecars warm and serving?

**Method:**
1. Confirm both sidecars are up: `systemctl is-active llama-secondary llama-embed`.
2. Warm them: one summarize call to :11435, one embed call to :11437.
3. Sample VRAM 10× at 1 sec intervals: `for i in $(seq 1 10); do rocm-smi --showmeminfo vram -d 0 --json; sleep 1; done`.
4. Repeat under synthetic load: drive 16-batch embeds in a loop while
   sampling. Use the same shape Library uses (`embedder.py:51`).

**Output:** `bench/vram-baseline-5700xt.csv` with min/max/mean
VRAM-used per phase (idle / warm / under load).

**Decision:** Hard floor on coder model size. If <2.5 GB free at peak,
even 3B is risky and we're forced into the swap-in coder design from
day one. If >5 GB free, 7B as sole resident is plausible.

**Unattended:** fully — single shell script, no human eyeball needed.

### E2 — Cold-load timing for Qwen2.5-Coder via Vulkan (unattended, ~15 min)

**Question:** How long does it take to load Qwen2.5-Coder-{1.5B, 3B, 7B}
Q4_K_M from page-cached GGUF into VRAM on this card, including Vulkan
shader compile?

**Method:**
1. Pull GGUFs from HF (Qwen2.5-Coder-{1.5B,3B,7B}-Instruct-Q4_K_M).
   Pre-warm page cache by reading each file once into `/dev/null`.
2. With **no other sidecar running on the 5700 XT** (stop
   llama-secondary and llama-embed for clean measurement), launch
   `llama-server -m <gguf> --n-gpu-layers 999 --device Vulkan1 --port 11500`
   and timestamp from process start to first successful
   `/v1/models` 200 response.
3. Repeat 3× per model size to capture cold-vs-pipeline-cache-warm
   variance.
4. Issue one inference request after each load to confirm the model
   actually serves; record token/sec on a fixed prompt.

**Output:** `bench/coder-load-times-5700xt.csv` with model, run, load_ms,
first_token_ms, tok_per_sec.

**Decision:** Sets the budget for swap-penalty-per-direction. If 3B
loads in <1 sec, the swap-in design is great. If 5+ sec, every Library
call eats unacceptable latency on first request after coder eviction
and we need to rethink TTLs aggressively (or accept that ingest
batches need to be large enough to amortize).

**Unattended:** yes, but **requires stopping the production sidecars**
during the run, so schedule it for a window where ingest isn't
expected. Script should `systemctl stop` → bench → `systemctl start`
with a trap to ensure restart on failure.

### E3 — RDNA1 Vulkan stability soak on Qwen2.5-Coder (unattended overnight)

**Question:** Are there crash-class bugs on Qwen2.5-Coder shapes
analogous to whisper.cpp's fillBuffer crash (issue #3611)? RDNA1 SDMA
quirks have bitten this card before.

**Method:**
1. Start `llama-server` on Qwen2.5-Coder-3B Q4_K_M, Vulkan, port 11500.
2. Drive a synthetic edit-prediction workload: random 200-token
   prefixes from a real code corpus (use this repo's `library/`
   directory), one request every 2 sec for 8 hours. Log every
   non-200 response and every llama-server stderr line.
3. Repeat with Qwen2.5-Coder-7B if E1+E2 say it fits.

**Output:** `bench/coder-soak-5700xt.log` plus a one-line summary:
total requests, failures, crash count, peak VRAM.

**Decision:** Go/no-go on Vulkan as the production backend. A single
crash in 8 hours is enough to require investigating a patched build
before V1 ships.

**Unattended:** ideal overnight job. Set systemd timer or just
`nohup` and walk away.

### E4 — llama-swap group/TTL behavior under realistic load (unattended, ~1 hr)

**Question:** Does llama-swap's TTL behavior actually keep coder
resident between Library calls, or does it thrash?

**Method:**
1. Stand up llama-swap with a config declaring three upstreams:
   `qwen-coder-3b` (default, no TTL), `qwen3-4b-summarize` (TTL=300s),
   `embed` (TTL=300s).
2. Front-door port 11500. Coder requests → upstream :11501. Summarize
   requests → :11502. Embed requests → :11503. Each upstream
   llama-server pinned to the 5700 XT.
3. Replay a recorded mixed workload: alternating bursts of
   coder requests (sub-second, simulated keystrokes) and
   embed batches (16 texts every 30 sec, simulating ingest), with
   one summarize call every 5 minutes.
4. Log every model swap event from llama-swap and time-of-flight
   for each workload.

**Output:** `bench/llama-swap-thrash-test.csv` — swap count, mean
time-to-first-token per workload, p95 latency on coder requests
during embed bursts.

**Decision:** Confirms or kills the design. If coder p95 latency
during ingest is acceptable (<2 sec including swap), we ship. If
coder requests routinely wait >5 sec for embed to finish, the
serialization cost is real and we either accept it, isolate ingest
to off-hours, or buy a bigger card.

**Unattended:** yes — synthetic workload runs from a script. Real
Zed traffic comes later in E5.

### E5 — Quality eyeball: 3B vs 7B on real edit-prediction (manual, 1–2 hrs)

**Question:** Is Qwen2.5-Coder-3B good enough for daily editing, or
do we need 7B?

**Method:** point Zed's `open_ai_compatible_api` at llama-swap. Use
the editor for a normal half-day of work with 3B. Switch to 7B (sole
resident, evicts the others) for the next half-day. Note subjective
quality, missed predictions, hallucinated APIs.

**Output:** a paragraph of notes in this doc, plus a vote.

**Decision:** Final coder model size for V1. This is the only
inherently-manual experiment — no synthetic benchmark substitutes
for "does it feel right while typing."

**Unattended:** no. Save for after E1–E4 settle the technical
feasibility.

## Spec sketch (post-experiment, conditional on results)

Pending E1–E5, the V1 shape is:

### Services

- **llama-swap** on `127.0.0.1:11500` (new port, register before
  binding per ports-registry rule).
- **upstream-coder** llama-server, Qwen2.5-Coder-{3B|7B} Q4_K_M,
  Vulkan, pinned to 5700 XT, port 11501. Default resident, no TTL.
- **upstream-summarize** llama-server, Qwen3-4B Q4_K_M (existing
  model), Vulkan, pinned to 5700 XT, port 11502. TTL TBD from E4.
- **upstream-embed** llama-server, multilingual-e5-large Q8_0 (current
  prod), Vulkan, pinned to 5700 XT, port 11503. TTL TBD from E4.

### llama-swap config (sketch)

```yaml
listen: 127.0.0.1:11500
models:
  qwen-coder:
    cmd: llama-server -m /var/lib/llama/qwen2.5-coder-3b-q4_k_m.gguf
         --port 11501 --device Vulkan1 --n-gpu-layers 999
    proxy: http://127.0.0.1:11501
    ttl: 0  # default resident
  qwen3-summarize:
    cmd: llama-server -m /var/lib/llama/qwen3-4b-q4_k_m.gguf
         --port 11502 --device Vulkan1 --n-gpu-layers 999
    proxy: http://127.0.0.1:11502
    ttl: 300  # tune from E4
  embed:
    cmd: llama-server -m /home/levine/models/multilingual-e5-large/multilingual-e5-large-Q8_0.gguf
         --port 11503 --device Vulkan1 -ngl 99 --embedding --pooling mean -c 512
    proxy: http://127.0.0.1:11503
    ttl: 300
groups:
  default:
    swap: true  # one resident at a time on the 5700 XT
    members: [qwen-coder, qwen3-summarize, embed]
```

### Library client changes

- `library/embedder.py:16` → point at `http://127.0.0.1:11500/v1/embeddings`, model name `embed`.
- `library/summarizer.py:15` → point at `http://127.0.0.1:11500/v1/chat/completions`, model name `qwen3-summarize`.

(llama-swap routes by the `model` field in the request body, so
Library has to start sending it. Today's clients post to fixed ports
and don't specify a model.)

### Zed client changes

- `open_ai_compatible_api` → `http://127.0.0.1:11500/v1`, model `qwen-coder`.

### Ports registry update

Before any of this binds:

- Add :11500 = llama-swap front door.
- Mark :11435 (llama-secondary) and :11437 (llama-embed) as
  superseded by llama-swap upstreams.
- Add :11501–:11503 as llama-swap-managed upstreams (not consumer-facing).

### Verification (the gap from the prior analysis)

Add a healthcheck that asserts the serving card. Two options:

1. `rocm-smi --showpids` after each upstream starts, fail loud if
   PID isn't on card 1.
2. Read `/proc/<pid>/environ` for `HIP_VISIBLE_DEVICES` /
   `ROCR_VISIBLE_DEVICES` and assert.

Either belongs in the systemd unit's ExecStartPost, not just docs.

## Out of scope for V1

- Speculative decoding (Zeta's n-gram trick). Worth revisiting if
  cold-load isn't the bottleneck and tokens/sec is.
- Multi-user concurrency on edit prediction (anny + levine typing
  simultaneously). llama-swap serializes; this is a known-limitation
  to revisit if it bites.
- Replacing Zed's edit-prediction protocol with a custom one tuned
  for next-edit (vs. completion). Stay on `open_ai_compatible_api`
  for V1.
- Buying hardware. The whole point is "use the card you have."
