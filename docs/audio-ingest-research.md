# Audio ingest -- engine landscape (research notes)

Date: 2026-04-28
Status: research only, no design committed yet

## Goal

Add audio sources (multi-speaker, mixed Hebrew/English with code-switching)
to Library's existing chunk -> embed -> summarize pipeline. Entry point is
`library_read_file` extended by extension (matches how docling handles
binary docs today). The new piece is `audio file -> diarized,
language-tagged transcript -> markdown`; everything downstream stays the
same.

## Hardware constraint

- **GPU1 -- RX 7900 XTX, 24GB, RDNA3, ROCm 7.2.1.** Primary card. Holds
  the chat model. Must stay free so opencode keeps running in parallel.
- **GPU0 -- RX 5700 XT, 8GB, RDNA1 (gfx1010), ROCm 7.2.1.** Where audio
  ingest needs to live in v1. RDNA1 is officially deprecated in ROCm and
  most PyTorch-based stacks are broken or community-patched here.
- CPU: Ryzen 9 5950X -- comfortably fast for CPU inference fallback.

## Engine matrix (filtered to RDNA1-viable + CPU)

| Stack | STT+Diar bundled? | Backend | RDNA1 / Vulkan | RDNA1 / ROCm | VRAM fp16 | Hebrew | Code-switch | License | Notes |
|---|---|---|---|---|---|---|---|---|---|
| whisper.cpp Vulkan large-v3-turbo q5_0 | STT only | GGML+Vulkan | Works (mind RDNA1 SDMA fillBuffer crash, issue #3611, patch exists) | n/a | ~1.5 GB | Vanilla Whisper, mediocre on HE | Yes (Whisper-native) | MIT | Sweet-spot size |
| whisper.cpp Vulkan large-v3 q5_0 | STT only | GGML+Vulkan | Same | n/a | ~1.8-2.3 GB | Vanilla, OK | Yes | MIT | Slower than turbo |
| ivrit-ai whisper-large-v3-turbo-ct2 | STT only | CTranslate2 | n/a (CPU only on RDNA1; ROCm fork targets RDNA3) | No usable path | int8 ~1.7 GB / fp16 ~3.2 GB | **Best Hebrew (Apache-2.0, trained on 22k+ hr)** | Yes (Whisper base) | Apache-2.0 | CPU on Ryzen 5950X ~= 1-2x RT |
| ivrit-ai large-v3-ct2 (full) | STT only | CTranslate2 | CPU only | No | int8 ~3.1 GB / fp16 ~6.5 GB | Best HE, slower | Yes | Apache-2.0 | Fall back if turbo accuracy insufficient |
| pyannote 3.1 (speaker-diarization-3.1) | Diar only | PyTorch | CPU works | Broken on gfx1010 | ~1.6 GB if GPU | Language-agnostic | n/a | MIT (HF-gated) | CPU realtime-ish on Ryzen |
| pyannote 4.0 / community-1 | Diar only | PyTorch | CPU only realistic | No | ~9.5 GB peak (issue #1963) | Language-agnostic | n/a | MIT (HF-gated) | Won't co-tenant on 8 GB GPU0 |
| resemblyzer | Diar (clustering DIY) | PyTorch CPU | CPU | CPU | <1 GB | n/a | n/a | Apache-2.0 | Lightweight, drops on overlap |
| simple-diarizer | Diar | SpeechBrain xvec/ECAPA + AHC/SC | CPU | CPU | <1 GB | n/a | n/a | MIT | Tiny, ages well |
| 3D-Speaker (modelscope) | Diar pipeline | ONNX Runtime | CPU only on Linux RDNA1 (no Vulkan EP) | n/a | 1-2 GB | n/a | n/a | Apache-2.0 | Apache fallback if pyannote gating annoys |
| WhisperX | Yes (STT + word-level + diar) | CT2 + PyTorch | n/a | No on RDNA1 | STT + diar | Inherits Whisper | Yes | BSD/MIT | RDNA3-only via BoredYama/whisperX-AMD-ROCM7.1 fork |

### Skipped (won't run on GPU0 or wrong language)

- **NeMo Parakeet/Canary/Sortformer** -- RDNA1 unsupported; Canary v2 covers
  EN + 24 EU langs, **no Hebrew**.
- **MOSS-Audio (Apr 13, 2026)** -- audio-LLM (QA/reasoning), not a
  diarized-transcript producer; ASR benchmarks ZH/EN only.
- **MOSS-Transcribe-Diarize (Jan 2026)** -- single-pass SATS, beats
  ElevenLabs / GPT-4o on Chinese benchmarks, but trained on
  ZH/EN/KO/JA/Cantonese -- Hebrew not in set.
- **Distil-Whisper, Phi-4-multimodal, Granite Speech, Canary-Qwen,
  ReazonSpeech** -- language coverage gap (no Hebrew audio) or VRAM.
- **Voxtral Mini/Small** -- RDNA1 unsupported; Voxtral-Small needs >24 GB.

## RDNA1 reality check

ROCm PyTorch on gfx1010 has been broken since PyTorch >= 2.0 (issue
[pytorch#106728](https://github.com/pytorch/pytorch/issues/106728)).
Community wheels exist (Efenstor, TheTrustedComputer) but require
compile-from-source and are brittle. The `HSA_OVERRIDE_GFX_VERSION=10.3.0`
masquerade works for some kernels and fails silently for others. **Don't
build a service around it.**

That leaves two working backends on the RX 5700 XT:

1. **whisper.cpp Vulkan** -- mature, well-tested. The current open issue
   [whisper.cpp#3611](https://github.com/ggml-org/whisper.cpp/issues/3611)
   reports an RDNA1 SDMA `vkCmdFillBuffer` crash; the documented fix
   (route fillBuffer to compute queue) is not yet merged. May need to
   pin a patched build.
2. **CPU** -- Ryzen 9 5950X handles ivrit-ct2 turbo at int8 in ~1-2x RT,
   which is fine for an offline ingest sidecar.

## The v1 tradeoff

On the 5700 XT we can have **fast** (Vulkan, vanilla Whisper Hebrew =
mediocre) or **good Hebrew** (CPU, ivrit-ct2), not both.

Three v1 options surfaced in brainstorming, decision pending:

- **A. CPU-only with ivrit-ct2 + pyannote 3.1 (CPU).** Best Hebrew,
  slowest wall-clock, GPU0 idle. Simplest deployment -- no Vulkan/ROCm
  involved.
- **B. whisper.cpp Vulkan large-v3-turbo on GPU0 + pyannote 3.1 (CPU).**
  Faster wall-clock, worse Hebrew. GPU0 earns its keep.
- **C. Swappable STT backend, ship A as default.** Build the converter
  with engine selection; default to ivrit-ct2 because Hebrew is the
  harder problem.

Pipeline shape (also pending):

- **A. Single bundled sidecar** (one HTTP service that does STT + diar,
  returns `{segments: [{start, end, speaker, language, text}, ...]}`).
  Mirrors `docling-serve`.
- **B. Two sidecars** -- STT and diarization independent; Library fuses.
  More moving parts, each piece swappable.
- **C. STT-only v1, diarization v2.** Punts the main feature.

## Future RDNA3 path

If the user later accepts taking GPU1 with a "transcribe job" mode that
unloads the chat model, the strongest stack is **WhisperX (ivrit ct2 +
pyannote 3.1) on GPU1 via the BoredYama whisperX-AMD-ROCM7.1 fork** --
one pipeline, word-level timestamps, integrated diarization. Best
quality available locally.

## Sources

- [whisper.cpp RDNA1 Vulkan crash -- issue #3611](https://github.com/ggml-org/whisper.cpp/issues/3611)
- [whisper.cpp Vulkan backend discussion #2375](https://github.com/ggml-org/whisper.cpp/discussions/2375)
- [whisper.cpp 1.8.3 Vulkan perf -- Phoronix](https://www.phoronix.com/news/Whisper-cpp-1.8.3-12x-Perf)
- [ivrit-ai/whisper-large-v3-turbo-ct2](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)
- [Training Whisper Turbo at ivrit.ai](https://www.ivrit.ai/en/2025/02/13/training-whisper/)
- [ivrit-ai Hebrew transcription leaderboard](https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard)
- [pyannote 4.0 VRAM regression -- issue #1963](https://github.com/pyannote/pyannote-audio/issues/1963)
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [PyTorch RDNA1/gfx1010 segfaults -- issue #106728](https://github.com/pytorch/pytorch/issues/106728)
- [Efenstor/PyTorch-ROCm-gfx1010](https://github.com/Efenstor/PyTorch-ROCm-gfx1010)
- [WhisperX on AMD ROCm 7.2 -- discussion #1364](https://github.com/m-bain/whisperX/discussions/1364)
- [BoredYama/whisperX-AMD-ROCM7.1](https://github.com/BoredYama/whisperX-AMD-ROCM7.1)
- [MOSS-Audio (OpenMOSS, Apr 13 2026)](https://github.com/OpenMOSS/MOSS-Audio)
- [MOSS-Audio coverage -- MarkTechPost](https://www.marktechpost.com/2026/04/27/openmoss-releases-moss-audio-an-open-source-foundation-model-for-speech-sound-music-and-time-aware-audio-reasoning/)
- [MOSS-Transcribe-Diarize collection](https://huggingface.co/collections/OpenMOSS-Team/moss-transcribe-diarize)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- [simple-diarizer](https://pypi.org/project/simple-diarizer/)
- [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
