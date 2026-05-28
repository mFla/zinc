#!/usr/bin/env bash
# Bench llama.cpp Qwen3-8B Q4_K_M on the local Metal backend so we have
# a comparator number for Effort 14 (M1 Max Qwen3-8B decode parity).
#
# Run AFTER the implement_metal.ts loop finishes — both processes contend
# for the GPU and concurrent runs poison each other's measurements.
#
# History (2026-05-13/14): the original version used `llama-cli` with
# `-no-cnv --simple-io`. Despite those flags, `llama-cli` hung 4
# separate times on the warmup invocation (96% CPU, no output, for
# hours), blocking the overnight launcher chain. The hang appears to
# be on stdin-handling regardless of the no-conversation flags.
# `llama-simple` is the minimal non-interactive driver in the same
# build dir; it does not touch the chat/conversation code path and
# ran cleanly on the first try. Switched to llama-simple as the
# benchmark driver. Each run has a 120s per-invocation timeout so a
# future regression in llama-simple cannot block the launcher again.

set -uo pipefail

MODEL=/Users/stepan/Library/Caches/zinc/models/models/qwen3-8b-q4k-m/model.gguf
BIN=$HOME/Workspace/llama.cpp/build-metal/bin/llama-simple
PROMPT="The capital of France is"
N_TOKENS=128
PER_RUN_TIMEOUT=180   # seconds; warmup is ~4s, runs ~4s — 180 is generous.

if [[ ! -x "$BIN" ]]; then
  echo "llama-simple not found at $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL" >&2
  exit 1
fi
if ! command -v gtimeout >/dev/null && ! command -v timeout >/dev/null; then
  echo "WARNING: no timeout(1) available — runs may hang indefinitely" >&2
fi
TIMEOUT_BIN=$(command -v gtimeout || command -v timeout || echo "")

# Stop strays so the bench has a clean GPU.
pkill -f 'zig-out/bin/zinc' 2>/dev/null || true
pkill -f 'llama-cli'         2>/dev/null || true
pkill -f 'llama-simple'      2>/dev/null || true
sleep 1

echo "=== llama.cpp Qwen3-8B baseline (Metal, M1 Max) ==="
echo "Model:  $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens: $N_TOKENS"
echo "Tool:   $BIN"
echo "Per-run timeout: ${PER_RUN_TIMEOUT}s"
echo

run_one() {
  local label=$1; shift
  echo "=== $label ==="
  if [[ -n "$TIMEOUT_BIN" ]]; then
    "$TIMEOUT_BIN" --kill-after=5s "${PER_RUN_TIMEOUT}s" \
      "$BIN" -m "$MODEL" -n "$N_TOKENS" -ngl 99 "$PROMPT" 2>&1 \
      | grep -E 'speed:|eval time|tokens per second|prompt eval' \
      || echo "(no perf line found — timeout or crash)"
  else
    "$BIN" -m "$MODEL" -n "$N_TOKENS" -ngl 99 "$PROMPT" 2>&1 \
      | grep -E 'speed:|eval time|tokens per second|prompt eval' \
      || echo "(no perf line found)"
  fi
  pkill -f 'llama-simple' 2>/dev/null || true
  sleep 1
}

run_one "WARMUP (discarded)"
for i in 1 2 3; do
  run_one "RUN $i"
done

echo
echo "Done. Compare 'eval' tok/s (decode-only) against ZINC decode for the same model."
echo "Reference number captured on 2026-05-14: 33.96 tok/s median decode."
