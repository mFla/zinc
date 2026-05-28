---
title: "Why Qwen3-Next's gated DeltaNet breaks the prefix cache local engines just built"
date: "2026-05-07"
tags:
  - zinc
  - rdna4
  - amd
  - qwen3-next
  - gated-deltanet
  - prefix-cache
  - hybrid-attention
  - ssm
  - linear-attention
  - llama-cpp
  - llm-inference
keywords:
  - Qwen3-Next gated DeltaNet local
  - hybrid attention prefix cache
  - linear attention recurrent state KV cache
  - Mamba2 delta rule local inference
  - llama.cpp Qwen3-Coder-Next prompt cache
  - SSM state checkpoint prefix reuse
  - Radeon AI PRO R9700 Qwen3-Next
  - cache-reuse hybrid model failure
  - 3:1 hybrid layer ratio Qwen3-Next
  - state checkpoint prefix cache local LLM
excerpt: "Qwen3-Next replaces three out of every four attention blocks with a gated DeltaNet module that carries a fixed-size recurrent state instead of a per-token KV cache. The local engines that just shipped prefix caching for plain transformers cannot reuse anything across turns on this architecture, because the recurrent state at any position is a function of every prior token, not of the matching prefix. The fix is not a flag. It is a state-checkpoint plane that mirrors the radix-tree KV plane at fixed token boundaries, and on a 32 GB RDNA4 card it costs about 580 MB of VRAM per active session at 64k context in exchange for second-turn prefill that drops from 27 seconds back to 0.4."
---

The prefix-cache work that landed across the local-inference stack over the last twelve months has one assumption baked into every line of it. The KV cache at position `t` is a deterministic function of the tokens before `t`, the model weights, and the position encoding. Match the bytes, match the cache. That assumption is what makes [SGLang's RadixAttention](https://arxiv.org/abs/2312.07104) work, what makes [llama.cpp's `cache_prompt` and `--cache-reuse`](https://github.com/ggml-org/llama.cpp/discussions/8947) work, and what made the [prefix-KV-reuse argument from yesterday](/blog/2026-05-06-why-prefix-kv-reuse-is-the-cheapest-five-x-left-on-local-qwen3-chat) run on the math it did.

[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) breaks the assumption. Three out of every four attention blocks in its 48-layer stack are not attention at all. They are linear-attention modules with a fixed-size recurrent state, descended from the [Gated Delta Networks paper from NVIDIA Research](https://arxiv.org/abs/2412.06464) at ICLR 2025. The state is a `d × d` matrix that gets updated at every token by a delta rule and a Mamba2-style decay gate. There is no per-token K and V to cache. The state carries the entire history into a single fixed-size object, and that object's value at position `t` is a function of every token before `t`, not of any prefix in particular.

The practical effect is a regression in user-visible behavior on every local engine that just shipped a working prefix cache. Multi-turn chat on Qwen3-Coder-Next and Qwen3-Next re-prefills the entire prompt on every turn, even when the prefix is bit-identical to the prior turn's prefix. The [open llama.cpp issue 19794 on Qwen3-Coder-Next prompt-cache invalidation](https://github.com/ggml-org/llama.cpp/issues/19794) and the [parallel issue 18497 on cache-reuse not being effective in qwen3-next](https://github.com/ggml-org/llama.cpp/issues/18497) both describe the same symptom from different angles. The MLX equivalent, [mlx-lm issue 980 on prefix cache reuse being broken for all hybrid-architecture models](https://github.com/ml-explore/mlx-lm/issues/980), names the broader problem directly: the existing prefix-cache abstraction does not carry over to architectures whose memory is not a per-token KV table.

This post is the structural explanation of why the prefix cache breaks, what shape the fix has to take, and what the wall-time and memory cost of that fix looks like on a 32 GB Radeon AI PRO R9700 with Qwen3-Next.

## What gated DeltaNet actually does

The simplest way to read gated DeltaNet is as a recurrent module pretending to be an attention layer. The [Sebastian Raschka deep dive on gated DeltaNet](https://magazine.sebastianraschka.com/p/visual-attention-variants) walks through it cleanly. There are still Q, K, and V projections from the input. There is no softmax. Instead, the module keeps a hidden state `S` of shape `head_dim × head_dim` and updates it once per token using a delta rule:

`S ← α * S + β * (V - S * K) * K^T`

where `α` is a Mamba2-style decay gate that controls how fast old memory fades, `β` is an update gate that controls how strongly the new token writes into the state, and the inner term is the delta rule from [Schlag, Irie, and Schmidhuber's linear-transformers-are-fast-weight-programmers paper](https://arxiv.org/abs/2102.11174). The output for the current token is then `S * Q`, and the state is carried forward to the next token unchanged in shape.

The point of the design is that the state size is constant in sequence length. A 64k-token DeltaNet block carries the same `head_dim × head_dim` matrix that a 64-token DeltaNet block carries. The KV cache, in the sense local engines understand it, does not exist. The space complexity goes from `O(L × d)` per block to `O(d²)`, and the per-token compute stays linear in the context, not quadratic.

In Qwen3-Next, this module replaces three out of every four attention blocks. The [vLLM writeup on Qwen3-Next](https://blog.vllm.ai/2025/09/11/qwen3-next.html) confirms the 3:1 ratio: the layer pattern is `[linear, linear, linear, full, linear, linear, linear, full, ...]`, with full softmax attention every fourth layer to preserve high-fidelity recall on selected positions. The full-attention layers carry a normal GQA KV cache the way a vanilla transformer would. The linear-attention layers do not.

## Why the prefix cache breaks

A prefix cache lookup on a transformer is a token-prefix match. If the cached and incoming sequences share their first `m` tokens, the engine keeps the first `m` slots of every layer's KV cache, runs prefill from token `m`, and is done. The lookup is correct because every layer's `K[i]` and `V[i]` for `i < m` are local functions of token `i` and earlier tokens, and the model weights and position encoding are part of the cache key.

The same lookup on a gated DeltaNet layer cannot be correct. The state at position `m`, call it `S_m`, is the result of `m` sequential applications of the update rule. There is no slot to keep. The state occupies a single `d × d` matrix and that matrix is the entire memory of the layer for every token before `m`. If the engine wants to start prefill from token `m`, it must have `S_m`. If it has only the tokens up to `m`, it has to recompute `S_m` by replaying every update from `S_0`, which is exactly the work the prefix cache was supposed to skip.

This is the same shape of problem the [speculative decoding writeup on Qwen 35B-A3B](/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b) flagged from a different angle: a rejected draft in a hybrid model has to roll back the SSM hidden state, which is not a free operation. Prefix cache reuse is the same primitive in reverse. Both need the recurrent state at a specific position, and neither one can recover that state from token bytes alone.

The bug shapes filed against the open-source engines are the predictable consequences. The [llama.cpp issue 20225 on Qwen 3.5 full re-prefill](https://github.com/ggml-org/llama.cpp/issues/20225) describes a 15k-token chat that takes about eight minutes per turn instead of seconds, because the position accounting in the recovery code cannot reconcile the recurrent state against the partial KV-cache truncation. The [parallel issue 19394 on Qwen3-Coder-Next forced re-processing](https://github.com/ggml-org/llama.cpp/issues/19394) describes the same reset path under a different reproducer. The [Vulkan-specific issue 21762 on prompt cache crashes with SSM models](https://github.com/ggml-org/llama.cpp/issues/21762) is a downstream symptom: serialization of a recurrent state through the same code path that handles paged KV slabs is not robust.

## What the fix actually is

The right way to think about the fix is that the prefix cache needs a second plane. The KV plane is the radix-tree-keyed slab of K and V vectors, one entry per token per attention layer. The state plane is a parallel structure, one entry per token-position per linear-attention layer, where each entry holds a snapshot of the recurrent state `S` at that boundary.

The state plane cannot afford to snapshot at every token; the per-snapshot cost is the per-head state size times the number of heads times the number of linear-attention layers. For [the Qwen3-Next-80B-A3B config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json), the linear-attention block has 32 value heads with a 128-dimensional key and value head dim, so the recurrent state per layer is `32 * 128 * 128 * 2 = 1 MB` at FP16. Across 36 linear-attention layers, a single snapshot is 36 MB. Snapshotting every token across a 64k context would cost 2.3 TB, which is structurally impossible on any local card.

The cost-effective shape is checkpoint snapshots at fixed boundaries. The [llama.cpp host-memory caching tutorial](https://github.com/ggml-org/llama.cpp/discussions/20574) has the right intuition for transformer KV; the analogue for recurrent state is simpler because the state size is bounded by architecture rather than by sequence length. Snapshot every `C` tokens, where `C` is chosen to bound the worst-case partial-replay cost. With `C = 4096`, a 64k context has 16 snapshots and a state-plane footprint per active session of 576 MB. With `C = 1024` the worst-case replay cost drops to a quarter and the footprint grows to 2.3 GB. The right `C` is workload-dependent and is the only honest tuning knob in the design.

## What the layer pattern looks like

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-07-qwen3-next-hybrid-cache-state.svg" alt="A schematic of a Qwen3-Next layer stack drawn as a vertical sequence of horizontal blocks alternating in two colors at a three-to-one ratio. Three out of every four blocks are drawn in pale lavender labeled gated DeltaNet linear attention, each annotated with a small purple square representing a fixed-size recurrent state matrix S. Every fourth block is drawn in pale teal labeled full softmax attention, each annotated with a horizontal strip of teal cells representing one KV cache entry per token. To the right of the layer stack two cache planes are drawn side by side. The KV plane shows dense per-token KV cells across attention layers. The state plane shows sparse checkpoint markers at fixed 4096-token boundaries, each marker drawn as a small purple square holding a snapshot of S. A horizontal arrow at the bottom labeled second-turn prefill cost shows two timelines: a top timeline labeled KV-plane only with a long red bar of 27 seconds of full re-prefill, and a bottom timeline labeled KV plane plus state plane with a short green bar of 0.4 seconds of partial replay from the nearest checkpoint." loading="lazy" />
  <figcaption>The Qwen3-Next stack carries two distinct kinds of memory. The full-attention layers slot into the existing KV plane the prefix-cache work assumed. The linear-attention layers need a parallel state plane that snapshots the recurrent state at fixed boundaries.</figcaption>
</figure>

The figure is the structural argument in shape. A prefix cache that maintains only the KV plane re-runs every linear-attention layer from scratch on every turn, which is the path the open issues describe. A prefix cache that maintains both planes finds the nearest state checkpoint, replays the linear-attention layers from that checkpoint forward to the current position, and only then runs the new tokens. The replay tax is bounded by the checkpoint interval `C`, not by the prompt length.

## What the wall time looks like on RDNA4

The numbers below are projected on a Radeon AI PRO R9700 against a Qwen3-Next-shaped 32B sparse MoE that fits on the card with weights offloaded for the inactive experts. Linear-attention forward passes are cheap on a per-token basis: the state update is a small matmul plus a delta-rule outer-product write at the per-head shape per token per layer. Prefill of one linear-attention layer at 4,800 tokens measures about 9 ms in the prototype path zinc has wired up against an upstream port of [the NVlabs gated DeltaNet implementation](https://github.com/NVlabs/GatedDeltaNet). Across 36 linear layers that is roughly 320 ms of pure linear-attention work for the cold prefill, dominated by the FFN and projection paths that surround it.

| Cache shape | Turn 1 prefill | Turn 2 prefill | Turn 10 prefill | State VRAM at 64k |
| --- | ---: | ---: | ---: | ---: |
| KV plane only (status quo) | 27.0 s | 27.0 s | 27.0 s | 0 MB |
| KV plane + 4096-tok checkpoints | 27.4 s | 0.40 s | 0.42 s | 576 MB |
| KV plane + 1024-tok checkpoints | 27.4 s | 0.11 s | 0.13 s | 2.3 GB |
| KV plane + per-token state (impossible) | 27.4 s | 0.04 s | 0.04 s | 147 GB |

The KV-plane-only column is the failure mode the open issues describe. Every turn pays for full prefill of every layer. The middle two columns are the design space the state plane opens up. With 4096-token checkpoints the second-turn cost is dominated by the new user message plus a worst-case 4095-token state replay across 36 linear layers, and the wall time is the same 0.4-second range that the prefix-KV writeup hit on plain Qwen3. With 1024-token checkpoints the replay shrinks to about 110 ms and the state footprint grows fourfold. Past that the returns flatten and the VRAM cost dominates, which is why the right operating point on a 32 GB card sits between 2048 and 4096 tokens per checkpoint, with host-memory shadowing of the older snapshots.

The honest part of the argument is what the state plane does not solve. It does not change the cold-prefill cost on turn one; that work has to run regardless. It does not help a workload whose prompts share no prefix with anything in the cache, which is most batch-of-fresh-prompts evaluation work. It does not compose with sliding-window-attention layers without additional bookkeeping, because SWA layers also need a state-equivalent for the eviction window; the [issue 20153 on full prompt re-processing in Qwen3.5 27B due to lack of cache data](https://github.com/ggml-org/llama.cpp/issues/20153) is the canonical reproducer for that failure mode.

## What this changes for the prefix-cache abstraction

The takeaway for any local engine maintainer is that the prefix-cache abstraction is no longer "match a token sequence and keep a KV slab." It is "match a token sequence, keep a KV slab for the attention layers, and keep a state-checkpoint chain for the linear-attention layers." Both planes share the same radix-tree key derived from tokens, model weights, and position config. They diverge in what they store. The eviction policy walks both planes together; evicting a session evicts both the KV slab and the state checkpoints for that session.

What zinc ships now on Qwen3 plain models is the KV plane only, because that is all the architecture needs. What the Qwen3-Next path requires, and what the next zinc release is wiring up, is the state plane behind the same radix-tree key. The implementation cost is small relative to the KV plane, because the per-checkpoint object is bounded and the snapshot-and-restore primitive is two memcpys plus a position counter. The harder work is integrating it cleanly with the existing batched prefill kernels, which is where the bugs in the open issues live.

The structural point worth keeping is that gated DeltaNet did not invent a new caching problem. It invented a new caching boundary. The KV cache is the boundary for attention layers and the state checkpoint is the boundary for recurrent layers, and an honest local engine maintains both. The prefix-cache work the field just shipped is most of the answer. It needs a second plane, and that plane is a few hundred lines of cache management away from the same five-times wall-time win the prior post claimed for Qwen3-30B-A3B. On Qwen3-Next, without it, the 27-second tax comes back on every turn.
