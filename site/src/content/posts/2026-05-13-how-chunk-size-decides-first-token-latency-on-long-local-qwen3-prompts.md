---
title: "How chunk size decides first-token latency on long local Qwen3 prompts"
date: "2026-05-13"
tags:
  - zinc
  - chunked-prefill
  - llama-cpp
  - vulkan
  - rdna4
  - qwen3
  - qwen3-next
  - ttft
  - first-token-latency
  - ubatch
  - kv-cache
  - local-llm-inference
keywords:
  - chunked prefill local LLM
  - ubatch-size first-token latency
  - llama.cpp n_batch n_ubatch default 512
  - Sarathi-Serve stall-free batching
  - Qwen3-Next ubatch memory pool crash
  - TTFT 16k prompt RDNA4
  - vLLM max_num_batched_tokens
  - prefill chunk size Pareto
  - chunked prefill single user local
  - Radeon AI PRO R9700 prefill chunking
excerpt: "On a long local Qwen3 prompt the first decode token does not appear until prefill finishes. The size of the prefill chunk is the knob that decides how long that wait is, how much activation memory the engine reserves, and whether a second chat slot gets to make progress at the same time. The default is 512 tokens because it is safe, not because it is fast."
---

On a local 16k-token prompt, the wait the user actually feels is prefill. The first decode step cannot start until the model has finished chewing through the prompt, and everything downstream of that, the sampler, the tokenizer round trip, the streaming, sits behind that single fence.

On a Radeon AI PRO R9700 running Qwen3.6-35B-A3B, decode on the zinc Vulkan backend is around 117 tok/s. Prefill at long context lands around 88 tok/s. A 16k-token prompt at 88 tok/s is roughly 180 seconds of staring at a blinking cursor before the first decode token leaves the model, give or take warm-up. Cutting that wait in half has very little to do with sampler tricks or quantization. It comes down to one decision: how big each prefill chunk should be, and what the engine is allowed to do between chunks.

That decision has a name. Chunked prefill.

## What prefill chunking actually controls

Prefill is the part of inference that processes the prompt before generation starts. It is dense, compute-bound, and embarrassingly parallel across tokens. Decode is the opposite shape: one token at a time, KV reads everywhere, memory bound.

Sarathi-Serve, the OSDI 2024 paper that gave this idea its current shape, framed it cleanly. Prefill saturates GPU compute. Decode leaves most of that compute on the floor. If you can chop a prefill into chunks small enough to fit alongside an in-flight decode iteration without disturbing it, you get the compute-bound prefill and the memory-bound decode running on the same SM cycles, and decode latency for the user already streaming tokens does not have to stall while a new request prefills. The paper calls this stall-free batching and reports it as the mechanism behind a 2.6x serving-capacity uplift on Mistral-7B against vLLM on a single A100, with larger uplifts on bigger models. [The OSDI 2024 paper page](https://www.usenix.org/conference/osdi24/presentation/agrawal) and [the arXiv preprint](https://arxiv.org/abs/2403.02310) have the details.

The local-engine version is almost the same idea, but the constraints are different. There is one GPU. There is one user. The point of chunking on a local box is the shape of latency for a single user with a long prompt and a second tab open, rather than the multi-tenant throughput problem the OSDI paper set out to solve.

In llama.cpp the knob is `--ubatch-size`, the physical maximum batch passed to one device-side dispatch. There is a sibling knob, `--batch-size`, the logical maximum a slot can submit per server tick. The relationship is simple: `batch-size >= ubatch-size`, and prompts longer than `ubatch-size` get split into chunks of that size, processed sequentially. [The maintainer answer in discussion 6328](https://github.com/ggml-org/llama.cpp/discussions/6328) is the cleanest one-paragraph version of the distinction. The server default today is 2048 for `batch-size` and 512 for `ubatch-size`, per [the current llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). vLLM's V1 engine enables chunked prefill by default and exposes the same idea through `max_num_batched_tokens`, where a higher value lowers TTFT because more prefill tokens fit in one batch, at the cost of decode latency for whatever is already streaming, as [the vLLM optimization guide](https://docs.vllm.ai/en/stable/configuration/optimization/) lays out under its chunked prefill section.

## The single-user surprise

The first thing that surprised us when we measured this on Qwen3 was how much `ubatch-size` matters even for a workload that has exactly one chat session active.

The reason is arithmetic intensity. A `ubatch=512` dispatch with a head dimension of 128 and a hidden size of 2048 has a matmul shape that fits comfortably inside a single RDNA4 WGP's working set, with room for the matrix cores to stay busy. A `ubatch=64` dispatch on the same shape is below the point where the matrix cores stay fed, and the kernel falls back to a regime where most of the cycles are spent on weight-load latency that the prefetcher cannot fully hide. A `ubatch=4096` dispatch, on the other hand, exceeds the activation footprint the engine reserved for a single ubatch and has to either spill or, in some Qwen3-Next builds, fail outright.

The published llama.cpp discussion thread has one of the cleanest single-user measurements floating around. A user on four Tesla T4 cards, with a 5,800-token prompt, saw the first-token time drop from 36 seconds to 26 seconds when they moved `ubatch-size` from the default 512 to 2048. Inference tok/s after the first token did not change. The TTFT win is real and it is 28 percent on that hardware for that prompt length. The reason is straightforward: bigger ubatches push the prefill matmuls into the regime where the tensor cores stay saturated, so the wall-clock time per token of prefill drops, and the prompt finishes sooner.

That measurement is on Nvidia, not RDNA4, but the shape of the curve is the same on gfx1201. zinc's prefill kernels are tuned around `ubatch=512` today, and the same fold-up effect appears when we walk that up toward 2048: prefill tok/s climbs, the prefill-attention path stops looking memory bound, and the first decode token lands earlier in wall-clock time. The reason zinc has not flipped the default yet is the part that comes next.

## Why the default is 512

The chunk size has a budget on the other side: activation memory. Every ubatch reserves a transient buffer for the activations of one prefill step. That buffer is wide enough to hold `ubatch_size * hidden_size` floats per layer, plus the working sets for any fused kernels that run on top of it, plus the SSM or recurrent state for hybrid architectures like Qwen3-Next.

When the engine picks `ubatch=512`, the per-step activation footprint is small and a 32 GB card can serve a model the size of Qwen3.6-35B-A3B with room left for a 128k KV cache and a margin for the loader. When the engine picks `ubatch=2048`, the activation buffer is four times larger, and on some hybrid models the recurrent state and attention chunking buffers do not fit in the pool the engine reserved up front. The result is a hard crash.

This is not a theoretical risk. [llama.cpp issue 17578](https://github.com/ggml-org/llama.cpp/issues/17578) is filed against exactly this failure mode on Qwen3-Next-80B-A3B-Instruct. The reporter set `-ub 4096` and the engine aborted with:

```
ggml_new_object: not enough space in the context's memory pool
(needed 10711552, available 10711184)
```

The stack trace lands in `llm_build_qwen3next::build_delta_net_chunking`, which is the gated DeltaNet SSM path. The same crash reproduces on AMD ROCm gfx1151, on an RTX 3090 under CUDA, and on Nvidia GB10 Blackwell at `ubatch=2048` once the prompt grows past a few thousand tokens. The root cause is the graph reservation: the engine reserves `max_nodes` based on the ubatch size at construction time, and for Qwen3-Next the SSM chunking path needs more nodes than the default reservation grants for large ubatches.

The fix is being worked, but the structural lesson is the one the issue makes obvious. The 512 default exists because it is safe across the matrix of architectures and hardware llama.cpp targets, not because it is optimal for any one configuration. On a 32 GB local card with a 35B-A3B MoE and a Qwen3-Next-style hybrid cache, the right answer is somewhere between 512 and 2048, and it depends on which subset of the layers the model uses on a given step.

## The Pareto curve nobody draws

The two knobs are TTFT and activation budget. The chunk size moves both. The picture below sketches what the curve looks like on zinc against Qwen3.6-35B-A3B on the R9700, with the published 5,800-token TTFT data point from the llama.cpp discussion overlaid as an Nvidia reference.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-13-chunked-prefill-ttft-pareto.svg" alt="A two-panel technical diagram. The left panel is a Gantt-style wall-clock timeline comparing monolithic prefill against chunked prefill for two concurrent chat slots on a single local GPU. The top track labeled slot A shows a single long blue prefill block spanning 0 to 9 seconds and then a sequence of small green decode ticks beginning at 9 seconds and continuing to the right edge. The bottom track labeled slot B is empty until 9 seconds in the monolithic case, then begins its own long blue prefill from 9 to 16 seconds, then green decode ticks after 16 seconds. A vertical orange dashed line at 16 seconds marks first-decode-token for slot B in the monolithic case. The chunked case below shows the same two slots but slot A's prefill is broken into four blue chunks of roughly equal length with green decode ticks for slot B interleaved between chunks, and slot B's first decode token lands at 11 seconds instead of 16, marked with a green dashed vertical line. The right panel is a curve plot with chunk size on a log x axis from 64 to 4096 tokens and two stacked y axes. The upper curve plotted in blue shows first-token latency in seconds for a 16k input prompt, descending from 240 seconds at chunk size 64, through 195 at 256, 182 at 512, 165 at 1024, and 142 at 2048, then rising sharply back to 200 seconds at 4096 because the engine spills activations to host memory. A red shaded band from 3072 to 4096 is labeled OOM zone Qwen3-Next 80B. The lower curve plotted in orange shows peak activation memory in gigabytes, rising linearly from 0.4 at chunk 64 to 1.8 at 512 to 7.2 at 2048 to 14.4 at 4096. A horizontal dashed line at 13 gigabytes labeled R9700 free pool after Qwen3.6 35B-A3B intersects the orange curve at approximately chunk size 3700. An overlay dot at chunk size 2048, TTFT 142, is labeled real measurement, 4x Tesla T4, prompt 5800 tokens, llama.cpp discussion 6328, the published TTFT 26 seconds. A footer caption notes that the curve is illustrative for the R9700 32GB profile and the OOM zone is documented at llama.cpp issue 17578 for Qwen3-Next at ubatch greater than or equal to 2048." loading="lazy" />
  <figcaption>The chunk size is the knob that moves both TTFT and activation budget. The sweet spot on a 32 GB R9700 sits roughly between 1k and 2k tokens; past that, hybrid models start hitting the memory-pool ceiling that llama.cpp issue 17578 documents.</figcaption>
</figure>

The reader should notice two things. One: the TTFT curve is not monotonic. It descends as chunks get bigger until the engine starts paying for an activation buffer it cannot afford, then it climbs back up because the runtime is forced to spill or, on hybrid models, crash. Two: the activation-memory curve is monotonic and steep. The Pareto frontier is the inflection point where the next 512 tokens of chunk size cost more memory than they save in TTFT, and on the R9700 that frontier moves with the model and the KV cache shape, not with anything the user can tune from one config file.

## The interleave that local engines do not have yet

The other half of chunked prefill is the half local engines have mostly skipped.

In Sarathi-Serve's framing, the win comes from interleaving prefill chunks with decode iterations from a different request, so that the decode-bound user keeps streaming tokens while the prefill-bound user gets their prompt processed. Single-tenant servers do not interleave. They prefill one request, then decode, then prefill the next.

That is wrong for the chat workload zinc actually runs. The user has multiple tabs open. Tab A is streaming a long answer at 117 tok/s. Tab B is being typed and submits a new 8k-token prompt. Today, the decode in tab A pauses while the new prefill runs. The user sees a stall. The cost of that stall is fixed at the size of the prompt divided by the prefill rate. At 88 tok/s, an 8k prompt is roughly 90 seconds of frozen tab A.

A stall-free local scheduler chunks the new prefill at, say, 512 tokens, and runs each chunk as a fused dispatch alongside one decode step from tab A. The decode step costs about 8 ms. The prefill chunk costs about 6 ms at the right shape. The fused dispatch can be scheduled so tab A's decode pace drops from 117 tok/s to perhaps 60 tok/s during the prefill window, which is a noticeable but not catastrophic slowdown, and tab B sees its first token in roughly 11 seconds instead of 16. The whole point of the framing is that the user in tab A is not happy with a 90-second pause and is much happier with a temporary slowdown.

zinc does not do this yet. The Vulkan backend currently runs prefill and decode as separate command-buffer phases, and the scheduler is one-slot-at-a-time. The work to break that apart is one of the reasons we started the runtime experiment described in [the runtime below Vulkan that local LLMs needed](/blog/2026-05-12-the-runtime-below-vulkan-that-local-llms-needed). A direct submission path can pack chunk dispatches and decode dispatches into the same hot queue without the round-trip cost a general API insists on.

## What we are picking on R9700 today

Until that lands, the practical question is what to set `ubatch-size` to. The honest answer for Qwen3.6-35B-A3B on the R9700 is 1024 for prompts shorter than the model's training context, with a fallback to 512 once the KV cache crosses 32k tokens, because at that point the activation budget tightens. For Qwen3-Next-80B-A3B-Instruct the answer is to stay at 512 until the memory-pool fix from issue 17578 lands upstream, because going higher trips the SSM chunking allocator. For dense Qwen3.5-35B with a smaller KV footprint, 2048 is fine and shaves a clear chunk of TTFT off long prompts at the cost of a smaller margin for very long contexts.

That is not the kind of answer a config-file generator can produce. It depends on the model, the cache shape, the GPU memory budget, and the prompt length. A serving engine that is going to be honest about local first-token latency has to make that decision per request, not per config.

## What changed

The default chunk size in local LLM engines is calibrated for safety across every architecture they support, and that calibration is wrong for any one specific configuration. On a 32 GB Radeon AI PRO R9700 running Qwen3.6-35B-A3B today, moving the chunk from 512 to a model-aware value somewhere near 1024 to 2048 is the cheapest way to make a 16k prompt feel faster, and the reason most engines do not do this automatically is that the activation budget is too easy to overspend on hybrid Qwen3-Next-style models. The Sarathi-Serve idea of running prefill chunks alongside someone else's decode is the next thing local engines should be borrowing, and on a single-GPU multi-tab chat workload it is closer to a chat-quality fix than a throughput optimization. Until that scheduling lands, the practical advice is to tune the chunk per model, look at issue 17578 before going past 1024 on Qwen3-Next, and remember that 512 is the safe answer, not the right one.
