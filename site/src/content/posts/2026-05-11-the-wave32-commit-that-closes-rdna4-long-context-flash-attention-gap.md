---
title: "The wave32 commit that closes RDNA4's long-context flash attention gap"
date: "2026-05-11"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - flash-attention
  - wave32
  - wavefront
  - subgroup
  - llama-cpp
  - llm-inference
  - decode
  - prefill
  - qwen3
keywords:
  - RDNA4 wave32 flash attention
  - Vulkan subgroup size RDNA
  - llama.cpp PR 19625 scalar flash attention
  - 0cc4m wave32 AMD RDNA
  - gfx1201 wavefront size flash attention
  - Radeon AI PRO R9700 long context decode
  - scalar FA Br row split RDNA
  - dynamic VGPR wave32 RDNA4
  - flash attention pp512 d16384 AMD
  - Vulkan flash attention long context regression
  - subgroup shuffle wave32 vs wave64
  - llama.cpp Vulkan FA refactor
excerpt: "Scalar flash attention in llama.cpp's Vulkan backend was running wave64 by default on AMD GPUs through the end of 2025, which is the wrong wavefront width for any RDNA part and especially wrong for RDNA4. The wave32 commit from PR 19625 is what finally restores long-context decode and prefill on the Radeon AI PRO R9700, with measured pp512@d16384 going from 84 to 247 tok/s on a sibling AMD card and an end-to-end 56 percent throughput gap closing on consumer RDNA hardware. The mechanism is small and architecturally clean: RDNA's native SIMD is 32 lanes wide, RDNA4's dynamic VGPRs only exist in wave32, and a scalar FA tile with row_split=1 has no work for the upper half of a wave64. The post walks through why the default was wrong, what the fix touches, and what the next two RDNA4 occupancy unlocks look like once wave32 is the floor."
---

The scalar flash attention path in llama.cpp's Vulkan backend defaults to wave64 on AMD. It has done so since the path was introduced, which made sense in 2023 when GCN was still the relevant AMD architecture and wave64 was the only width the silicon ran natively. It stopped making sense the moment RDNA shipped, and it has been quietly costing the engine 50 to 200 percent of its long-context throughput on RDNA-class hardware ever since. [PR 19625 in llama.cpp](https://github.com/ggml-org/llama.cpp/pull/19625) is a 32-commit Vulkan FA refactor by `0cc4m`, and one of those commits, named `Use wave32 on AMD RDNA for scalar FA`, is the entire reason the rest of the refactor lands a 193 percent prefill uplift at 16k context on a Radeon Pro VII and a 56 percent end-to-end throughput jump on gfx1151 Strix Halo against the same model file.

The number that matters is on the Radeon AI PRO R9700, gfx1201, which is the card zinc targets. RDNA4 inherits the same 32-lane SIMD design as every RDNA generation since RDNA1, and it adds two things that make wave32 the correct default and wave64 the wrong one. The 128 AI accelerators sit on top of a SIMD that runs 32 fp32 FMA lanes per cycle. The [dynamic VGPR feature that RDNA4 introduced for compute is only available in wave32 mode](https://github.com/azhirnov/cpu-gpu-arch/blob/main/gpu/AMD-RDNA4.md), per the gfx1201 ISA notes. Picking wave64 on RDNA4 for a flash attention tile with `Br=1` is leaving the SIMD half-utilized per cycle, leaving dynamic registers off the table, and paying a second cycle of latency to fold the 64-lane wavefront onto a 32-lane SIMD. The fix is to flip the subgroup-size hint on the pipeline.

This post walks the structural reason wave64 is the wrong default, the specific mechanism PR 19625 corrects on RDNA4, and the two adjacent occupancy unlocks that wave32 makes possible on gfx1201.

## Why scalar FA defaulted to wave64 in the first place

The scalar flash attention shader in llama.cpp is the fallback path the Vulkan backend uses when neither `VK_KHR_cooperative_matrix` (coopmat1) nor `VK_NV_cooperative_matrix2` (coopmat2) is available, or when those paths regress on a specific configuration. The shader runs the QK^T product, the softmax, and the PV accumulation across tiles of size `Bc` along the K dimension and `Br` along the query dimension. For pure decode the query batch is one token, so `Br=1`. For prefill at long context the verifier issues many queries at once and `Br` can be 4, 8, or 16 depending on the head dimension.

A wave64 dispatch made sense as the default on AMD because GCN ran wave64 natively and because [the AMD GPUOpen RDNA performance guide still recommends a workgroup size that is a multiple of 64](https://gpuopen.com/learn/rdna-performance-guide/) for portability across GCN and RDNA. That recommendation is correct as a portability floor. It is also misleading when applied to compute kernels that target RDNA only and that use cross-lane reductions, because RDNA's native wavefront is 32 lanes and a wave64 dispatch on RDNA is folded onto two cycles of a single 32-lane SIMD. Folding doubles the per-instruction latency in exchange for cross-lane shuffles that span 64 lanes for free. Whether that is a good trade depends on what the shader does between shuffles.

For scalar flash attention the answer is: it is a bad trade. The KQ dot product spans the head dimension, not the row dimension, and the row dimension is the one that sets the natural width of the wave. With `Br=1` the KQ dot product for one query position needs at most 32 useful lanes for any head dimension up to 1024 once vectorized loads are accounted for. The upper 32 lanes of a wave64 dispatch sit idle, which is one half of the throughput gone before the kernel does any cross-lane work. The masked softmax reduction across `Bc` columns needs a shuffle, but a 32-lane reduction is one tree level shorter and one barrier weaker than a 64-lane reduction, and the cost of the extra level is exactly what the wave64 fold was supposed to amortize.

The result on RDNA is what `0cc4m`'s benchmark grid shows on a Radeon Pro VII: pp512 at 16k context goes from 84 to 247 tok/s after the refactor lands, and tg128 at 16k context goes from 41 to 58 tok/s on the same card with the same model file, both on the Vulkan FA path. The same shape of uplift shows up on AMD 8060S (gfx1151, the Strix Halo iGPU) under the "without coopmat" configuration, where the only path available is scalar FA: pp512 at d8192 goes from 67 to 190 tok/s on Llama 8B Q4_0 and tg128 at d8192 goes from 31.5 to 35.3 tok/s. The end-to-end consequence is the one [open ollama issue 15601 measured directly](https://github.com/ollama/ollama/issues/15601): standalone llama.cpp built after PR 19625 lands runs 52 tok/s on Gemma4-26B-A4B on the same hardware where the December 2025 vendored llama.cpp ran 34 tok/s, a 56 percent gap that closes the moment wave32 becomes the FA subgroup size.

## What changes structurally on RDNA4

Three pieces of the gfx1201 SIMD make wave32 the structurally right answer for scalar FA on this card, and PR 19625's commit captures all three by toggling the subgroup-size hint on the pipeline.

First, the [gfx1201 WGP packs two CUs of two SIMD cores each, with 32 fp32 FMA lanes per SIMD and a maximum of 16 wavefronts per SIMD](https://github.com/azhirnov/cpu-gpu-arch/blob/main/gpu/AMD-RDNA4.md). A wave64 dispatch occupies the equivalent of two wavefront slots and runs across two cycles. A wave32 dispatch occupies one slot and runs in one cycle. For a kernel that is memory bound on KV reads at long context, the extra slot is what hides DRAM latency. RDNA4 also broke the false cross-wave memory dependencies that [Chester Lam's chipsandcheese teardown](https://chipsandcheese.com/p/rdna-4s-out-of-order-memory-accesses) confirmed RDNA3 still carried, splitting `vmcnt` into per-class counters so a wave can interleave global memory and shared-memory requests without serializing against other waves on the same WGP. That feature pays off best when there are more in-flight wavefronts to interleave. Wave32 doubles the maximum count of in-flight wavefronts per SIMD at the cost of halving the cross-lane reduction width, and at `Br=1` the cross-lane width is not on the critical path.

Second, RDNA4 added dynamic VGPRs for compute shaders, and the ISA documentation is explicit that the feature is wave32-only. The scalar FA shader pre-PR-19625 caches the Q tile in registers when `HSK_per_thread > 16` and stages K and V loads through shared memory on Nvidia, and the same caching is what the refactor turns on for RDNA after the wave32 toggle lands. Without dynamic VGPRs the scalar FA shader had to allocate the worst-case register footprint statically, which forced down occupancy on configurations that did not need it. With dynamic VGPRs in wave32 the shader gets to allocate registers per-wavefront on demand, which is exactly the right shape for a flash attention tile where the register pressure varies with the head dimension.

Third, the LDS bank layout on gfx1201 is 64 banks of 4 bytes each. A wave64 dispatch reads 64 lanes worth of LDS per cycle and is more likely to hit bank conflicts on small `Bc` tiles where multiple lanes index the same bank. A wave32 dispatch reads 32 lanes worth of LDS per cycle and the bank-conflict frequency drops accordingly. The mask buffer the refactor adds with `(Br + 1)` stride padding is specifically tuned to avoid the wave32 bank-conflict pattern; the same padding under wave64 would have to be wider, and the staging buffer would chew through more LDS budget.

The visual below sketches the SIMD-level mechanism and the measured long-context impact on a sibling AMD card running the same Vulkan FA path.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-11-rdna4-wave32-vs-wave64-fa.svg" alt="A two-panel technical diagram. The left panel is a SIMD-lane occupancy schematic for a single gfx1201 32-lane SIMD core under scalar flash attention with Br equal to one. The top half labeled wave64 default shows a 32-lane row drawn as 32 horizontal cells across two stacked cycles, with the first 16 cells of each cycle filled green to indicate useful KQ dot-product work and the remaining 16 cells per cycle filled gray to indicate idle lanes, annotated with the caption 50 percent SIMD utilization and a sidebar listing static VGPR allocation, 64-lane cross-lane shuffle latency, and two cycles per dispatch. The bottom half labeled wave32 RDNA4 fix shows two stacked 32-lane wavefronts each running in a single cycle, with 28 to 30 of the 32 cells filled green and a small tail of 2 to 4 cells filled light yellow representing partial KQ tail work, annotated with the caption greater than 90 percent SIMD utilization and a sidebar listing dynamic VGPR allocation, 32-lane shuffle in one cycle, and double the in-flight wavefronts per SIMD. The right panel is a grouped bar chart titled measured Vulkan scalar FA throughput on AMD Radeon Pro VII, with three pairs of bars at three context depths. At pp512 d0 the before bar is 800 tok per second and the after bar is 828 tok per second, marked plus 3.4 percent. At pp512 d8192 the before bar is 174 tok per second and the after bar is 389 tok per second, marked plus 122.9 percent. At pp512 d16384 the before bar is 84 tok per second and the after bar is 247 tok per second, marked plus 193.0 percent. A small inset chart below the bar chart shows tg128 at d16384 going from 41.5 to 57.7 tok per second, marked plus 39.2 percent. A footer caption credits llama.cpp PR 19625 with the commit titled use wave32 on AMD RDNA for scalar FA." loading="lazy" />
  <figcaption>Wave64 was leaving half of every RDNA4 SIMD cycle on the floor for scalar flash attention with Br=1. The wave32 toggle is what lets the rest of the FA refactor land its measured 193 percent uplift at long context.</figcaption>
</figure>

The reader should notice two things in the chart. The pp512@d0 column moves three percent because at zero context the FA tile dimensions and the KV cache are both small, and the dispatch is bandwidth bound on the weight matrices rather than on the attention reduction. The pp512@d16384 column moves nearly three times because the FA dispatch dominates at that context length, and the wave64-induced underutilization compounds with the larger `Bc` reductions over a 16k-token K matrix. The exact same effect drives the 56 percent end-to-end gap on Strix Halo decode in the ollama issue: at long context, flash attention is the work, and FA running at half-occupancy is what the rest of the engine is waiting on.

## What the refactor pairs with the wave32 toggle

The wave32 commit on its own would close most of the long-context gap, but PR 19625 ships 31 other commits that depend on wave32 being the floor. The three that matter for RDNA4 are: row splitting within workgroups, dynamic Br selection, and Q-register caching. Row splitting groups multiple `Br` rows into one workgroup with explicit cross-row synchronization through LDS, which is only profitable when the wavefront size is small enough to interleave several rows per workgroup without exhausting LDS; wave32 keeps the LDS footprint per workgroup low. Dynamic `Br` picks between `Br=1`, `Br=8`, and `Br=16` per dispatch based on head size and architecture, and the AMD-specific path picks `Br=1` more often than the Nvidia path because the RDNA scalar FA does better at narrower tiles. Q-register caching writes the Q tile to VGPRs once and reuses it across the `Bc` columns of the K matrix, which depends on dynamic VGPR allocation to avoid the worst-case occupancy hit.

The complementary commit `vulkan: use graphics queue on AMD` ([PR 20551](https://github.com/ollama/ollama/issues/15601), referenced in the same ollama issue) is the second half of the ollama 56 percent measurement, but its mechanism is independent: it routes Vulkan compute dispatches through the graphics queue on AMD to dodge an AMD driver bug that adds a memory-fence stall on the dedicated compute queue. The two PRs are independent of each other but compounding at long context, because the compute-queue stall and the wave64 underutilization both punish the same workload.

## What this opens up on gfx1201 next

Three RDNA4 occupancy levers are sitting unused once wave32 is the FA floor, and each maps to a specific kernel that has not been tuned for the architecture yet.

The first is the [`VK_AMD_shader_explicit_vertex_parameter`-adjacent](https://gpuopen.com/learn/rdna-performance-guide/) family of subgroup-size hints. The Vulkan spec exposes `subgroupSize` as a range of 32 to 64 on gfx1201, and the refactor pins it to 32 only for scalar FA. The dmmv kernel from [the April 22 RDNA4 32-column DMMV post](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm) and the dequant prelude shaders all currently inherit the default subgroup size, which on RDNA4 is 64. Forcing wave32 across the dmmv path is the obvious next experiment.

The second is FlashAttention via coopmat1. The refactor's `vulkan: allow using fp16 in coopmat1 flash attention shader` commit fixes the FP16 path for coopmat1, which was previously falling back to FP32 on AMD because of a missing capability check. Coopmat1 on RDNA4 maps directly to `v_wmma_f32_16x16x16_f16` and roughly doubles the FP16 matmul throughput by routing through the matrix cores instead of the SIMD packed-math path, landing at the 191 TFLOPS dense FP16 ceiling AMD publishes for the R9700. The scalar FA path closes the long-context gap; the coopmat1 path is what brings the FA dispatch up to the matrix-core ceiling for prefill, which is the next thing to measure.

The third is the KV cache layout on the scalar FA path. The refactor adds vectorized vec4 stores for the output but does not yet rearrange the KV reads to match the wave32 dispatch width. With wave64 the natural KV stride was 256 bytes per wavefront-cycle; with wave32 the natural stride is 128 bytes per wavefront-cycle, and the RDNA4 memory subsystem can issue more in-flight requests when those strides line up with cache-line boundaries. Aligning the K reads to the wave32 dispatch width is one more change that the post-wave32 baseline can absorb cleanly.

## What changed

For the eight months between the scalar FA path landing and PR 19625 going up for review, every RDNA-class card was paying a 50 to 200 percent long-context tax on the Vulkan FA path because the subgroup size hint was set for an architecture that AMD stopped shipping in 2019. The fix is a single commit that flips the hint, and the larger refactor it lives inside captures the rest of the uplift the wave32 default unlocks. On gfx1201 specifically, the wave32 toggle is what reconnects scalar FA to the parts of the RDNA4 SIMD that the architecture actually optimized for: dynamic VGPRs, the doubled out-of-order memory window, and the 32-lane LDS bank layout. The next round of RDNA4 inference unlocks all build on top of this commit, and they are visible the moment wave32 is the floor of the kernel set rather than the special case.
