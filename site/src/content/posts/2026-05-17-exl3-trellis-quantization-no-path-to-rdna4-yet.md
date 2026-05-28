---
title: "EXL3's 1.6 bpw trellis quantization has no path to RDNA4 yet"
date: "2026-05-17"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - quantization
  - exl3
  - exllamav3
  - qtip
  - trellis-coded-quantization
  - local-llm-inference
  - qwen3
keywords:
  - EXL3 quantization RDNA4
  - QTIP trellis coded quantization
  - ExLlamaV3 Vulkan support
  - bitshift trellis decode kernel
  - Marlin GEMM AMD
  - 1.6 bpw local LLM
  - VK_KHR_cooperative_matrix gfx1201
  - Qwen3 30B EXL3 perplexity
  - Hadamard incoherence processing GPU
  - llama.cpp vs ExLlamaV3 AMD
excerpt: "EXL3 is the cleanest open quantization story of the last twelve months. It is a QTIP variant, it is coherent at 1.6 bits per weight on a 70B, and it runs the Marlin-style memory-bound GEMM that AWQ never quite managed. It also has zero non-CUDA backends. For zinc on a Radeon AI PRO R9700, that gap is the most expensive open problem on the local inference shelf right now."
---

The cleanest open quantization paper of the last twelve months is also the one with the narrowest hardware footprint. EXL3, the format ExLlamaV3 ships, is a streamlined variant of [QTIP from Cornell RelaxML](https://github.com/Cornell-RelaxML/qtip), uses the same trellis-coded weight encoding the QTIP authors introduced at [NeurIPS 2024](https://arxiv.org/abs/2406.11235), and is coherent at 1.6 bits per weight on a 70B Llama. With the output projection at 3 bpw and a 4k cache, that lets a 70B model fit under 16 GB of VRAM. The [ExLlamaV3 README](https://github.com/turboderp-org/exllamav3) puts the result in one line and moves on.

The same README has a section called "What's missing?" The list is short: LoRA support, ROCm support. There is no Vulkan entry because Vulkan is not on the roadmap. The repository is 23 percent CUDA by line count. The [Marlin](https://github.com/IST-DASLab/marlin) GEMM kernel that EXL3 models its decode on requires Ampere (`sm_80`) or Ada and is itself "not yet optimized for Hopper." On a Radeon AI PRO R9700, the only way to read an EXL3 file today is to convert it back to FP16 in host RAM and ship the bytes to a different engine.

That is the most expensive open gap on the local inference shelf right now. A 30B-class Qwen3 at 3.0 bpw with EXL3 retains roughly the perplexity of a Q5 GGUF, costs about a third the disk and a third the VRAM, and would slot directly into the bandwidth envelope an R9700 already runs at 117 tok/s on Q4_K_M. The reason it does not is that the kernel zoo behind QTIP was written for one ISA.

## What EXL3 actually does on decode

A QTIP-style decode is not a normal dequantize-then-matmul. The weight matrix is stored as a sequence of trellis states, each of which is an integer pointer into a procedural codebook. Decoding a row means walking the trellis with a small finite-state machine and emitting one FP16 value per state transition. The codebook is not a lookup table in the usual sense; it is a function of the trellis state and a small set of "compute-based codes" that produce pseudorandom approximate Gaussians in as few as two integer instructions per weight.

The QTIP authors gave that decoder a name. The bitshift trellis. The point of the construction is that the kernel that decodes a 2- to 4-bit weight back to FP16 does not need to load a 16-entry table. It needs to perform a couple of shifts, an XOR, and a multiply-add. The Cornell paper calls this the HYB code at L=16, Q=9, V=2, and the [QTIP reference repo](https://github.com/Cornell-RelaxML/qtip) ships 2, 3, and 4 bit matrix-vector multiplication kernels at exactly those parameters with trellis tile dimensions T<sub>x</sub> = T<sub>y</sub> = 16.

The other half of the decode is the Marlin inheritance. EXL3 picked up Marlin's tile shape because Marlin demonstrated the cleanest path to a memory-bound int4/int3 matmul on Ampere and Ada: 16x16 weight tiles, a thread block per output tile, async copies to overlap weight load with compute, and a per-tile mma instruction that consumes the decoded weights without round-tripping through shared memory. The ExLlamaV3 README is candid that on Ampere this is not yet at peak, but on `sm_89` (RTX 4090) the 4 bpw kernel sits at roughly memory-bound latency.

The reason this works as a unit, and the reason it does not port to anything else, is that the Viterbi-style decoder, the Hadamard transform that turns the activation vector into an incoherent vector, and the tile-level mma instruction are all in the same thread block, running on the same warp lanes, on hardware that has Nvidia's specific cross-lane shuffle behavior and Nvidia's specific tensor core API. Take any one of those three pieces away and the kernel stops being memory-bound.

## Why the perplexity numbers matter

The case for paying the cost of porting the kernel is the perplexity curve. The numbers below come from the Qwen3-30B-A3B comparison [published on Hugging Face](https://huggingface.co/eaddario/Qwen3-30B-A3B-GGUF/discussions/1), which evaluated EXL3 at three bitrates against a BF16 baseline.

| Quantization | Layer bpw | Head bpw | Wiki PPL | VRAM |
| --- | ---: | ---: | ---: | ---: |
| HF BF16 baseline | 16.00 | 16.00 | 8.90 | 56.3 GB |
| EXL3 3.5 bpw | 3.53 | 6.01 | 9.25 | 12.5 GB |
| EXL3 3.0 bpw | 3.03 | 6.01 | 9.18 | 10.8 GB |
| EXL3 2.5 bpw | 2.53 | 6.01 | 10.10 | 9.0 GB |
| GGUF Q4_K_M (bartowski) | 4.86 | 6.56 | 9.04 | 17.3 GB |

Two things are notable in those rows. EXL3 at 3.0 bpw lands within 0.14 perplexity of Q4_K_M at 4.86 bpw, on the same model, in about a third less VRAM. EXL3 at 2.5 bpw degrades but still produces coherent text, which is the lower edge where it is competing with QuIP# and AQLM rather than with k-quant GGUFs. The break between 2.5 and 3.0 bpw is where Qwen3 starts to lose precision in routing tokens through its MoE experts; the unsloth crew has been documenting that boundary for [several model generations now](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks). For an R9700 with 32 GB of VRAM, the implication is straightforward. The largest Qwen3 dense model that fits at FP16 is the 14B. The Qwen3-Next-80B-A3B at EXL3 3.0 bpw lands close to 30 GB and starts to fit.

That is the prize. It is also exactly the prize that an AMD-only or Vulkan-only local engine cannot collect today.

## What the gap looks like as a kernel matrix

The work to bring EXL3 to RDNA4 is not "compile the CUDA file with HIP and ship." HIPify produces a working build of the trivial paths and silently elides the path that matters, because the path that matters lives in PTX that has no ROCm equivalent and in tensor-core intrinsics that the [RDNA4 WMMA instruction set](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) does not source-replace.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-17-exl3-rdna4-kernel-gap.svg" alt="A two-column technical diagram on a soft sage background. The left column titled CUDA path on Ampere and Ada is a vertical pipeline showing six stacked boxes connected by downward arrows. From top to bottom the boxes read: incoherence Hadamard transform, fused Viterbi bitshift trellis decode, Marlin sixteen by sixteen weight tile, tensor core m16 n8 k16 mma instruction, FP16 output accumulator, and decode loop output. Each box is drawn in pale green with a thin slate border. To the right of each box a small green check mark indicates a shipping kernel. The right column titled Vulkan path on RDNA4 gfx1201 mirrors the same six boxes in pale orange and shows the status of each stage with a check mark, a caution triangle, or a red X. The Hadamard box has a check mark and the label compute shader, fits naturally. The Viterbi decode box has a caution triangle and the label needs LDS lookup table or three instruction code rewrite. The sixteen by sixteen weight tile box has a check mark and the label VK_KHR_cooperative_matrix in RADV. The mma instruction box has a caution triangle and the label v_wmma_f16_16x16x16_f16 exists but no FP16 by int4 variant. The FP16 accumulator box has a check mark. The decode loop output box has a red X and the label Marlin async copy pattern requires manual rewrite. A horizontal divider below both columns leads to a small inset chart at the bottom of the figure titled Qwen3-30B-A3B perplexity versus bits per weight. The inset is a scatter plot with bpw on the x axis from one point five to five and perplexity on the y axis from eight point five to ten point five. Three orange diamond points represent EXL3 measurements at 2.5 bpw 10.10, 3.0 bpw 9.18, and 3.5 bpw 9.25. One blue square at 4.86 bpw 9.04 marks GGUF Q4 K M with an imatrix. A dashed horizontal line at perplexity 8.90 is labeled BF16 floor. The footer caption reads Qwen3-30B-A3B perplexity numbers from the eaddario Hugging Face discussion, kernel availability cross-referenced against the ExLlamaV3 README and the RADV cooperative matrix merge." loading="lazy" />
  <figcaption>The CUDA pipeline runs end-to-end on Ampere and Ada. On RDNA4 with the current Vulkan toolchain, four of the six stages are either incomplete or missing. The two that already exist are not enough on their own to recover EXL3's memory-bound decode profile.</figcaption>
</figure>

The reader should notice two things. One, the Hadamard transform and the cooperative-matrix tile already work on Vulkan with RDNA4. [VK_KHR_cooperative_matrix landed in RADV for gfx1201 in mid 2025](https://www.phoronix.com/news/RADV-Lands-RDNA4-Coop-Matrix), and zinc already uses the equivalent path for its FP8 wave32 attention kernel. Two, the Viterbi decoder and the Marlin async-copy GEMM are the parts that need to be written. The Viterbi step is the easier of the two on paper, because the inner loop is shifts and one fused multiply-add per weight, and RDNA4's scalar ALU is more than capable of running that. The harder part is fitting the trellis state into the same wavefront that has to feed `v_wmma_f16_16x16x16_f16`. RDNA4 ships INT4 and INT8 WMMA paths and a 16x32 INT4-by-INT8 variant, but it has no fused FP16-by-int4 instruction; the matmul has to consume an FP16 weight tile that was decoded one step earlier and written into VGPRs, and that hand-off is where the Marlin async-copy trick stops translating cleanly.

## What it would take, concretely

The work breaks into four pieces, all of which are local-engine work, none of which are exotic on RDNA4 considered alone, and together they amount to one of the larger backend efforts the open-source landscape currently has on its plate.

First, a Vulkan compute shader for the bitshift trellis decoder, parameterized at compile time on L, Q, and V via SPIR-V specialization constants. The same specialization-constant trick [zinc used for dmmv variants in April](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants) maps directly. The kernel emits an FP16 weight tile of shape 16x16 into a workgroup-local buffer. Two instructions per weight is the QTIP HYB code's headline number on Nvidia; on RDNA4 the equivalent is closer to three because the SALU pipeline does not fold the multiply into the shift on the same cycle, but the decode is still cheap relative to the matmul.

Second, a Vulkan cooperative-matrix matmul that consumes the decoded tile. RADV exposes the FP16 cooperative-matrix path through `VK_KHR_cooperative_matrix`, and the API is close enough to PTX `mma.sync` that a port keeps its shape. The cost is that the FP16 weight tile has to be in shared memory before the cooperative-matrix load fires, and that read-after-write fence is the place where Marlin's async copy was hiding the weight load latency. Without that latency hiding, the first cut will be limited by LDS bandwidth, not GDDR6 bandwidth.

Third, an incoherence-processing pass that runs the Hadamard transform on the activation vector before the matmul. This is structural code zinc already has for other purposes and is the easiest of the four. The QTIP paper is precise that the Hadamard transform has to match the one used at quantization time, so the tile size and sign convention need to be locked to whatever EXL3 uses on disk.

Fourth, a converter that reads an EXL3 file and writes whatever serialization zinc's runtime expects. EXL3 keeps the original HF tensor layout, so the per-tensor metadata translates one to one. The codebook parameters and the trellis dimensions are stored in the file. The only piece that needs new bytes is a header that names the quantization scheme so the runtime knows which kernel to dispatch.

A direct, complete first cut is not a weekend project. It is probably four to six weeks of focused work on a single backend, comparable to the wave32 commit, and the perplexity prize is roughly the same shape: an unlock that does not change the model but does change which models you can run on a 32 GB card.

## The tradeoff that is easy to forget

The case for porting EXL3 is not that it is faster than llama.cpp Q4_K_M. The Marlin-style GEMM the ExLlamaV3 README references is memory-bound at 4 bpw on a 4090, and Q4_K_M on a tuned Vulkan backend is also memory-bound on an R9700. At equal bpw, the throughput will be similar. What changes is what bpw is feasible. EXL3 at 2.5 bpw is still coherent on a 30B; Q4_K_M at 2.5 bpw is not a number llama.cpp produces. The win is in the lower half of the curve.

The other tradeoff is conversion cost. EXL3 conversion is faster than AQLM by a wide margin; the ExLlamaV3 README cites a couple of minutes for a 7B and a few hours for a 70B on a single RTX 4090. That is fast for an offline step, but it is still slower than llama-quantize, which runs in tens of minutes for a 70B on CPU. For a local user who downloads a quant rather than producing one, this disappears. For an engine that wants to ship a model-aware quantization path, it is a real difference.

The last tradeoff is the licensing one. The QTIP reference repo is GPLv3. ExLlamaV3 is MIT, and the EXL3 file format is documented in the [exl3.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md) writeup that ships with the repo. A clean Vulkan port reading the EXL3 file would inherit MIT for the runtime, but any conversion code that wants to share Hessian-aware quantization logic with the Cornell reference would have to be GPL. Most local engines have settled on a clean split where the converter is separate from the runtime, and that pattern fits EXL3 well.

## Why this slot opens up now

A year ago the case for putting four weeks of backend work into a trellis decoder would have been thin. The QTIP paper had just landed, the EXL3 format did not exist, and there was no equivalent open implementation on the consumer side. The reason it is the right time to fight this fight now is the bottom of the perplexity curve. Local engines have spent the last year picking up two- and three-bit quantization where the SOTA was AQLM and QuIP#, and the practical answer was "use a smaller model in 4 bpw GGUF." With EXL3 in the wild and converging on the 1.6 bpw lower edge of coherence on a 70B, that fallback is no longer obviously the right call.

The wave32 flash attention fix from last week and the LMHead specialization work from [yesterday's post](/blog/2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode) close out the decode-side bottlenecks on dense Qwen3. The next move is on the weight side. EXL3 is the cheapest open answer there. ROCm support is on turboderp's own list. Vulkan support is not on anyone's list yet, and that is the gap zinc is in the best position to take.

## What comes next

The order of work is straightforward. Convert one Qwen3-8B EXL3 file by hand, ship a one-shot Python script that decodes a trellis tile to FP16 on the CPU as a reference. Write the Vulkan compute shader for the bitshift decode and validate against the CPU reference. Wire that shader into a cooperative-matrix matmul against an FP16 activation. Measure the bandwidth utilization at 3 bpw on Qwen3-8B and compare against the engine's Q4_K_M decode rate on the same model. If the ratio comes in above 0.85, the rest of the path is mostly bookkeeping. If it comes in below, the inner loop needs another pass against the WMMA scheduling guidance the [chips and cheese piece on RDNA4 in LLVM](https://chipsandcheese.com/p/examining-amds-rdna-4-changes-in-llvm) lays out.

There is no scenario in which this is a small piece of work. There is also no scenario in which the next twelve months of local inference are interesting on AMD without it. Trellis quantization at near-Marlin throughput is the thing the open ecosystem just got and the thing the AMD half of the open ecosystem has no path to yet. The Vulkan path is the path that takes it there.
