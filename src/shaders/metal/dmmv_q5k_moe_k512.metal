#include <metal_stdlib>
using namespace metal;

// Qwen3.6 exact-shape Q5_K routed MoE down projection for K=512.
// Two output rows share one simdgroup, mirroring llama.cpp's row-pair Q8
// discipline where the input vector is reused while streaming adjacent rows.
// The two Q5_K blocks are unrolled so the hot expert-down path does not pay the
// generic K<=2048 loop overhead.

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

inline float2 get_scale_min_k5(uint j, device const uchar* scales) {
    if (j < 4u) {
        return float2(float(scales[j] & 63u), float(scales[4u + j] & 63u));
    }
    return float2(
        float((scales[4u + j] & 0x0Fu) | ((scales[j - 4u] >> 6u) << 4u)),
        float((scales[4u + j] >> 4u) | ((scales[j] >> 6u) << 4u))
    );
}

inline void accumulate_q5k_block_pair(
    device const uchar* block0,
    device const uchar* block1,
    device const float* input,
    uint col_base,
    uint lane,
    thread float& sum0,
    thread float& sum1
) {
    const float d0 = float(as_type<half>(*(device const ushort*)(block0)));
    const float dmin0 = float(as_type<half>(*(device const ushort*)(block0 + 2u)));
    device const uchar* scales0 = block0 + 4u;
    device const uchar* high_bits0 = block0 + 16u;
    device const uchar* quants0 = block0 + 48u;

    const float d1 = float(as_type<half>(*(device const ushort*)(block1)));
    const float dmin1 = float(as_type<half>(*(device const ushort*)(block1 + 2u)));
    device const uchar* scales1 = block1 + 4u;
    device const uchar* high_bits1 = block1 + 16u;
    device const uchar* quants1 = block1 + 48u;

    const uint qh_val0 = uint(high_bits0[lane]);
    const uint qh_val1 = uint(high_bits1[lane]);

    #pragma unroll
    for (uint g = 0u; g < 4u; g++) {
        const uint sb_lo = g * 2u;
        const uint sb_hi = sb_lo + 1u;
        const float2 sm0_lo = get_scale_min_k5(sb_lo, scales0);
        const float2 sm0_hi = get_scale_min_k5(sb_hi, scales0);
        const float factor0_lo = d0 * sm0_lo.x;
        const float bias0_lo = dmin0 * sm0_lo.y;
        const float factor0_hi = d0 * sm0_hi.x;
        const float bias0_hi = dmin0 * sm0_hi.y;

        const float2 sm1_lo = get_scale_min_k5(sb_lo, scales1);
        const float2 sm1_hi = get_scale_min_k5(sb_hi, scales1);
        const float factor1_lo = d1 * sm1_lo.x;
        const float bias1_lo = dmin1 * sm1_lo.y;
        const float factor1_hi = d1 * sm1_hi.x;
        const float bias1_hi = dmin1 * sm1_hi.y;

        const uint q0_byte = uint(quants0[g * 32u + lane]);
        const uint q1_byte = uint(quants1[g * 32u + lane]);
        const float v0_lo = factor0_lo * float((q0_byte & 0x0Fu) | (((qh_val0 >> sb_lo) & 1u) << 4u)) - bias0_lo;
        const float v0_hi = factor0_hi * float((q0_byte >> 4u) | (((qh_val0 >> sb_hi) & 1u) << 4u)) - bias0_hi;
        const float v1_lo = factor1_lo * float((q1_byte & 0x0Fu) | (((qh_val1 >> sb_lo) & 1u) << 4u)) - bias1_lo;
        const float v1_hi = factor1_hi * float((q1_byte >> 4u) | (((qh_val1 >> sb_hi) & 1u) << 4u)) - bias1_hi;

        const uint col_lo = col_base + g * 64u + lane;
        const uint col_hi = col_lo + 32u;
        const float x_lo = input[col_lo];
        const float x_hi = input[col_hi];
        sum0 += v0_lo * x_lo;
        sum0 += v0_hi * x_hi;
        sum1 += v1_lo * x_lo;
        sum1 += v1_hi * x_hi;
    }
}

#define TG_SIZE 512
#define ROWS_PER_TG ((TG_SIZE / 32) * 2)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    const uint row0 = tg_pos.x * ROWS_PER_TG + simdgroup * 2u;
    if (row0 >= p.M) return;
    const uint row1 = row0 + 1u;
    const bool has_row1 = row1 < p.M;

    device const float* input = X + (p.x_offset / 4u) + expert_slot * p.x_expert_stride;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* row0_ptr = W + expert_base + ulong(row0) * 352ul;
    device const uchar* row1_ptr = has_row1 ? (row0_ptr + 352ul) : row0_ptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    accumulate_q5k_block_pair(row0_ptr, row1_ptr, input, 0u, lane, sum0, sum1);
    accumulate_q5k_block_pair(row0_ptr + 176u, row1_ptr + 176u, input, 256u, lane, sum0, sum1);

    const float total0 = simd_sum(sum0);
    const float total1 = simd_sum(sum1);
    if (lane == 0u) {
        device float* output = Y + (p.y_offset / 4u) + expert_slot * p.M;
        output[row0] = total0;
        if (has_row1) {
            output[row1] = total1;
        }
    }
}
