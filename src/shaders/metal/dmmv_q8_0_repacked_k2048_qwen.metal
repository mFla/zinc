#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Qwen3.6 exact repacked Q8_0 DMMV for K=2048 and M%4==0.
//
// This is the tail-free sibling of dmmv_q8_0_repacked_k2048_quad.metal. It
// follows llama.cpp's fixed row-group Q8 matvec discipline and is only selected
// for production Qwen SSM/full-attn projections whose row counts are multiples
// of four; the generic quad kernel remains available for validation tail cases.
kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 4u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    constexpr ulong group_bytes = 1088ul;
    constexpr ulong row_bytes = 2176ul;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;
    device const uchar* row2 = row0 + 2ul * row_bytes;
    device const uchar* row3 = row0 + 3ul * row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #pragma unroll
    for (uint gi = 0u; gi < 2u; ++gi) {
        device const uchar* g0 = row0 + ulong(gi) * group_bytes;
        device const uchar* g1 = row1 + ulong(gi) * group_bytes;
        device const uchar* g2 = row2 + ulong(gi) * group_bytes;
        device const uchar* g3 = row3 + ulong(gi) * group_bytes;

        const float s0 = float(as_type<half>(*(device const ushort*)(g0 + lane * 2u)));
        const float s1 = float(as_type<half>(*(device const ushort*)(g1 + lane * 2u)));
        const float s2 = float(as_type<half>(*(device const ushort*)(g2 + lane * 2u)));
        const float s3 = float(as_type<half>(*(device const ushort*)(g3 + lane * 2u)));
        const uint x_base = (gi * 32u + lane) << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const uint qo = 64u + vi * 128u + lane * 4u;
            const char4 q0 = as_type<char4>(*(device const int*)(g0 + qo));
            const char4 q1 = as_type<char4>(*(device const int*)(g1 + qo));
            const char4 q2 = as_type<char4>(*(device const int*)(g2 + qo));
            const char4 q3 = as_type<char4>(*(device const int*)(g3 + qo));
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));

            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
            acc2 = fma(s2, dot(float4(q2), x), acc2);
            acc3 = fma(s3, dot(float4(q3), x), acc3);
        }
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    const float sum2 = simd_sum(acc2);
    const float sum3 = simd_sum(acc3);
    if (lane == 0u) {
        output[base_row] = sum0;
        output[base_row + 1u] = sum1;
        output[base_row + 2u] = sum2;
        output[base_row + 3u] = sum3;
    }
}
