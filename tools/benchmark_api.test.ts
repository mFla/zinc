import { expect, test } from "bun:test";

import {
  buildConcurrencyReport,
  defaultScenarios,
  evaluateConcurrencyGate,
  parseArgs,
  percentile,
  summarizeConcurrencyComparison,
  summarizeResult,
} from "./benchmark_api.mjs";

test("percentile interpolates between sorted values", () => {
  expect(percentile([10, 20, 30, 40], 0.5)).toBe(25);
  expect(percentile([10, 20, 30, 40], 0.95)).toBe(38.5);
});

test("defaultScenarios splits chat and raw modes", () => {
  expect(defaultScenarios("chat").every((scenario) => scenario.kind === "chat")).toBe(true);
  expect(defaultScenarios("raw").every((scenario) => scenario.kind === "raw")).toBe(true);
  expect(defaultScenarios("both").some((scenario) => scenario.kind === "chat")).toBe(true);
  expect(defaultScenarios("both").some((scenario) => scenario.kind === "raw")).toBe(true);
  expect(defaultScenarios("concurrency").some((scenario) => scenario.concurrency === 4)).toBe(true);
});

test("parseArgs reads explicit overrides", () => {
  const args = parseArgs([
    "--base",
    "http://example.test/v1",
    "--mode",
    "raw",
    "--output",
    "/tmp/out.json",
    "--timeout-ms",
    "1234",
    "--min-c4-aggregate-scale",
    "2.5",
    "--max-c4-p95-latency-multiplier",
    "2",
  ]);
  expect(args.base).toBe("http://example.test/v1");
  expect(args.mode).toBe("raw");
  expect(args.output).toBe("/tmp/out.json");
  expect(args.timeoutMs).toBe(1234);
  expect(args.minC4AggregateScale).toBe(2.5);
  expect(args.maxC4P95LatencyMultiplier).toBe(2);
});

test("summarizeResult formats non-streaming results", () => {
  const summary = summarizeResult({
    name: "raw_c1_t256",
    kind: "raw",
    prompt_chars: 24,
    max_tokens: 256,
    concurrency: 1,
    requests: 1,
    makespan_s: 15,
    latency_avg_s: 15,
    latency_p50_s: 15,
    latency_p95_s: 15,
    prompt_tokens_avg: 6,
    completion_tokens_avg: 256,
    completion_tps_avg: 16.7,
    aggregate_completion_tps: 16.7,
    aggregate_total_tps: 17.1,
  });
  expect(summary).toContain("raw_c1_t256");
  expect(summary).toContain("agg_completion_tps=16.70");
});

test("summarizeResult formats streaming results", () => {
  const summary = summarizeResult({
    name: "short_stream_c1_t64",
    kind: "chat",
    prompt_chars: 24,
    max_tokens: 64,
    concurrency: 1,
    stream: true,
    requests: 1,
    makespan_s: 5.5,
    latency_avg_s: 5.5,
    latency_p50_s: 5.5,
    latency_p95_s: 5.5,
    ttft_avg_s: 3.4,
    ttft_p50_s: 3.4,
    ttft_p95_s: 3.4,
    chunks_avg: 1,
  });
  expect(summary).toContain("short_stream_c1_t64");
  expect(summary).toContain("ttft_avg=3.40s");
});

test("buildConcurrencyReport compares c4 scenarios against c1 baseline", () => {
  const report = buildConcurrencyReport([
    {
      name: "raw_c1_t256",
      kind: "raw",
      prompt_chars: 24,
      max_tokens: 256,
      concurrency: 1,
      requests: 1,
      latency_p95_s: 10,
      aggregate_completion_tps: 20,
    },
    {
      name: "raw_c4_t256",
      kind: "raw",
      prompt_chars: 24,
      max_tokens: 256,
      concurrency: 4,
      requests: 4,
      latency_p95_s: 38,
      aggregate_completion_tps: 21,
    },
  ]);

  expect(report).toHaveLength(1);
  expect(report[0].comparisons[0].aggregate_completion_tps_scale).toBeCloseTo(1.05);
  expect(report[0].comparisons[0].p95_latency_multiplier).toBeCloseTo(3.8);
  expect(report[0].comparisons[0].serialized_likely).toBe(true);
  expect(summarizeConcurrencyComparison(report[0], report[0].comparisons[0])).toContain("SERIALIZED_LIKELY");
});

test("evaluateConcurrencyGate fails low c4 scaling", () => {
  const gate = evaluateConcurrencyGate([
    {
      baseline_name: "raw_c1_t256",
      comparisons: [
        {
          name: "raw_c4_t256",
          concurrency: 4,
          aggregate_completion_tps_scale: 1.2,
          p95_latency_multiplier: 3.5,
        },
      ],
    },
  ], {
    minC4AggregateScale: 2.5,
    maxC4P95LatencyMultiplier: 2,
  });

  expect(gate.passed).toBe(false);
  expect(gate.failures).toHaveLength(2);
});
