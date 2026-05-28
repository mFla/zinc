import { performance } from "node:perf_hooks";

const shortPrompt = "The capital of France is";
const mediumPrompt =
  "Context for load testing only. " +
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. ".repeat(12) +
  "\nNow continue this text naturally: The capital of France is";
const longPrompt =
  "Longer context for API benchmark only. " +
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. ".repeat(64) +
  "\nNow continue this text naturally: The capital of France is";

export function percentile(values, p) {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0];
  const xs = [...values].sort((a, b) => a - b);
  const rank = (xs.length - 1) * p;
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return xs[lo];
  const frac = rank - lo;
  return xs[lo] * (1 - frac) + xs[hi] * frac;
}

export function defaultScenarios(mode) {
  const chat = [
    { name: "short_c1_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 1 },
    { name: "short_c2_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 2 },
    { name: "short_c4_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 4 },
    { name: "medium_c1_t64", kind: "chat", prompt: mediumPrompt, maxTokens: 64, concurrency: 1 },
    { name: "medium_c2_t64", kind: "chat", prompt: mediumPrompt, maxTokens: 64, concurrency: 2 },
    { name: "medium_c4_t64", kind: "chat", prompt: mediumPrompt, maxTokens: 64, concurrency: 4 },
    { name: "long_c1_t64", kind: "chat", prompt: longPrompt, maxTokens: 64, concurrency: 1 },
    { name: "long_c2_t64", kind: "chat", prompt: longPrompt, maxTokens: 64, concurrency: 2 },
    { name: "long_c4_t64", kind: "chat", prompt: longPrompt, maxTokens: 64, concurrency: 4 },
    { name: "short_c1_t256", kind: "chat", prompt: shortPrompt, maxTokens: 256, concurrency: 1 },
    { name: "short_c4_t256", kind: "chat", prompt: shortPrompt, maxTokens: 256, concurrency: 4 },
    { name: "short_stream_c1_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 1, stream: true },
  ];
  const raw = [
    { name: "raw_c1_t256", kind: "raw", prompt: shortPrompt, maxTokens: 256, concurrency: 1 },
    { name: "raw_c4_t256", kind: "raw", prompt: shortPrompt, maxTokens: 256, concurrency: 4 },
  ];
  const concurrency = [
    { name: "short_c1_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 1 },
    { name: "short_c4_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 4 },
    { name: "raw_c1_t256", kind: "raw", prompt: shortPrompt, maxTokens: 256, concurrency: 1 },
    { name: "raw_c4_t256", kind: "raw", prompt: shortPrompt, maxTokens: 256, concurrency: 4 },
  ];
  if (mode === "chat") return chat;
  if (mode === "raw") return raw;
  if (mode === "concurrency") return concurrency;
  return [...chat, ...raw];
}

export function summarizeResult(result) {
  if (result.stream) {
    return `${result.name}: latency_avg=${result.latency_avg_s.toFixed(2)}s p95=${result.latency_p95_s.toFixed(2)}s ttft_avg=${result.ttft_avg_s.toFixed(2)}s chunks_avg=${result.chunks_avg.toFixed(2)}`;
  }
  return `${result.name}: latency_avg=${result.latency_avg_s.toFixed(2)}s p95=${result.latency_p95_s.toFixed(2)}s agg_completion_tps=${result.aggregate_completion_tps.toFixed(2)} per_req_tps_avg=${result.completion_tps_avg.toFixed(2)}`;
}

export function parseArgs(argv) {
  let base = "http://127.0.0.1:9090/v1";
  let mode = "both";
  let output = `/tmp/zinc_api_benchmark_${Date.now()}.json`;
  let timeoutMs = 600_000;
  let concurrencyReport = true;
  let minC4AggregateScale = null;
  let maxC4P95LatencyMultiplier = null;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--base":
        base = argv[++i] ?? base;
        break;
      case "--mode": {
        const value = argv[++i];
        if (value === "chat" || value === "raw" || value === "both" || value === "concurrency") mode = value;
        else throw new Error(`Invalid --mode '${value}'. Expected chat, raw, both, or concurrency.`);
        break;
      }
      case "--output":
        output = argv[++i] ?? output;
        break;
      case "--timeout-ms": {
        const raw = argv[++i] ?? "";
        const value = Number(raw);
        if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --timeout-ms '${raw}'`);
        timeoutMs = value;
        break;
      }
      case "--no-concurrency-report":
        concurrencyReport = false;
        break;
      case "--min-c4-aggregate-scale": {
        const raw = argv[++i] ?? "";
        const value = Number(raw);
        if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --min-c4-aggregate-scale '${raw}'`);
        minC4AggregateScale = value;
        break;
      }
      case "--max-c4-p95-latency-multiplier": {
        const raw = argv[++i] ?? "";
        const value = Number(raw);
        if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --max-c4-p95-latency-multiplier '${raw}'`);
        maxC4P95LatencyMultiplier = value;
        break;
      }
      case "-h":
      case "--help":
        printUsageAndExit();
        break;
      default:
        throw new Error(`Unknown argument '${arg}'`);
    }
  }

  return { base, mode, output, timeoutMs, concurrencyReport, minC4AggregateScale, maxC4P95LatencyMultiplier };
}

function printUsageAndExit() {
  console.log(`Usage: bun tools/benchmark_api.mjs [options]
  --base <url>         Base /v1 URL (default: http://127.0.0.1:9090/v1)
  --mode <mode>        chat | raw | both | concurrency (default: both)
  --output <path>      JSON artifact path (default: /tmp/zinc_api_benchmark_<ts>.json)
  --timeout-ms <ms>    Per-request timeout in milliseconds (default: 600000)
  --no-concurrency-report
                       Skip cN-vs-c1 scaling analysis in the artifact
  --min-c4-aggregate-scale <x>
                       Fail if any comparable c4 non-stream scenario has
                       aggregate completion tok/s below x times c1
  --max-c4-p95-latency-multiplier <x>
                       Fail if any comparable c4 scenario has p95 latency
                       above x times c1
  -h, --help           Show this help`);
  process.exit(0);
}

function makeBarrier(count) {
  let waiting = 0;
  let release = null;
  const gate = new Promise((resolve) => {
    release = resolve;
  });

  return async () => {
    waiting += 1;
    if (waiting === count && release) release();
    await gate;
  };
}

async function postJson(url, payload, timeoutMs) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

async function warmup(base, timeoutMs, kind) {
  if (kind === "chat") {
    const resp = await postJson(`${base}/chat/completions`, {
      model: "q",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 8,
      temperature: 0,
      stream: false,
    }, timeoutMs);
    await resp.text();
    return;
  }
  const resp = await postJson(`${base}/completions`, {
    model: "q",
    prompt: "hi",
    max_tokens: 8,
  }, timeoutMs);
  await resp.text();
}

async function runNonStreamingRequest(scenario, base, timeoutMs, waitForStart) {
  const url = scenario.kind === "chat" ? `${base}/chat/completions` : `${base}/completions`;
  const payload =
    scenario.kind === "chat"
      ? {
          model: "q",
          messages: [{ role: "user", content: scenario.prompt }],
          max_tokens: scenario.maxTokens,
          temperature: 0,
          stream: false,
        }
      : {
          model: "q",
          prompt: scenario.prompt,
          max_tokens: scenario.maxTokens,
        };

  await waitForStart();
  const t0 = performance.now();
  const resp = await postJson(url, payload, timeoutMs);
  const body = await resp.json();
  const t1 = performance.now();
  const promptTokens = body.usage?.prompt_tokens ?? 0;
  const completionTokens = body.usage?.completion_tokens ?? 0;
  return {
    latencyS: (t1 - t0) / 1000,
    promptTokens,
    completionTokens,
    completionTps: completionTokens / Math.max((t1 - t0) / 1000, 1e-9),
  };
}

async function runStreamingRequest(scenario, base, timeoutMs, waitForStart) {
  await waitForStart();
  const t0 = performance.now();
  const resp = await postJson(`${base}/chat/completions`, {
    model: "q",
    messages: [{ role: "user", content: scenario.prompt }],
    max_tokens: scenario.maxTokens,
    temperature: 0,
    stream: true,
  }, timeoutMs);
  if (!resp.body) throw new Error(`Streaming response for ${scenario.name} had no body`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let firstTokenS = null;
  let chunks = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      break;
    }
    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
    while (true) {
      const eventEnd = buffer.indexOf("\n\n");
      if (eventEnd === -1) break;
      const event = buffer.slice(0, eventEnd);
      buffer = buffer.slice(eventEnd + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") {
          const t1 = performance.now();
          return {
            latencyS: (t1 - t0) / 1000,
            ttftS: firstTokenS ?? (t1 - t0) / 1000,
            chunks,
          };
        }
        const chunk = JSON.parse(payload);
        const content = chunk.choices?.[0]?.delta?.content;
        if (typeof content === "string" && content.length > 0) {
          if (firstTokenS == null) firstTokenS = (performance.now() - t0) / 1000;
          chunks += 1;
        }
      }
    }
  }
  throw new Error(`Streaming response for ${scenario.name} ended without [DONE]`);
}

async function runScenario(scenario, base, timeoutMs) {
  const waitForStart = makeBarrier(scenario.concurrency);
  const started = performance.now();

  if (scenario.stream) {
    const rows = await Promise.all(
      Array.from({ length: scenario.concurrency }, () =>
        runStreamingRequest(scenario, base, timeoutMs, waitForStart),
      ),
    );
    const ended = performance.now();
    const latencies = rows.map((row) => row.latencyS);
    const ttfts = rows.map((row) => row.ttftS);
    return {
      name: scenario.name,
      kind: scenario.kind,
      prompt_chars: scenario.prompt.length,
      max_tokens: scenario.maxTokens,
      concurrency: scenario.concurrency,
      stream: true,
      requests: rows.length,
      makespan_s: (ended - started) / 1000,
      latency_avg_s: average(latencies),
      latency_p50_s: percentile(latencies, 0.5),
      latency_p95_s: percentile(latencies, 0.95),
      ttft_avg_s: average(ttfts),
      ttft_p50_s: percentile(ttfts, 0.5),
      ttft_p95_s: percentile(ttfts, 0.95),
      chunks_avg: average(rows.map((row) => row.chunks)),
    };
  }

  const rows = await Promise.all(
    Array.from({ length: scenario.concurrency }, () =>
      runNonStreamingRequest(scenario, base, timeoutMs, waitForStart),
    ),
  );
  const ended = performance.now();
  const latencies = rows.map((row) => row.latencyS);
  const promptTokens = rows.map((row) => row.promptTokens);
  const completionTokens = rows.map((row) => row.completionTokens);
  const completionTps = rows.map((row) => row.completionTps);

  return {
    name: scenario.name,
    kind: scenario.kind,
    prompt_chars: scenario.prompt.length,
    max_tokens: scenario.maxTokens,
    concurrency: scenario.concurrency,
    requests: rows.length,
    makespan_s: (ended - started) / 1000,
    latency_avg_s: average(latencies),
    latency_p50_s: percentile(latencies, 0.5),
    latency_p95_s: percentile(latencies, 0.95),
    prompt_tokens_avg: average(promptTokens),
    completion_tokens_avg: average(completionTokens),
    completion_tps_avg: average(completionTps),
    aggregate_completion_tps: sum(completionTokens) / Math.max((ended - started) / 1000, 1e-9),
    aggregate_total_tps: (sum(promptTokens) + sum(completionTokens)) / Math.max((ended - started) / 1000, 1e-9),
  };
}

function average(values) {
  return sum(values) / Math.max(values.length, 1);
}

function sum(values) {
  return values.reduce((acc, value) => acc + value, 0);
}

function scenarioFamilyKey(result) {
  return [
    result.kind,
    result.stream ? "stream" : "nonstream",
    result.prompt_chars,
    result.max_tokens,
  ].join(":");
}

export function buildConcurrencyReport(results) {
  const groups = new Map();
  for (const result of results) {
    const key = scenarioFamilyKey(result);
    const rows = groups.get(key) ?? [];
    rows.push(result);
    groups.set(key, rows);
  }

  const report = [];
  for (const rows of groups.values()) {
    const baseline = rows.find((row) => row.concurrency === 1);
    if (!baseline) continue;

    const comparisons = rows
      .filter((row) => row.concurrency > 1)
      .sort((a, b) => a.concurrency - b.concurrency)
      .map((row) => {
        const latencyP95Multiplier = row.latency_p95_s / Math.max(baseline.latency_p95_s, 1e-9);
        const comparison = {
          name: row.name,
          concurrency: row.concurrency,
          requests: row.requests,
          latency_p95_s: row.latency_p95_s,
          baseline_latency_p95_s: baseline.latency_p95_s,
          p95_latency_multiplier: latencyP95Multiplier,
        };

        if (!row.stream && row.aggregate_completion_tps != null && baseline.aggregate_completion_tps != null) {
          const aggregateScale = row.aggregate_completion_tps / Math.max(baseline.aggregate_completion_tps, 1e-9);
          comparison.aggregate_completion_tps = row.aggregate_completion_tps;
          comparison.baseline_aggregate_completion_tps = baseline.aggregate_completion_tps;
          comparison.aggregate_completion_tps_scale = aggregateScale;
          comparison.serialized_likely =
            row.concurrency >= 4 &&
            aggregateScale < 1.5 &&
            latencyP95Multiplier > 2.5;
        }

        if (row.stream && row.ttft_p95_s != null && baseline.ttft_p95_s != null) {
          comparison.ttft_p95_s = row.ttft_p95_s;
          comparison.baseline_ttft_p95_s = baseline.ttft_p95_s;
          comparison.ttft_p95_multiplier = row.ttft_p95_s / Math.max(baseline.ttft_p95_s, 1e-9);
        }

        return comparison;
      });

    if (comparisons.length === 0) continue;
    report.push({
      kind: baseline.kind,
      stream: Boolean(baseline.stream),
      prompt_chars: baseline.prompt_chars,
      max_tokens: baseline.max_tokens,
      baseline_name: baseline.name,
      baseline_latency_p95_s: baseline.latency_p95_s,
      baseline_aggregate_completion_tps: baseline.aggregate_completion_tps ?? null,
      comparisons,
    });
  }
  return report;
}

export function evaluateConcurrencyGate(report, options = {}) {
  const failures = [];
  for (const group of report) {
    for (const comparison of group.comparisons) {
      if (comparison.concurrency !== 4) continue;
      if (
        options.minC4AggregateScale != null &&
        comparison.aggregate_completion_tps_scale != null &&
        comparison.aggregate_completion_tps_scale < options.minC4AggregateScale
      ) {
        failures.push(
          `${comparison.name}: aggregate scale ${comparison.aggregate_completion_tps_scale.toFixed(2)}x < ${options.minC4AggregateScale.toFixed(2)}x`,
        );
      }
      if (
        options.maxC4P95LatencyMultiplier != null &&
        comparison.p95_latency_multiplier > options.maxC4P95LatencyMultiplier
      ) {
        failures.push(
          `${comparison.name}: p95 latency multiplier ${comparison.p95_latency_multiplier.toFixed(2)}x > ${options.maxC4P95LatencyMultiplier.toFixed(2)}x`,
        );
      }
    }
  }
  return { passed: failures.length === 0, failures };
}

export function summarizeConcurrencyComparison(group, comparison) {
  const parts = [
    `${comparison.name} vs ${group.baseline_name}`,
    `c=${comparison.concurrency}`,
    `p95=${comparison.p95_latency_multiplier.toFixed(2)}x`,
  ];
  if (comparison.aggregate_completion_tps_scale != null) {
    parts.push(`agg=${comparison.aggregate_completion_tps_scale.toFixed(2)}x`);
  }
  if (comparison.ttft_p95_multiplier != null) {
    parts.push(`ttft=${comparison.ttft_p95_multiplier.toFixed(2)}x`);
  }
  if (comparison.serialized_likely) {
    parts.push("SERIALIZED_LIKELY");
  }
  return `concurrency: ${parts.join(" ")}`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const scenarios = defaultScenarios(args.mode);
  const seenKinds = new Set(scenarios.map((scenario) => scenario.kind));
  for (const kind of seenKinds) {
    await warmup(args.base, args.timeoutMs, kind);
  }

  const results = [];
  for (const scenario of scenarios) {
    const result = await runScenario(scenario, args.base, args.timeoutMs);
    results.push(result);
    console.log(summarizeResult(result));
  }

  const concurrencyReport = args.concurrencyReport ? buildConcurrencyReport(results) : [];
  if (args.concurrencyReport) {
    for (const group of concurrencyReport) {
      for (const comparison of group.comparisons) {
        console.log(summarizeConcurrencyComparison(group, comparison));
      }
    }
  }

  const artifact = {
    base: args.base,
    mode: args.mode,
    generated_at_unix: Math.floor(Date.now() / 1000),
    results,
    concurrency_report: concurrencyReport,
  };
  await Bun.write(args.output, `${JSON.stringify(artifact, null, 2)}\n`);
  console.log(`ARTIFACT ${args.output}`);

  const gate = evaluateConcurrencyGate(concurrencyReport, {
    minC4AggregateScale: args.minC4AggregateScale,
    maxC4P95LatencyMultiplier: args.maxC4P95LatencyMultiplier,
  });
  if (!gate.passed) {
    for (const failure of gate.failures) {
      console.error(`CONCURRENCY_GATE_FAIL ${failure}`);
    }
    process.exitCode = 2;
  }
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
