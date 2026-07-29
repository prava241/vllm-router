# run_experiment.py
#
# Drives a workload (uniform or zipf) against a running controller, collects
# per-request metrics, and reports P50/P95 latency, TTFT, throughput, and
# cache hit rate.
#
# Example:
#   python experiments/run_experiment.py --workload zipf --num-requests 200 \
#       --rate 10 --controller-url http://localhost:8000 --output results/zipf.csv

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.workloads import (
    generate_uniform_prompts,
    generate_zipf_prompts,
    load_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["uniform", "zipf"], required=True)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--rate", type=float, default=5.0, help="requests/sec")
    parser.add_argument("--arrival", choices=["fixed", "poisson"], default="fixed")
    parser.add_argument("--controller-url", default="http://localhost:8000")
    parser.add_argument("--priority", choices=["LOW", "MEDIUM", "HIGH"], default="MEDIUM")
    parser.add_argument("--max-concurrency", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prefixes", type=int, default=10)
    parser.add_argument("--zipf-param", type=float, default=1.2)
    parser.add_argument("--output", default=None, help="CSV path for per-request records")
    return parser.parse_args()


def build_arrival_offsets(n: int, rate: float, arrival: str) -> list[float]:
    offsets = []
    t = 0.0
    for _ in range(n):
        offsets.append(t)
        if arrival == "fixed":
            t += 1.0 / rate
        else:
            t += random.expovariate(rate)
    return offsets


async def run_request(
    client: httpx.AsyncClient,
    controller_url: str,
    session_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    delay: float,
    run_start: float,
    semaphore: asyncio.Semaphore,
    results: list[dict],
):
    target_time = run_start + delay
    now = time.perf_counter()
    if target_time > now:
        await asyncio.sleep(target_time - now)

    async with semaphore:
        send_time = time.perf_counter()
        r = await client.post(
            f"{controller_url}/sessions/{session_id}/request",
            json={"prompt_token_ids": prompt_token_ids, "max_tokens": max_tokens},
        )
        recv_time = time.perf_counter()
        r.raise_for_status()
        body = r.json()
        metrics = body["metrics"]

        results.append(
            {
                "request_id": body["request_id"],
                "client_observed_latency": recv_time - send_time,
                "total_latency": metrics["total_latency"],
                "ttft": metrics["ttft"],
                "tps": metrics["tps"],
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "num_cached_tokens": metrics.get("num_cached_tokens") or 0,
            }
        )


async def run_experiment(args):
    tokenizer = load_tokenizer()

    if args.workload == "uniform":
        prompts = generate_uniform_prompts(args.num_requests, tokenizer)
    else:
        prompts = generate_zipf_prompts(
            args.num_requests,
            tokenizer,
            num_prefixes=args.num_prefixes,
            zipf_param=args.zipf_param,
        )

    offsets = build_arrival_offsets(args.num_requests, args.rate, args.arrival)

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{args.controller_url}/sessions",
            params={"priority": args.priority},
        )
        r.raise_for_status()
        session_id = r.json()

        results: list[dict] = []
        semaphore = asyncio.Semaphore(args.max_concurrency)
        run_start = time.perf_counter()

        try:
            await asyncio.gather(
                *[
                    run_request(
                        client,
                        args.controller_url,
                        session_id,
                        prompts[i],
                        args.max_tokens,
                        offsets[i],
                        run_start,
                        semaphore,
                        results,
                    )
                    for i in range(args.num_requests)
                ]
            )
        finally:
            run_duration = time.perf_counter() - run_start
            await client.delete(f"{args.controller_url}/sessions/{session_id}")

    return results, run_duration


def summarize(results: list[dict], run_duration: float, args) -> dict:
    latencies = [r["total_latency"] for r in results]
    client_latencies = [r["client_observed_latency"] for r in results]
    ttfts = [r["ttft"] for r in results]

    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_cached_tokens = sum(r["num_cached_tokens"] for r in results)

    mean_hit_rate = statistics.mean(
        (r["num_cached_tokens"] / r["prompt_tokens"]) if r["prompt_tokens"] else 0.0
        for r in results
    )
    aggregate_hit_rate = (
        total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0
    )

    def percentiles(data: list[float]) -> tuple[float, float]:
        if len(data) < 2:
            return (data[0], data[0]) if data else (0.0, 0.0)
        q = statistics.quantiles(data, n=100)
        return q[49], q[94]

    p50_latency, p95_latency = percentiles(latencies)
    p50_client_latency, p95_client_latency = percentiles(client_latencies)
    p50_ttft, p95_ttft = percentiles(ttfts)

    return {
        "workload": args.workload,
        "num_requests": len(results),
        "run_duration_sec": run_duration,
        "throughput_tokens_per_sec": total_completion_tokens / run_duration,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "p50_client_observed_latency": p50_client_latency,
        "p95_client_observed_latency": p95_client_latency,
        "p50_ttft": p50_ttft,
        "p95_ttft": p95_ttft,
        "mean_cache_hit_rate": mean_hit_rate,
        "aggregate_cache_hit_rate": aggregate_hit_rate,
        "rate": args.rate,
        "arrival": args.arrival,
    }


def main():
    args = parse_args()
    results, run_duration = asyncio.run(run_experiment(args))
    summary = summarize(results, run_duration, args)

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"wrote {len(results)} records to {output_path}")
        print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
