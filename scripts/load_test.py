"""Small-platform HTTP load test without third-party load-test dependencies.

Examples:
    python scripts/load_test.py --base-url http://127.0.0.1:5020 --scenario health
    python scripts/load_test.py --base-url http://127.0.0.1:5020 --scenario ask --rates 1,2,5

The ask scenarios intentionally use independent conversations so the
per-conversation lock does not serialize the benchmark.  Results are emitted
as JSON for later comparison and a short console summary for humans.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RequestResult:
    status: int | None
    latency_ms: float
    first_byte_ms: float | None
    error: str | None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil((percentile_value / 100) * len(ordered)) - 1)
    return round(ordered[max(0, index)], 2)


def parse_rates(raw: str) -> list[float]:
    rates = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("--rates 必须是正数，例如 1,2,5")
    return rates


class LoadRunner:
    def __init__(self, base_url: str, scenario: str, duration: int, timeout: float, workers: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.scenario = scenario
        self.duration = duration
        self.timeout = timeout
        self.workers = workers
        self._sequence = 0
        self._sequence_lock = threading.Lock()

    def _next_id(self) -> int:
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _request(self, request_number: int) -> RequestResult:
        started = time.perf_counter()
        first_byte: float | None = None
        request_id = uuid.uuid4().hex
        headers = {
            "Accept": "application/json",
            "X-Request-ID": f"load-{request_id}",
            # Each virtual client gets a stable synthetic address so the
            # benchmark can exercise aggregate capacity without one client
            # hitting the application's per-IP limiter.
            "X-Forwarded-For": f"198.51.100.{(request_number % 200) + 1}",
        }
        if self.scenario == "health":
            method = "GET"
            url = f"{self.base_url}/healthz"
            body = None
        else:
            method = "POST"
            stream = self.scenario == "stream"
            url = f"{self.base_url}/ask/stream" if stream else f"{self.base_url}/ask"
            headers["Content-Type"] = "application/json"
            headers["Accept"] = "text/event-stream" if stream else "application/json"
            headers["X-User-ID"] = f"load-user-{request_number % 200}"
            body = json.dumps(
                {
                    "question": "请给我一句通用的饮食建议。",
                    "conversation_id": f"load-{request_number}-{uuid.uuid4().hex[:12]}",
                },
                ensure_ascii=False,
            ).encode("utf-8")

        try:
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.status
                chunks: list[bytes] = []
                while True:
                    chunk = response.read(16 * 1024)
                    if chunk and first_byte is None:
                        first_byte = (time.perf_counter() - started) * 1000
                    if not chunk:
                        break
                    chunks.append(chunk)
                # Force full response consumption for both JSON and SSE. The
                # body itself is intentionally not retained in the report.
                _ = sum(len(chunk) for chunk in chunks)
            return RequestResult(status, (time.perf_counter() - started) * 1000, first_byte, None)
        except urllib.error.HTTPError as exc:
            try:
                exc.read()
            except Exception:
                pass
            return RequestResult(exc.code, (time.perf_counter() - started) * 1000, first_byte, f"HTTP {exc.code}")
        except Exception as exc:  # pragma: no cover - exercised by network faults
            return RequestResult(None, (time.perf_counter() - started) * 1000, first_byte, type(exc).__name__)

    def run_rate(self, rate: float) -> dict[str, object]:
        total = max(1, math.ceil(rate * self.duration))
        max_workers = max(self.workers, min(64, math.ceil(rate * self.timeout)))
        scheduled_at = time.perf_counter()
        futures: list[Future[RequestResult]] = []
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="load") as executor:
            for index in range(total):
                target_time = scheduled_at + (index / rate)
                delay = target_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                futures.append(executor.submit(self._request, self._next_id()))
            results = [future.result() for future in as_completed(futures)]
        elapsed_seconds = max(time.perf_counter() - scheduled_at, 0.001)
        latencies = [item.latency_ms for item in results]
        first_bytes = [item.first_byte_ms for item in results if item.first_byte_ms is not None]
        status_counts: dict[str, int] = {}
        for item in results:
            key = str(item.status) if item.status is not None else "network_error"
            status_counts[key] = status_counts.get(key, 0) + 1
        successful = sum(1 for item in results if item.status is not None and 200 <= item.status < 300)
        return {
            "scenario": self.scenario,
            "offered_rate_qps": rate,
            "duration_seconds": self.duration,
            "requests": total,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "achieved_rate_qps": round(successful / elapsed_seconds, 3),
            "success_count": successful,
            "error_count": total - successful,
            "error_rate": round((total - successful) / total, 4),
            "status_counts": status_counts,
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "max": round(max(latencies), 2) if latencies else None,
                # ``statistics.fmean`` is unavailable on some older Python
                # versions that are still commonly used for local tooling.
                "mean": round(statistics.mean(latencies), 2) if latencies else None,
            },
            "first_byte_ms": {
                "p50": percentile(first_bytes, 50),
                "p95": percentile(first_bytes, 95),
                "p99": percentile(first_bytes, 99),
            },
        }


def _summary(item: dict[str, object]) -> str:
    latency = item["latency_ms"]
    assert isinstance(latency, dict)
    return (
        f"{item['scenario']} offered={item['offered_rate_qps']} QPS | "
        f"success={item['success_count']}/{item['requests']} | "
        f"achieved={item['achieved_rate_qps']} QPS | "
        f"p95={latency['p95']} ms | errors={item['error_rate']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible HTTP load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:5020")
    parser.add_argument("--scenario", choices=("health", "ask", "stream"), default="ask")
    parser.add_argument("--rates", default="1,2,5", help="comma-separated offered QPS")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.duration <= 0 or args.workers <= 0:
        parser.error("--duration 和 --workers 必须为正数")

    runner = LoadRunner(args.base_url, args.scenario, args.duration, args.timeout, args.workers)
    report = {
        "base_url": args.base_url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenario": args.scenario,
        "results": [runner.run_rate(rate) for rate in parse_rates(args.rates)],
    }
    for result in report["results"]:
        print(_summary(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
