import random
import math
import numpy as np


class GPUSimulator:
    """
    Simple, illustrative latency simulator (NOT a hardware-accurate model).

    Idea:
    - H100: lower service rate -> lower steady-state utilization at same load -> fewer tail blow-ups.
    - Blackwell: higher service rate encourages higher concurrency / packing -> you run closer to the knee
      (high utilization + more co-tenant coupling) -> p99 shoots up even if p50 improves.

    Latency per request:
      TTFT = queue_delay + prefill_compute + contention_penalty + rare_spike
    """

    def __init__(
        self,
        name: str,
        tokens_per_sec: float,
        target_utilization: float,
        base_overhead_ms: float,
        interference_knee: float,
        interference_strength_ms: float,
        spike_prob_at_knee: float,
        spike_ms_range: tuple[float, float],
        batch_coupling_strength_ms: float,
        seed: int = 0,
    ):
        self.name = name
        self.tokens_per_sec = tokens_per_sec
        self.target_utilization = target_utilization
        self.base_overhead_ms = base_overhead_ms
        self.interference_knee = interference_knee
        self.interference_strength_ms = interference_strength_ms
        self.spike_prob_at_knee = spike_prob_at_knee
        self.spike_ms_range = spike_ms_range
        self.batch_coupling_strength_ms = batch_coupling_strength_ms
        self.rng = random.Random(seed)

    def _workload_tokens(self, prompt_len_tokens: int) -> int:
        # TTFT is dominated by prompt prefill + first decode step;
        # approximate "work" ~ prompt tokens (prefill) + small constant
        return prompt_len_tokens + 32

    def _utilization(self, arrival_rate_rps: float, avg_tokens_per_req: float) -> float:
        # Utilization ~ offered load / capacity
        offered_tokens_per_sec = arrival_rate_rps * avg_tokens_per_req
        return offered_tokens_per_sec / max(self.tokens_per_sec, 1e-9)

    def simulate(
        self,
        n_requests: int,
        arrival_rate_rps: float,
        mix_short_prob: float = 0.7,
        short_prompt: int = 512,
        long_prompt: int = 2048,
    ):
        # Precompute avg tokens/req for util estimate
        avg_prompt = mix_short_prob * short_prompt + (1 - mix_short_prob) * long_prompt
        avg_tokens_per_req = self._workload_tokens(int(avg_prompt))

        util = self._utilization(arrival_rate_rps, avg_tokens_per_req)

        # "Packing factor": Blackwell-like stacks tend to push higher active concurrency.
        # We approximate that the operator targets a higher utilization regime by tuning batching.
        # This is the key knob that can improve p50 yet worsen p99.
        effective_util = min(0.999, max(0.01, util * (self.target_utilization / max(util, 1e-6))))

        ttft_ms = []

        for _ in range(n_requests):
            # Workload mix
            prompt_len = short_prompt if (self.rng.random() < mix_short_prob) else long_prompt
            tokens = self._workload_tokens(prompt_len)

            # Compute time (ms)
            compute_ms = 1000.0 * (tokens / self.tokens_per_sec)

            # Queueing delay: M/M/1-ish blow-up as util -> 1 (simple proxy).
            # queue_ms ~ base * util/(1-util)
            queue_ms = 0.0
            queue_ms += 5.0 * (effective_util / max(1e-6, (1.0 - effective_util)))

            # Batch coupling penalty: long requests co-batched with short increase TTFT variance.
            # More coupling when effective utilization is high.
            coupling_ms = self.batch_coupling_strength_ms * (effective_util**2)
            if prompt_len == long_prompt:
                coupling_ms *= 1.2
            else:
                coupling_ms *= 0.8

            # Interference penalty rises sharply past a "knee" (e.g., 0.90).
            if effective_util > self.interference_knee:
                x = (effective_util - self.interference_knee) / max(1e-6, (1.0 - self.interference_knee))
                # Convex growth
                interference_ms = self.interference_strength_ms * (x**2) * 10.0
            else:
                interference_ms = 0.0

            # Rare spike probability increases near/above knee (noisy neighbor / KV eviction cascade proxy)
            spike_ms = 0.0
            if effective_util <= self.interference_knee:
                spike_prob = 0.0
            else:
                x = (effective_util - self.interference_knee) / max(1e-6, (1.0 - self.interference_knee))
                spike_prob = min(0.5, self.spike_prob_at_knee * (1 + 8 * x))  # ramps up fast
            if self.rng.random() < spike_prob:
                spike_ms = self.rng.uniform(*self.spike_ms_range)

            total_ms = self.base_overhead_ms + compute_ms + queue_ms + coupling_ms + interference_ms + spike_ms
            ttft_ms.append(total_ms)

        return {
            "name": self.name,
            "util_est": util,
            "effective_util": effective_util,
            "ttft_ms": ttft_ms,
        }


def summarize(name, ttft_ms, util_est, effective_util):
    p50 = np.percentile(ttft_ms, 50)
    p95 = np.percentile(ttft_ms, 95)
    p99 = np.percentile(ttft_ms, 99)
    mean = float(np.mean(ttft_ms))
    print(f"\n=== {name} ===")
    print(f"Utilization (estimated): {util_est*100:5.1f}%")
    print(f"Utilization (effective/packed): {effective_util*100:5.1f}%")
    print(f"Mean TTFT: {mean:8.1f} ms")
    print(f"p50  TTFT: {p50:8.1f} ms")
    print(f"p95  TTFT: {p95:8.1f} ms")
    print(f"p99  TTFT: {p99:8.1f} ms")


if __name__ == "__main__":
    # Workload + load
    N = 5000
    arrival_rate_rps = 35.0  # increase this to make p99 "shoot up"

    # H100-like: lower tokens/sec, but we assume operator runs it at a bit lower packed util (more headroom)
    h100 = GPUSimulator(
        name="H100 Simulator",
        tokens_per_sec=250_000,          # baseline capacity
        target_utilization=0.80,         # more headroom
        base_overhead_ms=15.0,           # fixed scheduling overhead
        interference_knee=0.90,
        interference_strength_ms=40.0,
        spike_prob_at_knee=0.02,
        spike_ms_range=(80.0, 250.0),
        batch_coupling_strength_ms=12.0,
        seed=42,
    )

    # Blackwell-like: ~4x tokens/sec, but the stack packs harder (90–95% util),
    # so it crosses the knee more often -> tail spikes.
    blackwell = GPUSimulator(
        name="Blackwell Simulator",
        tokens_per_sec=1_000_000,        # ~4x capacity
        target_utilization=0.95,         # packed harder
        base_overhead_ms=12.0,           # slightly lower overhead
        interference_knee=0.90,
        interference_strength_ms=70.0,   # stronger interference in packed regime
        spike_prob_at_knee=0.05,         # higher spike probability near knee
        spike_ms_range=(200.0, 900.0),   # bigger tail events (KV eviction cascades proxy)
        batch_coupling_strength_ms=22.0, # more coupling due to aggressive batching
        seed=43,
    )

    h = h100.simulate(n_requests=N, arrival_rate_rps=arrival_rate_rps)
    b = blackwell.simulate(n_requests=N, arrival_rate_rps=arrival_rate_rps)

    summarize(h["name"], h["ttft_ms"], h["util_est"], h["effective_util"])
    summarize(b["name"], b["ttft_ms"], b["util_est"], b["effective_util"])

    # Quick comparison highlight
    hp99 = np.percentile(h["ttft_ms"], 99)
    bp99 = np.percentile(b["ttft_ms"], 99)
    print("\n--- p99 Comparison ---")
    print(f"H100 p99:      {hp99:8.1f} ms")
    print(f"Blackwell p99: {bp99:8.1f} ms")
    print(f"p99 ratio (B/H): {bp99/max(hp99,1e-9):.2f}x")

    print("\nTip: Increase arrival_rate_rps (e.g., 45, 55) to see Blackwell p99 jump sharply while p50 stays good.")
