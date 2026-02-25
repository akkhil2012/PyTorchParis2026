import numpy as np
import random
import matplotlib.pyplot as plt


class InferenceSimulator:
    def __init__(self, capacity_rps, governance=False, max_util=0.85):
        """
        capacity_rps: max service rate (requests per second)
        governance: whether admission control is enabled
        max_util: cap utilization when governance is ON
        """
        self.capacity = capacity_rps
        self.governance = governance
        self.max_util = max_util

    def simulate(self, arrival_rate_rps, n_requests=5000):
        latencies = []

        utilization = arrival_rate_rps / self.capacity

        # Apply admission control if enabled
        if self.governance:
            utilization = min(utilization, self.max_util)

        for _ in range(n_requests):
            compute_ms = random.uniform(40, 60)

            # Queue delay (non-linear blowup)
            queue_ms = 10 * (utilization / max(1e-6, (1 - utilization)))

            # Interference penalty if high utilization
            contention_ms = 0
            if utilization > 0.90:
                contention_ms = random.uniform(100, 400)

            total_latency = compute_ms + queue_ms + contention_ms
            latencies.append(total_latency)

        return latencies


def summarize(name, latencies):
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"\n{name}")
    print(f"p50: {p50:.1f} ms")
    print(f"p95: {p95:.1f} ms")
    print(f"p99: {p99:.1f} ms")
    return p50, p95, p99


def plot_latency_distributions(lat_no_control, lat_with_control):
    # Histogram overlay (shows tail visually)
    plt.figure()
    plt.hist(lat_no_control, bins=60, alpha=0.6, label="No Governance")
    plt.hist(lat_with_control, bins=60, alpha=0.6, label="With Admission Control")
    plt.title("Latency Distribution: Admission Control Compresses the Tail")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


def plot_percentiles(lat_no_control, lat_with_control):
    # Percentile curve plot (shows p50..p99 range)
    percentiles = np.arange(50, 100, 1)
    p_no = [np.percentile(lat_no_control, p) for p in percentiles]
    p_gv = [np.percentile(lat_with_control, p) for p in percentiles]

    plt.figure()
    plt.plot(percentiles, p_no, marker="o", markersize=3, label="No Governance")
    plt.plot(percentiles, p_gv, marker="o", markersize=3, label="With Admission Control")
    plt.title("Percentile Curve: Tail Latency (p99) Under Control With Governance")
    plt.xlabel("Percentile")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.show()


# ---- Scenario ----
capacity = 100        # GPU can handle 100 rps
arrival_rate = 95     # heavy load → 95% utilization
n_requests = 5000

# ❌ No Governance
no_control = InferenceSimulator(capacity, governance=False)
lat_no_control = no_control.simulate(arrival_rate, n_requests=n_requests)

# ✅ With Admission Control
with_control = InferenceSimulator(capacity, governance=True, max_util=0.85)
lat_with_control = with_control.simulate(arrival_rate, n_requests=n_requests)

summarize("WITHOUT Admission Control", lat_no_control)
summarize("WITH Admission Control", lat_with_control)

# ---- Graphs ----
plot_latency_distributions(lat_no_control, lat_with_control)
plot_percentiles(lat_no_control, lat_with_control)
