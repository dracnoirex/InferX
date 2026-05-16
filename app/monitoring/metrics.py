from prometheus_client import Counter, Histogram, Gauge
import time

# ── Counters ──────────────────────────────────────────
REQUEST_COUNT = Counter(
    "inferx_request_total",
    "Total number of requests",
    ["endpoint", "status"]
)

CACHE_HIT_COUNT = Counter(
    "inferx_cache_hit_total",
    "Total number of cache hits"
)

CACHE_MISS_COUNT = Counter(
    "inferx_cache_miss_total",
    "Total number of cache misses"
)

# ── Histograms ────────────────────────────────────────
REQUEST_LATENCY = Histogram(
    "inferx_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

INFERENCE_LATENCY = Histogram(
    "inferx_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

# ── Gauges ────────────────────────────────────────────
GPU_MEMORY_USED = Gauge(
    "inferx_gpu_memory_used_mb",
    "GPU memory used in MB"
)

GPU_MEMORY_TOTAL = Gauge(
    "inferx_gpu_memory_total_mb",
    "GPU memory total in MB"
)

MODEL_LOADED = Gauge(
    "inferx_model_loaded",
    "Whether model is loaded (1=yes, 0=no)"
)

# ── Helpers ───────────────────────────────────────────
def update_gpu_metrics():
    """Update GPU memory metrics"""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            GPU_MEMORY_USED.set(used)
            GPU_MEMORY_TOTAL.set(total)
    except Exception:
        pass


class track_latency:
    """
    Context manager for tracking latency of code blocks.

    Usage:
        with track_latency(REQUEST_LATENCY, ["predict"]):
            # your code here
    """
    def __init__(self, histogram: Histogram, labels: list):
        self.histogram = histogram
        self.labels = labels
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start
        self.histogram.labels(*self.labels).observe(duration)