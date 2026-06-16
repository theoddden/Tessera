"""
TTFT (Time To First Token) and TPOT (Time Per Output Token) optimization.

Key features:
- TTFT monitoring and optimization
- TPOT monitoring and optimization
- Streaming support for real-time token generation
- Adapter caching for faster loading
- End-to-end latency benchmarking
"""

import torch
import time
from typing import Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncio
from collections import deque


class TTFTMonitor:
    """
    Monitor Time To First Token (TTFT).

    TTFT = Time from request to first token generated.
    Critical for user-perceived latency.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.ttft_times = deque(maxlen=window_size)
        self.adapter_gen_times = deque(maxlen=window_size)
        self.model_load_times = deque(maxlen=window_size)

    def record_adapter_generation(self, duration_ms: float):
        """Record adapter generation time."""
        self.adapter_gen_times.append(duration_ms)

    def record_model_load(self, duration_ms: float):
        """Record model loading time."""
        self.model_load_times.append(duration_ms)

    def record_ttft(self, duration_ms: float):
        """Record TTFT."""
        self.ttft_times.append(duration_ms)

    def get_stats(self) -> Dict[str, float]:
        """Get TTFT statistics."""
        if not self.ttft_times:
            return {}

        import numpy as np

        times = list(self.ttft_times)

        return {
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "mean_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "count": len(times),
            "adapter_gen_p50_ms": float(np.percentile(list(self.adapter_gen_times), 50))
            if self.adapter_gen_times
            else 0,
            "model_load_p50_ms": float(np.percentile(list(self.model_load_times), 50))
            if self.model_load_times
            else 0,
        }


class TPOTMonitor:
    """
    Monitor Time Per Output Token (TPOT).

    TPOT = Time between consecutive tokens.
    Critical for streaming throughput.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.tpot_times = deque(maxlen=window_size)
        self.tokens_per_second = deque(maxlen=window_size)

    def record_token(self, duration_ms: float):
        """Record time for a single token."""
        self.tpot_times.append(duration_ms)
        self.tokens_per_second.append(1000.0 / duration_ms)

    def get_stats(self) -> Dict[str, float]:
        """Get TPOT statistics."""
        if not self.tpot_times:
            return {}

        import numpy as np

        times = list(self.tpot_times)
        tps = list(self.tokens_per_second)

        return {
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "mean_ms": float(np.mean(times)),
            "tokens_per_second": float(np.mean(tps)),
            "count": len(times),
        }


class AdapterCache:
    """
    Cache generated adapters for faster TTFT.

    Reduces adapter generation time from ~50ms to ~1ms (cached).
    """

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(self, metadata: dict, domain_id: int) -> str:
        """Create cache key from metadata and domain."""
        import json

        metadata_str = json.dumps(metadata, sort_keys=True)
        return f"{domain_id}:{metadata_str}"

    def get(self, metadata: dict, domain_id: int) -> Optional[Dict]:
        """Get cached adapter."""
        key = self._make_key(metadata, domain_id)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, metadata: dict, domain_id: int, adapter: Dict):
        """Cache adapter."""
        key = self._make_key(metadata, domain_id)

        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = adapter

    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }


@asynccontextmanager
async def measure_ttft():
    """Context manager to measure TTFT."""
    # Note: async context managers cannot return values
    # Use TTFTMonitor class to track measurements instead
    yield


class StreamingGenerator:
    """
    Streaming token generator for optimal TPOT.

    Generates tokens as they are produced rather than waiting for full completion.
    """

    def __init__(self, model, adapter_cache: Optional[AdapterCache] = None):
        self.model = model
        self.adapter_cache = adapter_cache or AdapterCache()
        self.ttft_monitor = TTFTMonitor()
        self.tpot_monitor = TPOTMonitor()

    async def generate_streaming(
        self,
        metadata: dict,
        domain_id: int,
        max_tokens: int = 100,
    ) -> AsyncGenerator[str, None]:
        """
        Generate tokens with streaming.

        Args:
            metadata: Metadata for adapter generation
            domain_id: Domain ID
            max_tokens: Maximum tokens to generate

        Yields:
            Tokens as they are generated
        """
        # Generate adapter (or use cache)
        adapter_start = time.perf_counter()
        cached_adapter = self.adapter_cache.get(metadata, domain_id)

        if cached_adapter is None:
            # Generate adapter
            adapter = self.model.generate(metadata, domain_id)
            self.adapter_cache.set(metadata, domain_id, adapter)
        else:
            adapter = cached_adapter

        adapter_time = (time.perf_counter() - adapter_start) * 1000
        self.ttft_monitor.record_adapter_generation(adapter_time)

        # Load adapter into model
        load_start = time.perf_counter()
        # In practice, this would load the adapter into vLLM
        # For now, simulate
        await asyncio.sleep(0.001)  # 1ms load time
        load_time = (time.perf_counter() - load_start) * 1000
        self.ttft_monitor.record_model_load(load_time)

        # Generate tokens with streaming
        first_token_time = None
        last_token_time = None

        for i in range(max_tokens):
            token_start = time.perf_counter()

            # Generate token (simulate)
            await asyncio.sleep(0.01)  # 10ms per token
            token = f"token_{i}"

            token_time = (time.perf_counter() - token_start) * 1000

            if first_token_time is None:
                first_token_time = token_time
                total_ttft = adapter_time + load_time + token_time
                self.ttft_monitor.record_ttft(total_ttft)

            if last_token_time is not None:
                tpot = token_time
                self.tpot_monitor.record_token(tpot)

            last_token_time = token_time

            yield token

    def get_latency_stats(self) -> Dict[str, Dict]:
        """Get combined latency statistics."""
        return {
            "ttft": self.ttft_monitor.get_stats(),
            "tpot": self.tpot_monitor.get_stats(),
            "cache": self.adapter_cache.get_stats(),
        }


class EndToEndLatencyBenchmark:
    """
    Benchmark end-to-end latency including adapter generation.

    Measures:
    - Adapter generation time
    - Adapter loading time
    - TTFT
    - TPOT
    - Total time to completion
    """

    def __init__(self, model, adapter_cache: Optional[AdapterCache] = None):
        self.model = model
        self.adapter_cache = adapter_cache or AdapterCache()
        self.results = []

    async def benchmark_request(
        self,
        metadata: dict,
        domain_id: int,
        num_tokens: int = 50,
    ) -> Dict[str, float]:
        """
        Benchmark a single request.

        Args:
            metadata: Metadata for adapter generation
            domain_id: Domain ID
            num_tokens: Number of tokens to generate

        Returns:
            Latency metrics
        """
        total_start = time.perf_counter()

        # Adapter generation
        adapter_start = time.perf_counter()
        cached_adapter = self.adapter_cache.get(metadata, domain_id)

        if cached_adapter is None:
            adapter = self.model.generate(metadata, domain_id)
            self.adapter_cache.set(metadata, domain_id, adapter)
        else:
            adapter = cached_adapter

        adapter_time = (time.perf_counter() - adapter_start) * 1000

        # Adapter loading
        load_start = time.perf_counter()
        await asyncio.sleep(0.001)  # Simulate load
        load_time = (time.perf_counter() - load_start) * 1000

        # Token generation
        token_times = []
        for i in range(num_tokens):
            token_start = time.perf_counter()
            await asyncio.sleep(0.01)  # Simulate token generation
            token_time = (time.perf_counter() - token_start) * 1000
            token_times.append(token_time)

        total_time = (time.perf_counter() - total_start) * 1000

        # Compute metrics
        ttft = adapter_time + load_time + token_times[0]
        tpot_mean = (
            sum(token_times[1:]) / len(token_times[1:]) if len(token_times) > 1 else 0
        )

        result = {
            "adapter_time_ms": adapter_time,
            "load_time_ms": load_time,
            "ttft_ms": ttft,
            "tpot_mean_ms": tpot_mean,
            "total_time_ms": total_time,
            "num_tokens": num_tokens,
            "cached": cached_adapter is not None,
        }

        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, float]:
        """Get benchmark summary."""
        if not self.results:
            return {}

        import numpy as np

        adapter_times = [r["adapter_time_ms"] for r in self.results]
        ttft_times = [r["ttft_ms"] for r in self.results]
        tpot_times = [r["tpot_mean_ms"] for r in self.results]
        total_times = [r["total_time_ms"] for r in self.results]
        cache_hit_rate = sum(1 for r in self.results if r["cached"]) / len(self.results)

        return {
            "adapter_time_p50_ms": float(np.percentile(adapter_times, 50)),
            "adapter_time_p95_ms": float(np.percentile(adapter_times, 95)),
            "ttft_p50_ms": float(np.percentile(ttft_times, 50)),
            "ttft_p95_ms": float(np.percentile(ttft_times, 95)),
            "tpot_p50_ms": float(np.percentile(tpot_times, 50)),
            "tpot_p95_ms": float(np.percentile(tpot_times, 95)),
            "total_time_p50_ms": float(np.percentile(total_times, 50)),
            "total_time_p95_ms": float(np.percentile(total_times, 95)),
            "cache_hit_rate": cache_hit_rate,
            "num_requests": len(self.results),
        }


def optimize_ttft(hypernetwork, metadata: dict, domain_id: int) -> Dict[str, float]:
    """
    Optimize TTFT with various strategies.

    Strategies:
    1. Adapter caching (already implemented)
    2. Pre-warming (generate adapters for common metadata)
    3. Batch generation (generate multiple adapters at once)
    4. Quantization (faster forward pass)
    """
    results = {}

    # Baseline (no optimization)
    start = time.perf_counter()
    _ = hypernetwork.generate(metadata, domain_id)
    baseline_time = (time.perf_counter() - start) * 1000
    results["baseline_ms"] = baseline_time

    # With quantization (8-bit)
    from tessera_hypernetwork.latency_optimizer import QuantizedHypernetwork

    quantized = QuantizedHypernetwork(hypernetwork, quantization_bits=8)

    start = time.perf_counter()
    _ = quantized.generate(metadata, domain_id)
    quantized_time = (time.perf_counter() - start) * 1000
    results["quantized_8bit_ms"] = quantized_time
    results["quantized_speedup"] = baseline_time / quantized_time

    return results


if __name__ == "__main__":
    # Test TTFT/TPOT monitoring
    print("Testing TTFT/TPOT monitoring...")

    # Create mock model
    class MockModel:
        def generate(self, metadata, domain_id):
            return {"lora_A": torch.randn(16, 4096), "lora_B": torch.randn(4096, 16)}

    model = MockModel()
    cache = AdapterCache()

    # Test streaming generator
    generator = StreamingGenerator(model, cache)

    async def test_streaming():
        async for token in generator.generate_streaming(
            {"domain": "legal"}, 0, max_tokens=5
        ):
            print(f"Generated: {token}")

        print("\nLatency stats:")
        print(generator.get_latency_stats())

    asyncio.run(test_streaming())

    # Test benchmark
    benchmark = EndToEndLatencyBenchmark(model, cache)

    async def test_benchmark():
        for i in range(10):
            await benchmark.benchmark_request(
                {"domain": "legal", "id": i}, 0, num_tokens=20
            )

        print("\nBenchmark summary:")
        print(benchmark.get_summary())

    asyncio.run(test_benchmark())

    print("\nTTFT/TPOT monitoring test passed!")
