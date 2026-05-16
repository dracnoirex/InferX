import httpx
import time
import os
import statistics
import asyncio
from typing import List


BASE_URL = "http://localhost:8000"
USERNAME = os.getenv("TEST_USERNAME", "admin")
PASSWORD = os.getenv("TEST_PASSWORD", "changeme")

TEST_TEXTS = [
    "Machine learning is transforming the world",
    "Deep learning enables complex pattern recognition",
    "Natural language processing powers modern AI",
    "Transformer models revolutionized NLP tasks",
    "GPU acceleration enables faster model inference"
]


async def get_token(client: httpx.AsyncClient) -> str:
    """Get authentication token"""
    response = await client.post(
        f"{BASE_URL}/api/auth/token",
        data={"username": USERNAME, "password": PASSWORD}
    )
    return response.json()["access_token"]


async def encode_request(
    client: httpx.AsyncClient,
    token: str,
    texts: List[str]
) -> float:
    """Send single encode request and return latency"""
    start = time.time()
    response = await client.post(
        f"{BASE_URL}/api/v1/encode",
        json={"texts": texts, "normalize": True},
        headers={"Authorization": f"Bearer {token}"}
    )
    latency = time.time() - start

    if response.status_code != 200:
        print(f"❌ Request failed: {response.status_code}")
        return -1

    return latency


async def run_load_test(
    num_requests: int = 20,
    concurrency: int = 5
):
    """Run load test with concurrent requests"""
    print(f"\n🚀 InferX Load Test")
    print(f"   Requests: {num_requests}")
    print(f"   Concurrency: {concurrency}")
    print(f"   Target: {BASE_URL}\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Get token
        token = await get_token(client)
        print("✅ Authentication successful\n")

        latencies = []
        errors = 0

        # Run requests in batches
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            tasks = [
                encode_request(client, token, TEST_TEXTS[:2])
                for _ in range(batch_size)
            ]

            results = await asyncio.gather(*tasks)

            for latency in results:
                if latency == -1:
                    errors += 1
                else:
                    latencies.append(latency)

            print(f"Progress: {min(i + concurrency, num_requests)}/{num_requests}")

        # Print results
        if latencies:
            print(f"\n📊 Results:")
            print(f"   Total requests : {num_requests}")
            print(f"   Successful     : {len(latencies)}")
            print(f"   Failed         : {errors}")
            print(f"   Min latency    : {min(latencies):.3f}s")
            print(f"   Max latency    : {max(latencies):.3f}s")
            print(f"   Avg latency    : {statistics.mean(latencies):.3f}s")
            print(f"   Median latency : {statistics.median(latencies):.3f}s")
            print(f"   Throughput     : {len(latencies)/sum(latencies):.2f} req/s")


if __name__ == "__main__":
    asyncio.run(run_load_test(num_requests=20, concurrency=5))