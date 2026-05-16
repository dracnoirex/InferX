import httpx
import sys
import os

BASE_URL = "http://localhost:8000"

USERNAME = os.getenv("TEST_USERNAME", "admin")
PASSWORD = os.getenv("TEST_PASSWORD", "changeme")

def check(name: str, condition: bool, detail: str = ""):
    status = "✅" if condition else "❌"
    print(f"{status} {name}", f"— {detail}" if detail else "")
    return condition


def run_healthcheck():
    print("\n🔍 InferX Health Check\n")
    all_passed = True

    with httpx.Client(timeout=30.0) as client:
        # 1. Root endpoint
        try:
            r = client.get(f"{BASE_URL}/")
            passed = check("Root endpoint", r.status_code == 200)
            all_passed = all_passed and passed
        except Exception as e:
            check("Root endpoint", False, str(e))
            all_passed = False

        # 2. Health endpoint
        try:
            r = client.get(f"{BASE_URL}/api/v1/health")
            data = r.json()
            passed = check(
                "Health endpoint",
                r.status_code == 200 and data.get("model_loaded"),
                f"model={data.get('model_name')}, device={data.get('device')}"
            )
            all_passed = all_passed and passed
        except Exception as e:
            check("Health endpoint", False, str(e))
            all_passed = False

        # 3. Authentication
        try:
            r = client.post(
            f"{BASE_URL}/api/auth/token",
            data={"username": USERNAME, "password": PASSWORD}
        )
            token = r.json().get("access_token")
            passed = check("Authentication", token is not None)
            all_passed = all_passed and passed
        except Exception as e:
            check("Authentication", False, str(e))
            all_passed = False
            token = None

        # 4. Inference
        if token:
            try:
                r = client.post(
                    f"{BASE_URL}/api/v1/encode",
                    json={"texts": ["Health check test"], "normalize": True},
                    headers={"Authorization": f"Bearer {token}"}
                )
                data = r.json()
                passed = check(
                    "Inference endpoint",
                    r.status_code == 200,
                    f"shape={data.get('shape')}, time={data.get('processing_time'):.3f}s"
                )
                all_passed = all_passed and passed
            except Exception as e:
                check("Inference endpoint", False, str(e))
                all_passed = False

        # 5. Metrics
        try:
            r = client.get(f"{BASE_URL}/metrics")
            passed = check("Prometheus metrics", r.status_code == 200)
            all_passed = all_passed and passed
        except Exception as e:
            check("Prometheus metrics", False, str(e))
            all_passed = False

    print(f"\n{'✅ All checks passed!' if all_passed else '❌ Some checks failed!'}\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_healthcheck())