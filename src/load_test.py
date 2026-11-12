
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
TOP_K = 10
CONCURRENT_USERS = 20
TOTAL_REQUESTS = 100

vocab_path = Path("models/serving_vocabularies/customer_id_vocabulary.json")
with open(vocab_path, encoding="utf-8") as f:
    customer_ids = json.load(f)

# Use 20 different customer IDs
test_users = customer_ids[:CONCURRENT_USERS]

def send_request(customer_id):
    url = f"{API_URL}/recommend/{customer_id}?top_k={TOP_K}"
    start = time.perf_counter()

    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - start
            return {
                "success": response.status == 200,
                "time": elapsed,
                "count": len(result.get("recommendations", [])),
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "time": time.perf_counter() - start,
            "count": 0,
            "error": str(e)
        }

print(f"Starting load test: {TOTAL_REQUESTS} requests, {CONCURRENT_USERS} concurrent users")

results = []
start_all = time.perf_counter()

with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
    futures = [
        executor.submit(send_request, test_users[i % len(test_users)])
        for i in range(TOTAL_REQUESTS)
    ]

    for future in as_completed(futures):
        results.append(future.result())

total_time = time.perf_counter() - start_all
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]
latencies = sorted(r["time"] for r in successful)

print("\n========== LOAD TEST RESULTS ==========")
print("Total requests:", len(results))
print("Successful:", len(successful))
print("Failed:", len(failed))
print("Total duration:", round(total_time, 2), "seconds")
print("Throughput:", round(len(results) / total_time, 2), "requests/sec")

if latencies:
    print("Average response:", round(sum(latencies) / len(latencies), 3), "seconds")
    print("P95 response:", round(latencies[min(len(latencies)-1, int(len(latencies)*0.95))], 3), "seconds")
    print("Maximum response:", round(max(latencies), 3), "seconds")
    print("Average recommendations:", round(sum(r["count"] for r in successful) / len(successful), 1))

if failed:
    print("\nSample errors:")
    for item in failed[:5]:
        print("-", item["error"])

print("\nLOAD TEST FINISHED")
