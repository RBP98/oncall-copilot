import time
import random
import logging
from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("demo-app")

app = FastAPI(title="demo-app")

REQS = Counter("demo_http_requests_total", "HTTP requests", ["path", "status"])
LAT  = Histogram("demo_http_request_duration_seconds", "Latency seconds", ["path"])

@app.get("/")
def root():
    path = "/"
    with LAT.labels(path).time():
        REQS.labels(path, "200").inc()
        return {"ok": True, "service": "demo-app"}

@app.get("/slow")
def slow(ms: int = 500):
    path = "/slow"
    with LAT.labels(path).time():
        time.sleep(ms / 1000.0)
        REQS.labels(path, "200").inc()
        log.info("slow called: ms=%s", ms)
        return {"ok": True, "slept_ms": ms}

@app.get("/error")
def error():
    path = "/error"
    with LAT.labels(path).time():
        # randomly simulate 500s
        if random.random() < 0.8:
            REQS.labels(path, "500").inc()
            log.error("simulated 500 error in /error")
            return Response(content="simulated error", status_code=500)
        REQS.labels(path, "200").inc()
        return {"ok": True, "note": "lucky this time"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
