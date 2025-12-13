import json
import subprocess
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from copilot.providers.llm.factory import create_llm_provider


PROM_URL = "http://localhost:9090"  # via port-forward
QDRANT_URL = "http://localhost:6333"
COLLECTION = "runbooks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="oncall-copilot")

qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)

class ChatIn(BaseModel):
    question: str
    namespace: str = "oncall"
    app_label: str = "demo-app"

def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"ERROR running {cmd}: {r.stderr}"
    return r.stdout

def docs_search(q: str, top_k=3):
    vec = embedder.encode(q).tolist()

    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k,
        with_payload=True,
    )

    hits = res.points
    return [
        {
            "source": h.payload.get("source"),
            "text": h.payload.get("text"),
            "score": h.score,
        }
        for h in hits
    ]


def prom_query(promql: str):
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": promql}, timeout=3)
        r.raise_for_status()
        return r.json()["data"]["result"]
    except Exception as e:
        return {"error": str(e), "query": promql}


@app.post("/chat")
def chat(inp: ChatIn):
    # 1) Pull docs
    docs = docs_search(inp.question)

    # 2) K8s state
    pods_json = sh(["kubectl", "get", "pods", "-n", inp.namespace, "-l", f"app={inp.app_label}", "-o", "json"])
    pods = json.loads(pods_json).get("items", []) if pods_json.strip().startswith("{") else []
    pod_names = [p["metadata"]["name"] for p in pods]

    # 3) Logs (last ~80 lines each)
    logs = {}
    for pn in pod_names[:2]:
        logs[pn] = sh(["kubectl", "logs", "-n", inp.namespace, pn, "--tail=80"])

    # 4) Metrics: error rate + p95 slow latency
    err_q = 'rate(demo_http_requests_total{path="/error",status="500"}[2m])'
    p95_q = 'histogram_quantile(0.95, sum by (le) (rate(demo_http_request_duration_seconds_bucket{path="/slow"}[5m])))'
    err = prom_query(err_q)
    p95 = prom_query(p95_q)

    # 5) Simple diagnosis (LLM can replace this later)
    diagnosis = []
    if any("CrashLoopBackOff" in sh(["kubectl","get","pods","-n",inp.namespace]).strip() for _ in [0]):
        diagnosis.append("Pods appear to be crash-looping (check events/logs).")
    if err:
        diagnosis.append("5xx errors are occurring on /error (check logs for repeating error lines).")
    if p95:
        diagnosis.append("p95 latency for /slow is elevated (likely intentional delay or resource pressure).")
        
    pods_summary = [{"name": p["metadata"]["name"], "phase": p["status"]["phase"]} for p in pods]

    prompt = f"""
        You are an on-call copilot. Use ONLY the evidence provided.
        If something is missing, say what is missing and what command to run next.

        Question:
        {inp.question}

        Evidence: Top runbook chunks (docs_top):
        {docs}

        Evidence: Kubernetes pods:
        {pods_summary}

        Evidence: Recent logs (logs_tail):
        {logs}

        Evidence: Prometheus metrics:
        error_rate_query: {err_q}
        error_rate_result: {err}
        p95_query: {p95_q}
        p95_result: {p95}

        Return:
        1) Most likely cause (1–3 bullets)
        2) Evidence (cite runbook source filenames + mention log lines/metric results you used)
        3) Immediate next steps (3–6 bullets)
        """

    narrative = None
    try:
        llm = create_llm_provider()
        if llm:
            narrative = llm.generate(prompt)
    except Exception as e:
        narrative = f"(LLM unavailable: {e})"


    return {
        "question": inp.question,
        "narrative": narrative,
        "likely_findings": diagnosis or ["No obvious issue detected from quick checks."],
        "evidence": {
            "docs_top": docs,
            "pods": pods_summary,
            "logs_tail": logs,
            "prometheus": {
                "error_rate_query": err_q,
                "error_rate_result": err,
                "p95_query": p95_q,
                "p95_result": p95
            },
        },
        "next_steps": [
            "If errors: inspect logs_tail for repeated error patterns and correlate with metrics spike time.",
            "If slow: check pod CPU/memory and consider adding resource limits/throttling to simulate pressure.",
            "Later: plug in an LLMProvider to turn evidence into a polished narrative."
        ]
    }

