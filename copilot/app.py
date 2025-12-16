import json
from fastapi import FastAPI
from pydantic import BaseModel

from copilot.providers.llm.factory import create_llm_provider

from copilot.tools.docs_tool import search as docs_search, DocsSearchArgs
from copilot.tools.k8s_tool import get_pods, get_logs, K8sGetPodsArgs, K8sGetLogsArgs
from copilot.tools.prom_tool import query as prom_query, PromQueryArgs

from copilot.agent_loop import run_agent

app = FastAPI(title="oncall-copilot")


class ChatIn(BaseModel):
    question: str
    namespace: str = "oncall"
    app_label: str = "demo-app"

@app.post("/agent_chat")
def agent_chat(inp: ChatIn):
    llm = create_llm_provider()
    if not llm:
        return {"question": inp.question, "final": "LLM_PROVIDER is not set; agent mode requires an LLM.", "trace": [], "evidence": {}}

    out = run_agent(inp.question, inp.namespace, inp.app_label, llm)
    return {"question": inp.question, **out}

@app.post("/chat")
def chat(inp: ChatIn):
    tool_trace = []

    # 1) Pull docs
    docs_env = docs_search(DocsSearchArgs(question=inp.question, top_k=3))
    tool_trace.append(docs_env)

    docs = []
    if docs_env.result and "chunks" in docs_env.result:
        docs = docs_env.result["chunks"]

    # 2) K8s state
    pods_env = get_pods(K8sGetPodsArgs(namespace=inp.namespace, app_label=inp.app_label, limit=50))
    tool_trace.append(pods_env)

    pods_summary = []
    pod_names = []
    if pods_env.result and "pods" in pods_env.result:
        pods_summary = pods_env.result["pods"]
        pod_names = [p["name"] for p in pods_summary if p.get("name")]

    # 3) Logs (last ~80 lines each)
    logs = {}
    logs_envs = []
    for pn in pod_names[:2]:
        env = get_logs(K8sGetLogsArgs(namespace=inp.namespace, pod_name=pn, tail_lines=80))
        logs_envs.append(env)
        tool_trace.append(env)
        if env.result and "logs_tail" in env.result:
            logs[pn] = env.result["logs_tail"]
        else:
            logs[pn] = f"(no logs: {env.error.message if env.error else 'unknown'})"

    # 4) Metrics: error rate + p95 slow latency
    err_q = 'rate(demo_http_requests_total{path="/error",status="500"}[2m])'
    p95_q = 'histogram_quantile(0.95, sum by (le) (rate(demo_http_request_duration_seconds_bucket{path="/slow"}[5m])))'

    err_env = prom_query(PromQueryArgs(promql=err_q))
    p95_env = prom_query(PromQueryArgs(promql=p95_q))
    tool_trace.extend([err_env, p95_env])

    err = err_env.result.get("result") if err_env.result else {"error": err_env.error.message if err_env.error else "unknown"}
    p95 = p95_env.result.get("result") if p95_env.result else {"error": p95_env.error.message if p95_env.error else "unknown"}

    # 5) Simple diagnosis (still deterministic)
    diagnosis = []
    if any((p.get("waiting_reason") == "CrashLoopBackOff") for p in pods_summary):
        diagnosis.append("Pods appear to be crash-looping (CrashLoopBackOff reported; check logs/events).")
    if err:
        diagnosis.append("5xx errors are occurring on /error (check logs for repeating error lines).")
    if p95:
        diagnosis.append("p95 latency for /slow is elevated (likely intentional delay or resource pressure).")

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
""".strip()

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
                "p95_result": p95,
            },
        },
        # Phase 1 bonus: you now have a tool trace ready for Phase 2 agent loops
        "tool_trace": [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in tool_trace],
        "next_steps": [
            "If errors: inspect logs_tail for repeated error patterns and correlate with metrics spike time.",
            "If slow: check pod CPU/memory and consider adding resource limits/throttling to simulate pressure.",
            "Later: plug in an LLMProvider to turn evidence into a polished narrative.",
        ],
    }
