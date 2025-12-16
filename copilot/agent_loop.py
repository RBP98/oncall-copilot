from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ValidationError

from copilot.tools.docs_tool import search as docs_search, DocsSearchArgs
from copilot.tools.k8s_tool import get_pods, get_logs, K8sGetPodsArgs, K8sGetLogsArgs
from copilot.tools.prom_tool import query as prom_query, PromQueryArgs


MAX_STEPS = 6

# ✅ Use the SAME known-good queries as /chat (do not let the model guess names/labels)
ERR_Q = 'rate(demo_http_requests_total{path="/error",status="500"}[2m])'
P95_Q = 'histogram_quantile(0.95, sum by (le) (rate(demo_http_request_duration_seconds_bucket{path="/slow"}[5m])))'
PROMQL_ALLOWLIST = {ERR_Q, P95_Q}


class AgentAction(BaseModel):
    type: Literal["tool", "final"]
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    final: Optional[str] = None


def _as_dict(x: Any) -> Dict[str, Any]:
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    if isinstance(x, dict):
        return x
    return {"value": x}


def _extract_json_object(text: str) -> Dict[str, Any]:
    t = (text or "").strip()

    # strip common ```json fences
    if t.startswith("```"):
        t = t.strip().strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")

    return json.loads(t[start : end + 1])


def _tool_schemas() -> Dict[str, Any]:
    def schema(m):
        return m.model_json_schema() if hasattr(m, "model_json_schema") else {}

    return {
        "docs.search": schema(DocsSearchArgs),
        "k8s.get_pods": schema(K8sGetPodsArgs),
        "k8s.get_logs": schema(K8sGetLogsArgs),
        "prom.query": schema(PromQueryArgs),
    }


def _registry():
    return {
        "docs.search": (DocsSearchArgs, lambda a: docs_search(a)),
        "k8s.get_pods": (K8sGetPodsArgs, lambda a: get_pods(a)),
        "k8s.get_logs": (K8sGetLogsArgs, lambda a: get_logs(a)),
        "prom.query": (PromQueryArgs, lambda a: prom_query(a)),
    }


def _has_tool(trace: List[Dict[str, Any]], tool_name: str) -> bool:
    return any((t.get("tool") == tool_name) for t in trace)


def _first_pod_name(trace: List[Dict[str, Any]]) -> Optional[str]:
    for t in trace:
        if t.get("tool") == "k8s.get_pods":
            pods = (t.get("result") or {}).get("pods") or []
            if pods and pods[0].get("name"):
                return pods[0]["name"]
    return None


def _has_promql(trace: List[Dict[str, Any]], promql: str) -> bool:
    for t in trace:
        if t.get("tool") == "prom.query":
            args = t.get("args") or {}
            if args.get("promql") == promql:
                return True
    return False


def _missing_summary(trace: List[Dict[str, Any]]) -> Dict[str, bool]:
    return {
        "docs": not _has_tool(trace, "docs.search"),
        "pods": not _has_tool(trace, "k8s.get_pods"),
        "logs": not _has_tool(trace, "k8s.get_logs"),
        "err_metric": not _has_promql(trace, ERR_Q),
        "p95_metric": not _has_promql(trace, P95_Q),
    }


def _build_evidence_from_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence = {"docs_top": [], "pods": [], "logs_tail": {}, "prometheus": []}

    for item in trace:
        tool = item.get("tool")
        res = item.get("result") or {}
        if tool == "docs.search":
            evidence["docs_top"] = res.get("chunks", res.get("hits", [])) or []
        elif tool == "k8s.get_pods":
            evidence["pods"] = res.get("pods", []) or []
        elif tool == "k8s.get_logs":
            pod = res.get("pod_name")
            if pod:
                evidence["logs_tail"][pod] = res.get("logs_tail", "")
        elif tool == "prom.query":
            evidence["prometheus"].append(
                {"promql": (res.get("promql") or item.get("args", {}).get("promql")), "result": res.get("result")}
            )

    return evidence


def run_agent(question: str, namespace: str, app_label: str, llm) -> Dict[str, Any]:
    tools = _registry()
    schemas = _tool_schemas()

    trace: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_calls: set[Tuple[str, str]] = set()

    system = f"""
You are an on-call copilot running in AGENT MODE.

You can call tools to gather evidence. Your FINAL answer must use ONLY tool evidence.
If evidence is insufficient, say what is missing and what tool/command you'd run next.

Available tools and their arg JSON schemas:
{json.dumps(schemas, indent=2)}

Hard rules:
- Reply with ONLY ONE JSON object:
  {{
    "type": "tool" | "final",
    "tool_name": "docs.search" | "k8s.get_pods" | "k8s.get_logs" | "prom.query",
    "tool_args": {{...}},
    "final": "..."
  }}
- If type=="tool": include tool_name + tool_args (no final).
- If type=="final": include final (no tool_name/tool_args).

CRITICAL:
- Do NOT invent Prometheus metric names/labels.
- For this demo, ONLY use these PromQL queries:
  ERR_Q = {ERR_Q}
  P95_Q = {P95_Q}

Context:
- Namespace is fixed to "{namespace}"
- App label is fixed to "{app_label}"
""".strip()

    for step in range(MAX_STEPS):
        missing = _missing_summary(trace)

        prompt = f"""
{system}

USER QUESTION:
{question}

MISSING EVIDENCE:
{json.dumps(missing, indent=2)}

TOOL TRACE (last 8):
{json.dumps(trace[-8:], indent=2)}
""".strip()

        raw = llm.generate(prompt, meta={"mode": "agent", "step": step})
        try:
            action = AgentAction(**_extract_json_object(raw))
        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            return {
                "final": f"Agent failed to produce a valid JSON action. Raw:\n{raw}",
                "trace": trace,
                "evidence": _build_evidence_from_trace(trace),
                "error": str(e),
            }

        # ✅ Guardrail: don't allow "final" until minimum evidence is present
        # (otherwise the model will confidently guess)
        if action.type == "final" and (missing["docs"] or missing["pods"] or missing["logs"]):
            action = AgentAction(type="tool", tool_name="docs.search", tool_args={"question": question, "top_k": 3})

        # ✅ Hard-enforce the required evidence order (light “autopilot”)
        # This prevents skipping docs and prevents wrong PromQL.
        if missing["docs"]:
            action = AgentAction(type="tool", tool_name="docs.search", tool_args={"question": question, "top_k": 3})
        elif missing["pods"]:
            action = AgentAction(type="tool", tool_name="k8s.get_pods", tool_args={"namespace": namespace, "app_label": app_label, "limit": 50})
        elif missing["logs"]:
            pn = _first_pod_name(trace)
            if pn:
                action = AgentAction(type="tool", tool_name="k8s.get_logs", tool_args={"namespace": namespace, "pod_name": pn, "tail_lines": 80})
        else:
            # After docs/pods/logs exist, enforce known-good PromQL collection
            if missing["err_metric"]:
                action = AgentAction(type="tool", tool_name="prom.query", tool_args={"promql": ERR_Q})
            elif missing["p95_metric"]:
                action = AgentAction(type="tool", tool_name="prom.query", tool_args={"promql": P95_Q})

        if action.type == "final":
            return {
                "final": action.final or "",
                "trace": trace,
                "evidence": _build_evidence_from_trace(trace),
            }

        tool_name = action.tool_name or ""
        tool_args = action.tool_args or {}

        if tool_name not in tools:
            trace.append({"tool": tool_name, "args": tool_args, "error": {"type": "UnknownTool", "message": "Tool not supported"}})
            continue

        # Enforce fixed namespace/app_label so model can't escape
        if tool_name == "k8s.get_pods":
            tool_args["namespace"] = namespace
            tool_args["app_label"] = app_label
        if tool_name == "k8s.get_logs":
            tool_args["namespace"] = namespace
        if tool_name == "prom.query":
            # ✅ Reject/override any non-allowlisted PromQL (prevents guessing)
            if tool_args.get("promql") not in PROMQL_ALLOWLIST:
                tool_args["promql"] = ERR_Q if not _has_promql(trace, ERR_Q) else P95_Q

        key = (tool_name, json.dumps(tool_args, sort_keys=True))
        if key in seen_calls:
            trace.append({"tool": tool_name, "args": tool_args, "warning": "Repeated identical tool call blocked"})
            break
        seen_calls.add(key)

        if key in cache:
            env = cache[key]
        else:
            args_model, runner = tools[tool_name]
            try:
                args_obj = args_model(**tool_args)
                env_obj = runner(args_obj)
                env = _as_dict(env_obj)
            except Exception as e:
                env = {"tool": tool_name, "args": tool_args, "result": None, "error": {"type": type(e).__name__, "message": str(e)}}
            cache[key] = env

        trace.append(env)

    return {
        "final": "Reached max agent steps. Returning gathered evidence and trace.",
        "trace": trace,
        "evidence": _build_evidence_from_trace(trace),
    }
