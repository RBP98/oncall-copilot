from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .types import (
    ALLOWED_APP_LABELS,
    ALLOWED_NAMESPACES,
    MAX_LOG_LINES,
    ToolEnvelope,
    make_envelope,
    make_error_envelope,
    run_kubectl,
    truncate_text,
)


class K8sGetPodsArgs(BaseModel):
    namespace: str = Field(..., description="Allowlisted namespace")
    app_label: str = Field(..., description="Allowlisted app label value")
    limit: int = Field(20, ge=1, le=200)


class PodInfo(BaseModel):
    name: str
    phase: Optional[str] = None
    restarts: int = 0
    waiting_reason: Optional[str] = None


class K8sGetPodsResult(BaseModel):
    pods: List[PodInfo]


class K8sGetLogsArgs(BaseModel):
    namespace: str
    pod_name: str
    tail_lines: int = Field(MAX_LOG_LINES, ge=1, le=MAX_LOG_LINES)


class K8sGetLogsResult(BaseModel):
    pod_name: str
    logs_tail: str
    truncated: bool = False


def _validate(ns: str, app_label: str) -> Optional[str]:
    if ns not in ALLOWED_NAMESPACES:
        return f"Namespace '{ns}' not allowed. Allowed: {sorted(ALLOWED_NAMESPACES)}"
    if app_label not in ALLOWED_APP_LABELS:
        return f"app_label '{app_label}' not allowed. Allowed: {sorted(ALLOWED_APP_LABELS)}"
    return None


def get_pods(args: K8sGetPodsArgs, timeout_s: int = 10) -> ToolEnvelope:
    started = time.time()
    try:
        err = _validate(args.namespace, args.app_label)
        if err:
            return make_error_envelope("k8s.get_pods", args, "ValidationError", err, None, started)

        selector = f"app={args.app_label}"
        cp = run_kubectl(["get", "pods", "-n", args.namespace, "-l", selector, "-o", "json"], timeout_s=timeout_s)
        if cp.returncode != 0:
            msg, _ = truncate_text(cp.stderr or cp.stdout or "kubectl get pods failed")
            return make_error_envelope(
                "k8s.get_pods",
                args,
                "KubectlError",
                msg,
                {"returncode": cp.returncode},
                started,
            )

        data = json.loads(cp.stdout)
        items = (data.get("items") or [])[: args.limit]
        pods: List[PodInfo] = []

        for it in items:
            name = it.get("metadata", {}).get("name")
            phase = it.get("status", {}).get("phase")
            restarts = 0
            waiting_reason = None

            for cs in it.get("status", {}).get("containerStatuses", []) or []:
                restarts += int(cs.get("restartCount") or 0)
                state = cs.get("state") or {}
                waiting = state.get("waiting") or {}
                if waiting.get("reason"):
                    waiting_reason = waiting.get("reason")

            pods.append(PodInfo(name=name, phase=phase, restarts=restarts, waiting_reason=waiting_reason))

        return make_envelope("k8s.get_pods", args, K8sGetPodsResult(pods=pods), started)

    except Exception as e:
        return make_error_envelope("k8s.get_pods", args, type(e).__name__, str(e), None, started)


def get_logs(args: K8sGetLogsArgs, timeout_s: int = 10) -> ToolEnvelope:
    started = time.time()
    try:
        if args.namespace not in ALLOWED_NAMESPACES:
            return make_error_envelope(
                "k8s.get_logs",
                args,
                "ValidationError",
                f"Namespace '{args.namespace}' not allowed. Allowed: {sorted(ALLOWED_NAMESPACES)}",
                None,
                started,
            )

        cp = run_kubectl(
            ["logs", "-n", args.namespace, args.pod_name, f"--tail={args.tail_lines}"],
            timeout_s=timeout_s,
        )
        if cp.returncode != 0:
            msg, _ = truncate_text(cp.stderr or cp.stdout or "kubectl logs failed")
            return make_error_envelope(
                "k8s.get_logs",
                args,
                "KubectlError",
                msg,
                {"returncode": cp.returncode},
                started,
            )

        text = cp.stdout or ""
        text, truncated = truncate_text(text)
        return make_envelope("k8s.get_logs", args, K8sGetLogsResult(pod_name=args.pod_name, logs_tail=text, truncated=truncated), started)

    except Exception as e:
        return make_error_envelope("k8s.get_logs", args, type(e).__name__, str(e), None, started)
