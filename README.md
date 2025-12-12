\# On-Call Copilot (Local)



A local end-to-end “on-call copilot” prototype that answers incident-style questions using:

\- Runbooks (RAG via Qdrant)

\- Kubernetes state (pods/logs/events via kubectl)

\- Prometheus metrics (latency/errors)



\## Stack

\- Minikube (Docker driver)

\- kube-prometheus-stack (Prometheus + Grafana)

\- Demo app (FastAPI) deployed to K8s (exports Prometheus metrics)

\- Qdrant (vector DB) for runbooks

\- Copilot API (FastAPI) running locally



\## Quickstart (Windows PowerShell)

See: `docs/oncall-copilot-command-reference-windows.docx`



\## Example Questions

\- Why are we seeing errors?

\- Why is the service slow?

\- Are any pods crash-looping?



