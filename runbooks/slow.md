# Slow requests runbook
If /slow p95 latency is high:
- Check CPU throttling and pod restarts
- Check app logs around the time latency increased
- Confirm traffic/error changes in Prometheus
