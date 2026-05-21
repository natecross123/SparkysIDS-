# SparkysIDS
A intrusion detection system utilising ML and written rules.

## Near-real-time botnet rule engine
This repository now includes `botnet_rules.py`, a 5-tuple flow-based rule engine designed for near-real-time IDS experiments.

### Highlights
- Uses **5-tuple flow keys**: `(src_ip, dst_ip, src_port, dst_port, protocol)`.
- Supports **live packet capture** with Scapy `sniff()` via `run_live_capture()`.
- Adds **time-based features** (inter-arrival mean/std, packets/sec, bytes/sec).
- Handles DNS separately and avoids treating port 53 as automatically malicious.
- Returns transparent decision metadata (`rule_score`, `threshold`, flow statistics) for easy fusion with ML model output.

### Quick usage
```python
from botnet_rules import run_live_capture

results = run_live_capture(interface="eth0", timeout=30)
for flow_key, decision in results:
    print(flow_key, decision)
```
