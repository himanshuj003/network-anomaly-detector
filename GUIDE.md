# Network Anomaly Detector — User Guide

## 1. Overview

This project demonstrates **network anomaly detection** for cybersecurity use-cases:

- Detect port scans, floods, large data transfers, and unusual behavior
- Compare statistical, machine-learning, and rule-based approaches
- Evaluate performance when ground-truth labels are available
- Explore results interactively in a dashboard

---

## 2. Installation

```bash
cd network-anomaly-detector
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Generate Demo Data

```bash
python cli.py generate -n 5000 -r 0.05 -o data/synthetic_flows.csv
```

Injected anomaly types: `port_scan`, `ddos_flood`, `exfiltration`, `rare_port`, `rst_storm`.

---

## 4. Run Detection from CLI

```bash
python cli.py detect -i data/synthetic_flows.csv -m hybrid -o results.csv
python cli.py detect -i data/synthetic_flows.csv -m statistical --threshold 3.5
python cli.py detect -i data/synthetic_flows.csv -m isolation_forest --contamination 0.05
python cli.py detect -i data/synthetic_flows.csv -m rule_based
```

---

## 5. Evaluate Against Ground Truth

```bash
python cli.py evaluate -i data/synthetic_flows.csv -m hybrid
```

---

## 6. One-Command Demo

```bash
python cli.py demo
```

---

## 7. Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 8. Using Your Own Data

Export flows from Zeek, Suricata, or nfdump. Map columns to: `packet_count`, `byte_count`, `duration_sec`, `syn_count`, `rst_count`, `unique_dst_ports`, `dst_port`, etc.

---

## 9. Understanding the Detectors

| Method | Strength |
|--------|----------|
| Statistical (MAD) | Fast, explainable |
| Isolation Forest | Complex multi-feature outliers |
| Rule-based | Explicit attack signatures |
| Hybrid | Best precision/recall balance |

---

## 10. Safety & Ethics

This tool is for **defensive** monitoring and research. Do not use it to attack or scan networks you do not own or have permission to test.

---

## 11. Troubleshooting

| Issue | Fix |
|-------|-----|
| ModuleNotFoundError | Run from project root |
| Low recall | Lower threshold / raise contamination |
| Too many FPs | Raise threshold / min-votes |
