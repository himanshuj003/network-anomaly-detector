# Network Anomaly Detector

A practical **cybersecurity** project for detecting anomalous network traffic using multiple complementary approaches:

| Method | Type | Best for |
|--------|------|----------|
| **Statistical (MAD / Z-score)** | Unsupervised stats | Fast baseline, explainable outliers |
| **Isolation Forest** | Unsupervised ML | Complex multi-feature anomalies |
| **Rule-based** | Signature / heuristics | Port scans, SYN floods, known bad ports |
| **Hybrid ensemble** | Voting + score fusion | Higher confidence detections |

Includes:
- **CLI** for generation, detection, and evaluation
- **Streamlit dashboard** for interactive analysis
- **Synthetic data generator** with realistic anomalies
- Full project scaffold ready to extend

---

## Quick Start

```bash
cd network-anomaly-detector
pip install -r requirements.txt
python cli.py demo
streamlit run dashboard/app.py
```

---

## CLI Usage

```bash
python cli.py generate -n 5000 -r 0.05 -o data/flows.csv
python cli.py detect -i data/flows.csv -m hybrid -o results.csv
python cli.py detect -i data/flows.csv -m statistical --threshold 3.5
python cli.py detect -i data/flows.csv -m isolation_forest --contamination 0.05
python cli.py detect -i data/flows.csv -m rule_based
python cli.py evaluate -i data/flows.csv -m hybrid
```

---

## Project Structure

```
network-anomaly-detector/
├── cli.py
├── requirements.txt
├── README.md
├── GUIDE.md
├── src/
│   ├── data_generator.py
│   ├── features.py
│   ├── detectors.py
│   └── evaluator.py
├── dashboard/
│   └── app.py
├── data/
├── examples/
├── docs/
└── tests/
```

## Detection Methods

1. **Statistical (Robust MAD)** — median absolute deviation on log-transformed features
2. **Isolation Forest** — unsupervised ML for complex outliers
3. **Rule-based** — port scans, SYN floods, RST storms, large transfers, suspicious ports
4. **Hybrid** — majority vote + score fusion

## License

MIT
