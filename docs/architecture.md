# Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Data Sources   │     │  Feature Layer   │     │   Detectors     │
│                 │     │                  │     │                 │
│ • Synthetic gen │────▶│ • Numeric select │────▶│ • Statistical   │
│ • CSV upload    │     │ • log1p transform│     │ • Isolation For.│
│ • (future) pcap │     │ • derived rates  │     │ • Rule-based    │
│ • (future) Zeek │     │                  │     │ • Hybrid        │
└─────────────────┘     └──────────────────┘     └────────┐────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────┐
                                                 │  Output Layer   │
                                                 │                 │
                                                 │ • CLI tables    │
                                                 │ • CSV results   │
                                                 │ • Streamlit UI  │
                                                 │ • Metrics       │
                                                 └────────────────┘
```

## Design Principles

1. **Modular detectors** — each implements `fit` / `predict` / `fit_predict` and returns a common `DetectionResult`.
2. **Explainability first** — statistical and rule-based methods are first-class citizens alongside ML.
3. **Evaluation-ready** — ground-truth support is built-in for synthetic and labeled datasets.
4. **Minimal friction** — one `pip install` + CLI / one-command dashboard.
