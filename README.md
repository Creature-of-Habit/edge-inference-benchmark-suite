# Edge Inference Benchmark Suite

A lightweight, highly modular Python Command Line Interface (CLI) tool designed to profile the hardware performance of machine learning models on edge CPUs.

This suite utilizes **ONNX Runtime** to isolate inference execution, dynamically generate mock tensors based on model metadata, and calculate rigorous latency and throughput metrics while properly isolating "cold start" anomalies.

## 🚀 Features

* **Dynamic Data Generation:** Automatically reads `.onnx` input metadata (shapes and data types) to synthesize exact matching N-Dimensional dummy arrays.
* **Thread-Locked Profiling:** Configured for strict single-thread execution (`intra_op` and `inter_op`) to accurately simulate resource-constrained edge environments.
* **Cold-Start Isolation:** Automatically separates initial memory-allocation/cache-warming spikes from warm execution metrics.
* **Persistent Analytics:** Exports benchmark runs (Min, Max, Mean, P95, and FPS) directly to a structured CSV dashboard for long-term hardware comparison.
* **Modular Architecture:** Cleanly separated concerns (`engine`, `data`, `metrics`, `storage`) for high maintainability and easy extension.

## 📂 Project Structure

```text
edge-inference-benchmark-suite/
│
├── src/
│   ├── __init__.py
│   ├── engine.py       # Handles ONNX session loading and execution loop
│   ├── data.py         # Handles dynamic metadata parsing and tensor generation
│   ├── metrics.py      # Calculates latency statistics and throughput math
│   └── storage.py      # Manages CSV data persistence
│
├── benchmark.py        # The main CLI entry point
├── benchmark_results.csv # Auto-generated performance dashboard
└── README.md
```

## 🛠️ Prerequisites

* Python 3.8+
* [`uv`](https://github.com/astral-sh/uv) package manager

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Creature-of-Habit/edge-inference-benchmark-suite.git
cd edge-inference-benchmark-suite
```

2. Install dependencies:
```bash
uv pip install onnxruntime numpy
```

## 📖 How to Use

### Step 1: Prepare Your Model

You need an `.onnx` model file in the project root. If you don't have one, download a small test model (e.g. a ~400KB MNIST classifier) from the [ONNX Model Zoo](https://github.com/onnx/models).

### Step 2: Run the Benchmark

```bash
uv run benchmark.py --model your_model.onnx
```

To increase the number of inference runs (default is 50):

```bash
uv run benchmark.py --model your_model.onnx --runs 1000
```

### Step 3: Read the Terminal Output

```text
Loading session for your_model.onnx...
Generating matching mock input tensors...
Running execution loop for 50 iterations...
Aggregating edge performance metrics...

--- RESULTS ---
min_latency_ms:   0.0503
max_latency_ms:   0.0761
mean_latency_ms:  0.0559
p95_latency_ms:   0.0690
throughput_fps:   17899.76
```

### Step 4: Analyze the CSV Dashboard

Every run automatically appends a row to `benchmark_results.csv`. Open it in Excel, Google Sheets, or load it with Pandas to compare models across hardware over time.

```python
import pandas as pd
df = pd.read_csv("benchmark_results.csv")
print(df)
```

## 📊 Metrics Explained

| Metric | Description |
|---|---|
| `min_latency_ms` | Fastest inference cycle recorded |
| `max_latency_ms` | Slowest cycle recorded (cold-start excluded) |
| `mean_latency_ms` | Average time per inference |
| `p95_latency_ms` | 95th percentile — worst-case consistency indicator |
| `throughput_fps` | Inferences per second (`1000 / mean_latency_ms`) |