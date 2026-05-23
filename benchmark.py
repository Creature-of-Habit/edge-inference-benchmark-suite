import argparse
import csv
from pathlib import Path
import time                  # Will be needed for tracking latency clock cycles
import numpy as np           # Will be needed to handle the array/tensor math
import onnxruntime as ort    # Handles model execution and thread constraints

def load_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # Lock to a single execution thread
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],  # Force pure CPU execution
    )
    return session

def build_dummy_inputs(session: ort.InferenceSession) -> dict:
    inputs = {}
    for inp in session.get_inputs():
        # Handle dynamic dimensions (replace -1 or None with 1)
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
        
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }
        dtype = dtype_map.get(inp.type, np.float32)
        
        # Determine if the model requires integers or floating points
        if np.issubdtype(dtype, np.integer):
            inputs[inp.name] = np.random.randint(1, 1000, size=shape, dtype=dtype)
        else:
            inputs[inp.name] = np.random.rand(*shape).astype(dtype)
    return inputs
    
def run_benchmark(session: ort.InferenceSession, inputs: dict, runs: int = 10):
    # Grab the exact name of the output layer from metadata
    output_names = [out.name for out in session.get_outputs()]
    
    latencies = []
    
    for _ in range(runs):
        start_time = time.perf_counter()
        
        # Run a single mathematical pass through the network
        session.run(output_names, inputs)
        
        end_time = time.perf_counter()
        
        # Calculate how long that specific pass took in milliseconds
        latency = (end_time - start_time) * 1000
        latencies.append(latency)
        
    return latencies

def summarise(latencies: list) -> dict:
    # Exclude the first run to remove the "cold start" cache distortion
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    
    metrics = {
        "min_latency_ms":  float(np.min(warm_latencies)),
        "max_latency_ms":  float(np.max(warm_latencies)),
        "mean_latency_ms": float(np.mean(warm_latencies)),
        "p95_latency_ms":  float(np.percentile(warm_latencies, 95)),
        "throughput_fps":  float(1000.0 / np.mean(warm_latencies))
    }
    return metrics
def save_results(metrics: dict, model_path: str, output_csv: str = "benchmark_results.csv"):
    file_exists = Path(output_csv).exists()
    
    # Open the file in append mode ('a') so we don't overwrite previous tests
    with open(output_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        
        # If it's a brand new file, write the column headers first
        if not file_exists:
            writer.writerow(["model_name"] + list(metrics.keys()))
            
        # Write the actual data row
        model_name = Path(model_path).name
        writer.writerow([model_name] + list(metrics.values()))

# NEW: The CLI orchestration block
def main():
    parser = argparse.ArgumentParser(description="Edge Inference Benchmark Suite")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--runs", type=int, default=50, help="Number of inference iterations")
    args = parser.parse_args()

    print(f"Loading session for {args.model}...")
    session = load_session(args.model)
    
    print("Generating matching mock input tensors...")
    inputs = build_dummy_inputs(session)
    
    print(f"Running execution loop for {args.runs} iterations...")
    latencies = run_benchmark(session, inputs, args.runs)
    
    print("Aggregating edge performance metrics...")
    metrics = summarise(latencies)
    
    print("\n--- RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    save_results(metrics, args.model)
    print(f"\nSuccessfully saved results to benchmark_results.csv!")

if __name__ == "__main__":
    main()