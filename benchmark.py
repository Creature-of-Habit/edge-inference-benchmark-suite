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