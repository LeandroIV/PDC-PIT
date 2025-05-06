"""
Simple GPU Programming and Hardware Acceleration Demo

This script demonstrates the performance difference between CPU and GPU
for matrix operations. It's a simplified version that doesn't require
matplotlib and focuses on clear console output.

Requirements:
- Python 3.6+
- PyTorch
"""

import torch
import time
import os

def print_separator():
    print("\n" + "="*70 + "\n")

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_separator()
print(f"DEVICE INFORMATION:")
print(f"Using device: {device}")

# If GPU is available, print some information about it
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No GPU available. Running on CPU only.")
    print("Note: Performance comparison will still run, but both will use CPU.")

def matrix_multiplication_benchmark(sizes):
    """Benchmark matrix multiplication on CPU and GPU."""
    print_separator()
    print("MATRIX MULTIPLICATION BENCHMARK")
    print("This test multiplies two matrices of increasing sizes")
    print("and compares the time taken on CPU vs GPU.")
    print_separator()
    
    results = []
    
    for size in sizes:
        print(f"Testing matrix multiplication with size {size}x{size}...")
        
        # Create random matrices
        matrix_a = torch.rand(size, size)
        matrix_b = torch.rand(size, size)
        
        # CPU benchmark
        start_time = time.time()
        result_cpu = torch.matmul(matrix_a, matrix_b)
        cpu_time = time.time() - start_time
        print(f"  CPU time: {cpu_time:.6f} seconds")
        
        # GPU benchmark (if available)
        if device.type == "cuda":
            # Move matrices to GPU
            matrix_a_gpu = matrix_a.to(device)
            matrix_b_gpu = matrix_b.to(device)
            
            # Warm-up run
            _ = torch.matmul(matrix_a_gpu, matrix_b_gpu)
            torch.cuda.synchronize()
            
            # Timed run
            start_time = time.time()
            result_gpu = torch.matmul(matrix_a_gpu, matrix_b_gpu)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = time.time() - start_time
            
            # Verify results match
            result_from_gpu = result_gpu.cpu()
            is_close = torch.allclose(result_cpu, result_from_gpu, rtol=1e-3, atol=1e-3)
            
            print(f"  GPU time: {gpu_time:.6f} seconds")
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
            print(f"  Results match: {is_close}")
            
            results.append((size, cpu_time, gpu_time, cpu_time / gpu_time))
        else:
            # If no GPU, just duplicate CPU time
            results.append((size, cpu_time, cpu_time, 1.0))
    
    return results

def neural_network_benchmark(batch_sizes):
    """Benchmark neural network forward and backward pass on CPU and GPU."""
    print_separator()
    print("NEURAL NETWORK BENCHMARK")
    print("This test performs forward and backward passes through a neural network")
    print("with different batch sizes and compares CPU vs GPU performance.")
    print_separator()
    
    results = []
    
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = torch.nn.Linear(784, 256)
            self.fc2 = torch.nn.Linear(256, 128)
            self.fc3 = torch.nn.Linear(128, 10)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    for batch_size in batch_sizes:
        print(f"Testing neural network with batch size {batch_size}...")
        
        # Create random input and target data
        input_data = torch.rand(batch_size, 784)
        target = torch.randint(0, 10, (batch_size,))
        
        # CPU benchmark
        model_cpu = SimpleNN()
        optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        start_time = time.time()
        # Forward pass
        output_cpu = model_cpu(input_data)
        loss_cpu = criterion(output_cpu, target)
        # Backward pass
        optimizer_cpu.zero_grad()
        loss_cpu.backward()
        optimizer_cpu.step()
        
        cpu_time = time.time() - start_time
        print(f"  CPU time: {cpu_time:.6f} seconds")
        
        # GPU benchmark (if available)
        if device.type == "cuda":
            model_gpu = SimpleNN().to(device)
            optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), lr=0.01)
            input_data_gpu = input_data.to(device)
            target_gpu = target.to(device)
            
            # Warm-up run
            _ = model_gpu(input_data_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            # Forward pass
            output_gpu = model_gpu(input_data_gpu)
            loss_gpu = criterion(output_gpu, target_gpu)
            # Backward pass
            optimizer_gpu.zero_grad()
            loss_gpu.backward()
            optimizer_gpu.step()
            torch.cuda.synchronize()  # Wait for GPU to finish
            
            gpu_time = time.time() - start_time
            print(f"  GPU time: {gpu_time:.6f} seconds")
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
            
            results.append((batch_size, cpu_time, gpu_time, cpu_time / gpu_time))
        else:
            # If no GPU, just duplicate CPU time
            results.append((batch_size, cpu_time, cpu_time, 1.0))
    
    return results

def vector_addition_benchmark(sizes):
    """Benchmark vector addition on CPU and GPU."""
    print_separator()
    print("VECTOR ADDITION BENCHMARK")
    print("This test adds two vectors of increasing sizes")
    print("and compares the time taken on CPU vs GPU.")
    print_separator()
    
    results = []
    
    for size in sizes:
        size_str = f"{size:,}"
        print(f"Testing vector addition with size {size_str}...")
        
        # Create random vectors
        vector_a = torch.rand(size)
        vector_b = torch.rand(size)
        
        # CPU benchmark
        start_time = time.time()
        result_cpu = vector_a + vector_b
        cpu_time = time.time() - start_time
        print(f"  CPU time: {cpu_time:.6f} seconds")
        
        # GPU benchmark (if available)
        if device.type == "cuda":
            # Move vectors to GPU
            vector_a_gpu = vector_a.to(device)
            vector_b_gpu = vector_b.to(device)
            
            # Warm-up run
            _ = vector_a_gpu + vector_b_gpu
            torch.cuda.synchronize()
            
            # Timed run
            start_time = time.time()
            result_gpu = vector_a_gpu + vector_b_gpu
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = time.time() - start_time
            
            print(f"  GPU time: {gpu_time:.6f} seconds")
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
            
            results.append((size, cpu_time, gpu_time, cpu_time / gpu_time))
        else:
            # If no GPU, just duplicate CPU time
            results.append((size, cpu_time, cpu_time, 1.0))
    
    return results

def print_summary_table(title, headers, rows):
    """Print a nicely formatted table of results."""
    print_separator()
    print(title)
    print_separator()
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    
    # Print header
    header_row = "".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
    print(header_row)
    print("-" * sum(col_widths))
    
    # Print rows
    for row in rows:
        formatted_row = []
        for i, item in enumerate(row):
            if isinstance(item, float):
                formatted_row.append(f"{item:.6f}".ljust(col_widths[i]))
            else:
                formatted_row.append(str(item).ljust(col_widths[i]))
        print("".join(formatted_row))

def main():
    print("\n" + "="*70)
    print("GPU PROGRAMMING AND HARDWARE ACCELERATION DEMO")
    print("="*70 + "\n")
    
    # Vector addition benchmark
    vector_sizes = [10_000_000, 50_000_000, 100_000_000, 200_000_000]
    vector_results = vector_addition_benchmark(vector_sizes)
    
    # Matrix multiplication benchmark
    matrix_sizes = [1000, 2000, 4000, 6000]
    matrix_results = matrix_multiplication_benchmark(matrix_sizes)
    
    # Neural network benchmark
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    nn_results = neural_network_benchmark(batch_sizes)
    
    # Print summary tables
    formatted_vector_results = []
    for size, cpu_time, gpu_time, speedup in vector_results:
        formatted_vector_results.append((f"{size:,}", cpu_time, gpu_time, speedup))
    
    print_summary_table(
        "VECTOR ADDITION SUMMARY",
        ["Vector Size", "CPU Time (s)", "GPU Time (s)", "Speedup"],
        formatted_vector_results
    )
    
    formatted_matrix_results = []
    for size, cpu_time, gpu_time, speedup in matrix_results:
        formatted_matrix_results.append((f"{size}x{size}", cpu_time, gpu_time, speedup))
    
    print_summary_table(
        "MATRIX MULTIPLICATION SUMMARY",
        ["Matrix Size", "CPU Time (s)", "GPU Time (s)", "Speedup"],
        formatted_matrix_results
    )
    
    formatted_nn_results = []
    for batch_size, cpu_time, gpu_time, speedup in nn_results:
        formatted_nn_results.append((batch_size, cpu_time, gpu_time, speedup))
    
    print_summary_table(
        "NEURAL NETWORK SUMMARY",
        ["Batch Size", "CPU Time (s)", "GPU Time (s)", "Speedup"],
        formatted_nn_results
    )
    
    print_separator()
    print("Demo completed!")
    if device.type == "cuda":
        print("You have a GPU available, and should see significant speedup in the benchmarks.")
        avg_speedup = sum(s for _, _, _, s in matrix_results) / len(matrix_results)
        print(f"Average speedup for matrix multiplication: {avg_speedup:.2f}x")
    else:
        print("You're running on CPU only, so both implementations have similar performance.")
        print("To see the benefits of GPU acceleration, run this on a system with a CUDA-capable GPU.")
    print_separator()

if __name__ == "__main__":
    main()
