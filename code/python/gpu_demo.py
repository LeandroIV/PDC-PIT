"""
GPU Programming and Hardware Acceleration Demo

This script demonstrates the performance difference between CPU and GPU
for matrix operations and neural network training.

Requirements:
- Python 3.6+
- PyTorch
- Matplotlib
- NumPy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import FuncAnimation

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If GPU is available, print some information about it
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("No GPU available. Running on CPU only.")
    print("Note: Performance comparison will still run, but both will use CPU.")

# Create output directory for plots
os.makedirs("output", exist_ok=True)

def matrix_multiplication_benchmark(sizes):
    """Benchmark matrix multiplication on CPU and GPU."""
    cpu_times = []
    gpu_times = []
    
    for size in sizes:
        print(f"Testing matrix multiplication with size {size}x{size}...")
        
        # Create random matrices
        matrix_a = torch.rand(size, size)
        matrix_b = torch.rand(size, size)
        
        # CPU benchmark
        start_time = time.time()
        result_cpu = torch.matmul(matrix_a, matrix_b)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
        print(f"  CPU time: {cpu_time:.4f} seconds")
        
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
            gpu_times.append(gpu_time)
            print(f"  GPU time: {gpu_time:.4f} seconds")
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            # If no GPU, just duplicate CPU time for plotting
            gpu_times.append(cpu_time)
    
    return cpu_times, gpu_times

def neural_network_benchmark(batch_sizes):
    """Benchmark neural network training on CPU and GPU."""
    cpu_times = []
    gpu_times = []
    
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
        cpu_times.append(cpu_time)
        print(f"  CPU time: {cpu_time:.4f} seconds")
        
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
            gpu_times.append(gpu_time)
            print(f"  GPU time: {gpu_time:.4f} seconds")
            print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            # If no GPU, just duplicate CPU time for plotting
            gpu_times.append(cpu_time)
    
    return cpu_times, gpu_times

def plot_results(sizes, cpu_times, gpu_times, title, xlabel, filename):
    """Plot the benchmark results."""
    plt.figure(figsize=(10, 6))
    
    # Convert to milliseconds for better readability
    cpu_times_ms = [t * 1000 for t in cpu_times]
    gpu_times_ms = [t * 1000 for t in gpu_times]
    
    # Calculate speedup
    speedup = [cpu / gpu if gpu > 0 else 1 for cpu, gpu in zip(cpu_times, gpu_times)]
    
    # Bar chart
    bar_width = 0.35
    x = np.arange(len(sizes))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot bars for CPU and GPU times
    ax1.bar(x - bar_width/2, cpu_times_ms, bar_width, label='CPU', color='#3498db')
    ax1.bar(x + bar_width/2, gpu_times_ms, bar_width, label='GPU', color='#2ecc71')
    
    # Set labels and title for time axis
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Time (milliseconds)')
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    
    # Create second y-axis for speedup
    ax2 = ax1.twinx()
    ax2.plot(x, speedup, 'ro-', label='Speedup')
    ax2.set_ylabel('Speedup (CPU time / GPU time)')
    ax2.grid(False)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Add speedup values as text
    for i, s in enumerate(speedup):
        ax2.annotate(f'{s:.1f}x', xy=(i, s), xytext=(0, 5), 
                     textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join("output", filename))
    print(f"Plot saved to output/{filename}")

def create_animation():
    """Create an animation showing parallel vs sequential processing."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('CPU vs GPU Processing Visualization', fontsize=16)
    
    # Set up the axes
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 10)
    ax1.set_title('CPU: Sequential Processing')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Tasks')
    ax1.set_yticks(range(1, 11))
    
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 10)
    ax2.set_title('GPU: Parallel Processing')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Tasks')
    ax2.set_yticks(range(1, 11))
    
    # Initialize the lines
    cpu_lines = [ax1.plot([], [], 'o-', lw=2)[0] for _ in range(10)]
    gpu_lines = [ax2.plot([], [], 'o-', lw=2)[0] for _ in range(10)]
    
    # Colors for the lines
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    for i, line in enumerate(cpu_lines):
        line.set_color(colors[i])
    for i, line in enumerate(gpu_lines):
        line.set_color(colors[i])
    
    def init():
        for line in cpu_lines + gpu_lines:
            line.set_data([], [])
        return cpu_lines + gpu_lines
    
    def animate(frame):
        # CPU: Sequential processing
        for i, line in enumerate(cpu_lines):
            if frame >= i * 10:
                progress = min(frame - i * 10, 10)
                line.set_data([0, progress], [i + 1, i + 1])
            else:
                line.set_data([], [])
        
        # GPU: Parallel processing
        for i, line in enumerate(gpu_lines):
            progress = min(frame, 10)
            line.set_data([0, progress], [i + 1, i + 1])
        
        return cpu_lines + gpu_lines
    
    ani = FuncAnimation(fig, animate, frames=100, init_func=init, blit=True, interval=50)
    ani.save(os.path.join("output", "cpu_vs_gpu_animation.gif"), writer='pillow', fps=20)
    print("Animation saved to output/cpu_vs_gpu_animation.gif")

def main():
    print("\n" + "="*50)
    print("GPU Programming and Hardware Acceleration Demo")
    print("="*50 + "\n")
    
    # Matrix multiplication benchmark
    print("\nRunning matrix multiplication benchmark...")
    matrix_sizes = [500, 1000, 2000, 3000]
    cpu_times_mm, gpu_times_mm = matrix_multiplication_benchmark(matrix_sizes)
    
    # Neural network benchmark
    print("\nRunning neural network benchmark...")
    batch_sizes = [64, 128, 256, 512, 1024]
    cpu_times_nn, gpu_times_nn = neural_network_benchmark(batch_sizes)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(
        matrix_sizes, 
        cpu_times_mm, 
        gpu_times_mm, 
        "Matrix Multiplication: CPU vs GPU Performance", 
        "Matrix Size", 
        "matrix_multiplication_benchmark.png"
    )
    
    plot_results(
        batch_sizes, 
        cpu_times_nn, 
        gpu_times_nn, 
        "Neural Network Training: CPU vs GPU Performance", 
        "Batch Size", 
        "neural_network_benchmark.png"
    )
    
    # Create animation
    print("\nCreating animation...")
    create_animation()
    
    print("\nDemo completed! Check the 'output' directory for results.")
    print("\nIf you have a GPU, you should see significant speedup in the benchmarks.")
    print("If you're running on CPU only, both implementations will have similar performance.")

if __name__ == "__main__":
    main()
