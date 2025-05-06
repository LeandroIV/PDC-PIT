# GPU Programming Code Examples

This document provides practical code examples of GPU programming using different frameworks and for various applications.

## Table of Contents
1. [CUDA Examples](#cuda-examples)
2. [OpenCL Examples](#opencl-examples)
3. [Python with GPU Acceleration](#python-gpu)
4. [Deep Learning Examples](#deep-learning)
5. [Performance Comparison](#performance)

<a name="cuda-examples"></a>
## 1. CUDA Examples

### 1.1 Vector Addition

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    // Vector size
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < n; i++) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            fprintf(stderr, "Verification failed at element %d!\n", i);
            break;
        }
    }
    
    printf("Vector addition completed successfully\n");
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

### 1.2 Matrix Multiplication

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel for matrix multiplication
__global__ void matrixMul(float *a, float *b, float *c, int width) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute if within matrix bounds
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    // Matrix dimensions
    int width = 1024;
    size_t bytes = width * width * sizeof(float);
    
    // Host matrices
    float *h_a, *h_b, *h_c;
    
    // Device matrices
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host matrices
    for (int i = 0; i < width * width; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (width + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    printf("Matrix multiplication completed successfully\n");
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

<a name="opencl-examples"></a>
## 2. OpenCL Examples

### 2.1 Vector Addition in OpenCL

```c
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// OpenCL kernel source
const char *kernelSource = 
"__kernel void vectorAdd(__global const float *a,      \n" \
"                        __global const float *b,      \n" \
"                        __global float *c,            \n" \
"                        const int n)                  \n" \
"{                                                     \n" \
"    // Get global thread ID                           \n" \
"    int id = get_global_id(0);                        \n" \
"                                                      \n" \
"    // Boundary check                                 \n" \
"    if (id < n) {                                     \n" \
"        c[id] = a[id] + b[id];                        \n" \
"    }                                                 \n" \
"}                                                     \n";

int main() {
    // Vector size
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Host vectors
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // OpenCL variables
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_a, d_b, d_c;
    
    // Get platform and device
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    
    // Create and build program
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    // Create kernel
    kernel = clCreateKernel(program, "vectorAdd", NULL);
    
    // Create device buffers
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    // Copy data to device
    clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(kernel, 3, sizeof(int), &n);
    
    // Execute kernel
    size_t globalSize = n;
    size_t localSize = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    
    // Copy result back to host
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
    
    // Verify result
    for (int i = 0; i < n; i++) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            fprintf(stderr, "Verification failed at element %d!\n", i);
            break;
        }
    }
    
    printf("Vector addition completed successfully\n");
    
    // Clean up
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

<a name="python-gpu"></a>
## 3. Python with GPU Acceleration

### 3.1 NumPy with CuPy

```python
import numpy as np
import cupy as cp
import time

# Vector size
n = 10000000

# CPU implementation
def vector_add_cpu(a, b):
    return a + b

# Create random vectors
a_cpu = np.random.random(n).astype(np.float32)
b_cpu = np.random.random(n).astype(np.float32)

# CPU timing
start_time = time.time()
c_cpu = vector_add_cpu(a_cpu, b_cpu)
cpu_time = time.time() - start_time
print(f"CPU time: {cpu_time:.6f} seconds")

# GPU implementation
# Transfer data to GPU
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)

# GPU timing
start_time = time.time()
c_gpu = a_gpu + b_gpu
cp.cuda.Stream.null.synchronize()  # Ensure GPU operations complete
gpu_time = time.time() - start_time
print(f"GPU time: {gpu_time:.6f} seconds")

# Verify results
c_from_gpu = cp.asnumpy(c_gpu)
np.testing.assert_allclose(c_cpu, c_from_gpu, rtol=1e-5)
print(f"Results match! Speedup: {cpu_time/gpu_time:.2f}x")
```

<a name="deep-learning"></a>
## 4. Deep Learning Examples

### 4.1 PyTorch GPU Acceleration

```python
import torch
import time
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return self.log_softmax(x)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Initialize model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Train and test the model
start_time = time.time()
for epoch in range(1, 3):  # Just 2 epochs for demonstration
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")
```

<a name="performance"></a>
## 5. Performance Comparison

The following table shows typical performance improvements when moving from CPU to GPU implementation:

| Application | CPU Performance | GPU Performance | Speedup Factor |
|-------------|----------------|-----------------|----------------|
| Vector Addition (1M elements) | 5.2 ms | 0.3 ms | 17x |
| Matrix Multiplication (1024x1024) | 2.3 s | 45 ms | 51x |
| Neural Network Training (MNIST) | 45 min | 2 min | 22.5x |
| Image Processing (4K image) | 320 ms | 12 ms | 26.7x |
| Monte Carlo Simulation (1M paths) | 8.5 s | 180 ms | 47x |

Note: Performance numbers are representative and will vary based on specific hardware configurations.
