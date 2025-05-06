# GPU Programming and Hardware Acceleration Report

This repository contains materials for a comprehensive report on real-world applications of GPU programming and hardware acceleration. The report includes detailed information, visual assets, code examples, and interactive demonstrations.

## Directory Structure

The repository is organized into the following directories:

```
PIT/
├── report/              # Report documents
├── presentation/        # Presentation materials
├── code/                # Code examples
│   ├── python/          # Python examples
│   └── cuda/            # CUDA examples
├── visuals/             # Visual assets
├── demo/                # Interactive demos
└── README.md            # This file
```

## Contents

1. **Report Documents** (`report/`)
   - `GPU_Programming_Report.md` - Comprehensive report covering real-world GPU applications

2. **Presentation Materials** (`presentation/`)
   - `GPU_Programming_Presentation.md` - Slides for presenting the key points of the report

3. **Code Examples** (`code/`)
   - `GPU_Programming_Code_Examples.md` - Sample code snippets demonstrating GPU programming concepts

   **Python Examples** (`code/python/`)
   - `simple_gpu_demo.py` - Python script demonstrating basic GPU acceleration
   - `gpu_demo.py` - Comprehensive Python demo with visualizations
   - `image_processing_demo.py` - GPU-accelerated image processing examples

   **CUDA Examples** (`code/cuda/`)
   - `cuda_vector_add.cu` - Native CUDA example in C

4. **Visual Assets** (`visuals/`)
   - `GPU_Programming_Visual_Assets.md` - Descriptions and links to visual assets for the report

5. **Interactive Demo** (`demo/`)
   - `GPU_Programming_Interactive_Demo.html` - Interactive web-based demonstration of GPU vs CPU processing

## How to Use These Materials

### Reading the Report

The main report document (`report/GPU_Programming_Report.md`) provides a comprehensive overview of GPU programming and its applications. It covers:

- Introduction to GPU computing
- GPU architecture
- Real-world applications in various industries
- Performance comparisons
- Programming models
- Future trends

### Presenting the Material

The presentation file (`presentation/GPU_Programming_Presentation.md`) contains slide content that can be converted to a PowerPoint or other presentation format. Each slide includes:

- Key points to discuss
- Suggested visuals to include
- Code examples where relevant

### Exploring Code Examples

The code examples file (`code/GPU_Programming_Code_Examples.md`) contains sample code in various languages and frameworks:

- CUDA examples for direct GPU programming
- OpenCL examples for cross-platform GPU programming
- Python examples using GPU acceleration libraries
- Deep learning examples with GPU acceleration

#### Python Examples

The Python examples in the `code/python/` directory demonstrate GPU acceleration for various tasks:

- `simple_gpu_demo.py`: Basic vector operations and neural network computations
- `gpu_demo.py`: Comprehensive benchmarks with visualizations
- `image_processing_demo.py`: Image processing tasks like blurring and edge detection

#### CUDA Examples

The CUDA examples in the `code/cuda/` directory show how to program directly for NVIDIA GPUs:

- `cuda_vector_add.cu`: Vector addition implementation in C/CUDA

### Using Visual Assets

The visual assets file (`visuals/GPU_Programming_Visual_Assets.md`) provides:

- Links to diagrams illustrating GPU architecture
- Performance comparison charts
- Application-specific visualizations
- Hardware images
- Instructions for creating custom visuals

### Interactive Demonstration

The interactive demo (`demo/GPU_Programming_Interactive_Demo.html`) is a web-based visualization that demonstrates:

- The difference between CPU (sequential) and GPU (parallel) processing
- Performance comparisons across different applications
- A simple CUDA programming example

To use the interactive demo:
1. Open the HTML file in a web browser
2. Click "Start Demo" to see the visualization
3. Click "Reset" to restart the demonstration

## Customizing the Report

These materials can be customized for your specific needs:

- Add your own code examples relevant to your field
- Include additional visual assets specific to your application
- Modify the performance comparisons to reflect your hardware
- Expand sections that are most relevant to your audience

## Running the Executable Demos

This repository includes several executable demos that showcase GPU acceleration in action:

### Simple GPU Demo (`code/python/simple_gpu_demo.py`)

This is the easiest example to run. It demonstrates GPU acceleration for:
- Vector addition
- Matrix multiplication
- Neural network operations

Run with:
```
cd code/python
python simple_gpu_demo.py
```

### Image Processing Demo (`code/python/image_processing_demo.py`)

Demonstrates GPU acceleration for image processing tasks:
- Gaussian blur
- Edge detection
- Image resizing

This demo downloads a sample image and shows visual results with timing information.

Run with:
```
cd code/python
python image_processing_demo.py
```

### CUDA Vector Addition (`code/cuda/cuda_vector_add.cu`)

A native CUDA example showing vector addition in C/CUDA.

Compile and run with:
```
cd code/cuda
nvcc -o cuda_vector_add cuda_vector_add.cu
./cuda_vector_add
```

## Prerequisites for Running Demos

### For Python Examples
- Python 3.6 or higher
- PyTorch (with CUDA support for GPU acceleration)
- NumPy
- Matplotlib (for visualization)
- OpenCV (for image processing demo)

### For CUDA Example
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C compiler (GCC, MSVC, etc.)

## Additional Resources

For more information on GPU programming, consider these resources:

1. NVIDIA CUDA Documentation: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
2. OpenCL Documentation: [https://www.khronos.org/opencl/](https://www.khronos.org/opencl/)
3. PyTorch GPU Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
4. TensorFlow GPU Guide: [https://www.tensorflow.org/guide/gpu](https://www.tensorflow.org/guide/gpu)
5. "Programming Massively Parallel Processors" by David B. Kirk and Wen-mei W. Hwu

## Viewing the Interactive Demo

To view the interactive demonstration:

1. Open the `demo/GPU_Programming_Interactive_Demo.html` file in a web browser
2. Click the "Start Demo" button to see the visualization of CPU vs GPU processing
3. The performance chart shows the speedup factors for different applications
4. The CUDA code example demonstrates how parallel operations are expressed in CUDA
