"""
GPU-Accelerated Image Processing Demo

This script demonstrates GPU acceleration for image processing tasks.
It applies various filters and transformations to images and compares
the performance between CPU and GPU implementations.

Requirements:
- Python 3.6+
- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torchvision import transforms
from PIL import Image
import urllib.request

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs("output", exist_ok=True)

def download_sample_image():
    """Download a sample image if not already present."""
    image_path = "sample_image.jpg"
    if not os.path.exists(image_path):
        print("Downloading sample image...")
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/1280px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
        urllib.request.urlretrieve(url, image_path)
        print(f"Image downloaded to {image_path}")
    return image_path

def load_image(path):
    """Load an image and convert to PyTorch tensor."""
    # Read image with OpenCV
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
    
    return img, img_tensor

def gaussian_blur_cpu(img_tensor, kernel_size=15, sigma=5.0):
    """Apply Gaussian blur using CPU."""
    # Convert to numpy for OpenCV
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    
    start_time = time.time()
    blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
    cpu_time = time.time() - start_time
    
    # Convert back to tensor
    result = torch.from_numpy(blurred).float().permute(2, 0, 1) / 255.0
    
    return result, cpu_time

def gaussian_blur_gpu(img_tensor, kernel_size=15, sigma=5.0):
    """Apply Gaussian blur using GPU."""
    if device.type != "cuda":
        # Fall back to CPU if GPU not available
        return gaussian_blur_cpu(img_tensor, kernel_size, sigma)
    
    # Move tensor to GPU
    img_gpu = img_tensor.to(device)
    
    # Create Gaussian kernel
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d).to(device)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(3, 1, 1, 1)  # One kernel per channel
    
    # Prepare input for convolution
    img_gpu = img_gpu.unsqueeze(0)  # Add batch dimension
    
    # Apply convolution
    start_time = time.time()
    padding = kernel_size // 2
    blurred = torch.nn.functional.conv2d(
        img_gpu, kernel_2d, padding=padding, groups=3
    )
    torch.cuda.synchronize()  # Ensure GPU operations complete
    gpu_time = time.time() - start_time
    
    # Move result back to CPU
    result = blurred.squeeze(0).cpu()
    
    return result, gpu_time

def sobel_edge_detection_cpu(img_tensor):
    """Apply Sobel edge detection using CPU."""
    # Convert to numpy for OpenCV
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    start_time = time.time()
    
    # Apply Sobel operators
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cpu_time = time.time() - start_time
    
    # Convert back to tensor (single channel)
    result = torch.from_numpy(magnitude).float().unsqueeze(0) / 255.0
    
    return result, cpu_time

def sobel_edge_detection_gpu(img_tensor):
    """Apply Sobel edge detection using GPU."""
    if device.type != "cuda":
        # Fall back to CPU if GPU not available
        return sobel_edge_detection_cpu(img_tensor)
    
    # Move tensor to GPU
    img_gpu = img_tensor.to(device)
    
    # Convert to grayscale
    gray_weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    img_gpu = img_gpu.unsqueeze(0)  # Add batch dimension
    img_gray = torch.sum(img_gpu * gray_weights, dim=1, keepdim=True)
    
    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    start_time = time.time()
    
    # Apply convolution
    grad_x = torch.nn.functional.conv2d(img_gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(img_gray, sobel_y, padding=1)
    
    # Compute magnitude
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize
    magnitude = magnitude / magnitude.max()
    
    torch.cuda.synchronize()  # Ensure GPU operations complete
    gpu_time = time.time() - start_time
    
    # Move result back to CPU
    result = magnitude.squeeze(0).cpu()
    
    return result, gpu_time

def image_resize_cpu(img_tensor, scale_factor=0.5):
    """Resize image using CPU."""
    # Convert to numpy for OpenCV
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    
    start_time = time.time()
    
    # Resize image
    h, w = img_np.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    cpu_time = time.time() - start_time
    
    # Convert back to tensor
    result = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    
    return result, cpu_time

def image_resize_gpu(img_tensor, scale_factor=0.5):
    """Resize image using GPU."""
    if device.type != "cuda":
        # Fall back to CPU if GPU not available
        return image_resize_cpu(img_tensor, scale_factor)
    
    # Move tensor to GPU
    img_gpu = img_tensor.to(device)
    
    start_time = time.time()
    
    # Add batch dimension
    img_gpu = img_gpu.unsqueeze(0)
    
    # Resize using PyTorch's interpolate function
    resized = torch.nn.functional.interpolate(
        img_gpu, 
        scale_factor=scale_factor, 
        mode='bicubic',
        align_corners=False
    )
    
    torch.cuda.synchronize()  # Ensure GPU operations complete
    gpu_time = time.time() - start_time
    
    # Move result back to CPU
    result = resized.squeeze(0).cpu()
    
    return result, gpu_time

def display_results(original, cpu_result, gpu_result, operation, cpu_time, gpu_time):
    """Display original image and results with timing information."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy for display
    if original.dim() == 3:
        original_np = original.permute(1, 2, 0).numpy()
    else:
        original_np = original.squeeze().numpy()
        
    if cpu_result.dim() == 3:
        cpu_result_np = cpu_result.permute(1, 2, 0).numpy()
    else:
        cpu_result_np = cpu_result.squeeze().numpy()
        
    if gpu_result.dim() == 3:
        gpu_result_np = gpu_result.permute(1, 2, 0).numpy()
    else:
        gpu_result_np = gpu_result.squeeze().numpy()
    
    # Display images
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(cpu_result_np, cmap='gray' if cpu_result.dim() == 2 else None)
    axes[1].set_title(f"CPU: {cpu_time:.4f}s")
    axes[1].axis('off')
    
    axes[2].imshow(gpu_result_np, cmap='gray' if gpu_result.dim() == 2 else None)
    axes[2].set_title(f"GPU: {gpu_time:.4f}s")
    axes[2].axis('off')
    
    plt.suptitle(f"{operation} - Speedup: {cpu_time/gpu_time:.2f}x")
    plt.tight_layout()
    
    # Save figure
    filename = f"output/{operation.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Saved result to {filename}")
    
    return fig

def main():
    print("\n" + "="*50)
    print("GPU-Accelerated Image Processing Demo")
    print("="*50 + "\n")
    
    # Download and load sample image
    image_path = download_sample_image()
    img_np, img_tensor = load_image(image_path)
    
    print(f"Image loaded: {image_path}")
    print(f"Image shape: {img_np.shape}")
    
    # 1. Gaussian Blur
    print("\nApplying Gaussian Blur...")
    cpu_blur, cpu_time = gaussian_blur_cpu(img_tensor)
    gpu_blur, gpu_time = gaussian_blur_gpu(img_tensor)
    print(f"  CPU time: {cpu_time:.4f} seconds")
    print(f"  GPU time: {gpu_time:.4f} seconds")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    display_results(img_tensor, cpu_blur, gpu_blur, "Gaussian Blur", cpu_time, gpu_time)
    
    # 2. Sobel Edge Detection
    print("\nApplying Sobel Edge Detection...")
    cpu_edge, cpu_time = sobel_edge_detection_cpu(img_tensor)
    gpu_edge, gpu_time = sobel_edge_detection_gpu(img_tensor)
    print(f"  CPU time: {cpu_time:.4f} seconds")
    print(f"  GPU time: {gpu_time:.4f} seconds")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    display_results(img_tensor, cpu_edge, gpu_edge, "Edge Detection", cpu_time, gpu_time)
    
    # 3. Image Resizing
    print("\nPerforming Image Resizing...")
    cpu_resize, cpu_time = image_resize_cpu(img_tensor, scale_factor=0.25)
    gpu_resize, gpu_time = image_resize_gpu(img_tensor, scale_factor=0.25)
    print(f"  CPU time: {cpu_time:.4f} seconds")
    print(f"  GPU time: {gpu_time:.4f} seconds")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    display_results(img_tensor, cpu_resize, gpu_resize, "Image Resizing", cpu_time, gpu_time)
    
    print("\nDemo completed! Check the 'output' directory for results.")
    if device.type == "cuda":
        print("You have a GPU available, and should see significant speedup in the operations.")
    else:
        print("You're running on CPU only, so both implementations have similar performance.")
        print("To see the benefits of GPU acceleration, run this on a system with a CUDA-capable GPU.")

if __name__ == "__main__":
    main()
