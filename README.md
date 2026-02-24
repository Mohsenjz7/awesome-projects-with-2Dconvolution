# 2D Convolution with SIMD (AVX2): Image Processing and Pattern Recognition

This project demonstrates the performance optimization of 2D Convolution using SIMD instructions (`AVX2` and `FMA`) on the CPU. The core convolution operation is implemented in both standard C and optimized Assembly (AVX2), allowing for a direct performance comparison. This foundational convolution logic is then applied to three main domains: image blurring, pattern (square) detection, and handwritten digit recognition (MNIST).

---

##  Features & Project Structure

The project is divided into three main applications:

### 1. Image Blurring & Performance Benchmarking (`pictureProcessing.c`)

* Implements a standard blur filter using a convolution kernel.
* Runs the same algorithm using both standard C and AVX2 Assembly.
* Measures and compares execution times to demonstrate the **Speedup** gained by utilizing CPU vector instructions.

### 2. Pattern (Square) Detection (`patternDetection.c`)

* Utilizes a Laplacian kernel for edge detection.
* Applies a custom `detect_square` convolution function to identify square-like patterns within an input image.

### 3. Handwritten Digit Recognition (`numberDetector.c`)

* Implements a basic Convolutional Neural Network (CNN) structure for digit classification using the MNIST dataset.
* Processes over 100 sequential images strictly on the CPU.
* Achieves an accuracy of **~80%** in recognizing handwritten digits.

---

##  Compilation & Execution

All programs are designed to run in a Linux environment (or WSL) and require the `GCC` compiler with AVX2 and FMA extensions enabled.

### 1. Running the Blur Filter (Benchmarking)

To compile and execute the image blurring program and see the performance difference:

```bash
gcc pictureProcessing.c -o my_app -mavx2 -mfma -O3 -lm
./my_app
```

### 2. Running Pattern (Square) Detection

To compile and execute the square detection script using the Laplacian kernel:

```bash
gcc patternDetection.c -o my_app -mavx2 -mfma -O3 -lm
./my_app
```

### 3. Running Handwritten Digit Recognition (MNIST)

To compile and evaluate the digit detector against the dataset:

```bash
gcc numberDetector.c -o my_cnn -mavx2 -mfma -O3 -lm
./my_cnn
```

---

##  Benchmark Results

### C vs. AVX2 Assembly (Image Blurring)

Based on direct execution on the CPU, utilizing AVX2 SIMD instructions yielded a significant performance boost over the standard C implementation:

| Implementation    | Average Execution Time | Speedup         |
| ----------------- | ---------------------- | --------------- |
| **Standard C**    | 0.067686 seconds       | 1.0x (Baseline) |
| **AVX2 Assembly** | 0.012710 seconds       | **~5.3x**       |

> **Conclusion:** The AVX2 implementation successfully blurred the image while being over 5 times faster than the standard, unoptimized C code.

### Digit Recognition (MNIST)

* **Hardware:** CPU-only execution.
* **Dataset:** Over 100 sequential test images.
* **Accuracy:** ~80%.
