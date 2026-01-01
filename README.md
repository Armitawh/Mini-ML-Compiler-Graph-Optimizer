# Mini ML Compiler: Graph Optimizer

## About
This project demonstrates a **mini ML compiler** built in Python that performs **graph-level optimizations** on PyTorch models. The optimizations include:

- Operator fusion (Conv + ReLU)
- Constant folding
- Graph simplification

The project benchmarks **inference time before and after optimization** and visualizes computation graphs using NetworkX and Matplotlib.

---

## Features

- Train a simple CNN on MNIST dataset
- Build a **computation graph** of the model
- Apply **compiler-style optimizations**
- Compare **original vs optimized graph**
- Measure **inference performance improvement**
- Visualize the graphs side by side in a single image

