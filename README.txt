Deep Learning Model Optimization for EEG Classification (BEED Dataset)
Project Overview

This project focuses on building and optimizing a deep learning model for EEG signal classification using the BEED (Bangalore EEG Epilepsy Dataset).
The primary goal was not only to achieve high classification accuracy, but also to explore hardware-aware optimization techniques to improve training speed, reduce memory usage, and enhance inference efficiency.

The project includes:
Baseline CNN training on CPU
Baseline CNN training on GPU (CUDA-enabled)
Magnitude-based unstructured pruning
Post-training FP16 quantization
Performance benchmarking (accuracy, latency, memory, parameter count)

All experiments were conducted using PyTorch, with GPU experiments performed on a cloud RTX 5090.

Dataset

Source:
BEED – Bangalore EEG Epilepsy Dataset
https://www.kaggle.com/datasets/varunrajput/beed-bangalore-eeg-epilepsy-dataset

Dataset Description
Total samples: 8,000
Features: 16 EEG-derived numerical features (X1–X16)
Target: 1 label column (4 classes: 0, 1, 2, 3)
Task: 4-class classification problem

Each row represents a processed EEG sample with extracted statistical and signal-based features. 
The objective is to classify each sample into one of four neurological categories.

Preprocessing Steps
80/20 train-test split
Standardization using StandardScaler (fit on training data only)
Reshaping to (batch_size, 1, 16) for Conv1D compatibility
Batch size = 64

Model Architecture
Baseline CNN:
Conv1D Layer 1: 64 filters, kernel size = 5, ReLU
Conv1D Layer 2: 64 filters, kernel size = 5, ReLU
Flatten
Fully connected layer
Output layer (4 neurons)

Loss: CrossEntropyLoss
Optimizer: Adam

Optimization Techniques Implemented
1. GPU Acceleration (CUDA)
Baseline model trained on both CPU and GPU to compare:
Training time
Memory usage
Inference latency

2. Magnitude-Based Pruning
20% unstructured pruning
Applied to Conv1D layers only
One-shot pruning
10-epoch fine-tuning to regain performance
Measured:
  Test accuracy
  Nonzero parameter count
  GPU latency
  GPU memory usage

3. Post-Training FP16 Quantization
Applied to Conv1D + Linear layers
20% of training data used for calibration
Measured:
  Accuracy drop (tolerance ≤ 2.5%)
  GPU memory usage
  Inference latency per batch and per sample
  Parameter count


Repository Structure
/cpucnn.py
Trains the baseline CNN entirely on CPU.
Measures:
  Final train/test accuracy
  Total training time
  Peak CPU memory usage

/cnngpu.py
Trains the baseline CNN on GPU using PyTorch CUDA tensors.
Measures:
  Final train/test accuracy
  Total training time
  Peak GPU memory usage
  Average GPU inference latency
  Total parameter count

/beed_pruning.py
Implements magnitude-based unstructured pruning.
Features:
  20% pruning applied to Conv1D layers
  One-shot pruning
  10-epoch fine-tuning
  Reports:
    Test accuracy after retraining
    Nonzero parameter count
    GPU memory usage
    Inference latency

/fp16_quantize_beed.py
Applies post-training FP16 quantization to the trained model.
Reports:
  Test accuracy
  Accuracy drop vs FP32
  GPU memory usage
  Inference latency per batch and per sample
  Total and nonzero parameters

/results.txt
Contains the recorded metrics from:
CPU baseline
GPU baseline
Pruned model
Quantized model
Includes training time, memory usage, accuracy, latency, and parameter counts.

/conclusion.txt
Summarizes key findings from the experiments, including:
CPU vs GPU performance differences
Impact of pruning on parameter count and latency
Effect of FP16 quantization on memory usage and accuracy
Overall tradeoffs between model size, speed, and performance
