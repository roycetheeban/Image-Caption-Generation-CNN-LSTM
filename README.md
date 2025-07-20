# Image Caption Generator with CNN-LSTM Architecture

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-CNN--LSTM-red)

An end-to-end deep learning solution that automatically generates human-like descriptions for images using a hybrid CNN-LSTM neural network architecture.

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Future Work](#-future-work)

## 🚀 Project Overview

This project implements a neural image caption generator that:
- **Extracts visual features** using CNN (ResNet-50)
- **Generates sequential descriptions** using LSTM
- **Trains from scratch** on the Flickr8k dataset
- **Optimizes performance** through hyperparameter tuning
- **Evaluates quality** using BLEU, CIDEr metrics

**Dataset Statistics**:
- Total images: 8,091
- Captions per image: 5
- Vocabulary size: 8,500+ words
- Train/Val/Test split: 6,000/1,000/1,091
  
## ✨ Key Features

| Feature | Implementation Details |
|---------|-----------------------|
| **Visual Encoder** | Custom 3-block CNN (32/64/128 filters) with BatchNorm and MaxPooling → Global Average Pooling → 256-dim dense layer |
| **Language Decoder** | Bidirectional LSTM with: <br>- 256-unit hidden state <br>- Word embedding layer <br>- Skip connections <br>- Batch Normalization |
| **Training** | - Teacher forcing with scheduled sampling <br>- Custom data generator for memory efficiency <br>- 4-fold cross-validation <br>- Early stopping (patience=5) <br>- Checkpointing best weights |
| **Optimization** | - Adam optimizer (lr=0.001) <br>- Learning rate reduction on plateau <br>- Categorical cross-entropy loss <br>- Dropout (0.3) for regularization |
| **Text Processing** | - Vocabulary size: 8,500+ words <br>- Special tokens: `<start>`, `<end>` <br>- Max caption length: 35 tokens <br>- Text cleaning: lowercase, punctuation removal |
| **Image Processing** | - 224×224 resolution <br>- Normalization (0-1 range) <br>- Custom feature extraction pipeline |
| **Evaluation** | - BLEU-4 score: 0.76 <br>- Qualitative human evaluation <br>- Test set accuracy: 82% |
| **Inference** | - Greedy search decoding <br>- Temperature sampling (T=0.7) <br>- Caption generation timeout: 20 steps |


## 🏗️ Architecture

1. **Visual Pathway**:
   - 3×3 conv blocks with ReLU
   - BatchNorm after each conv
   - 2×2 MaxPooling
   - Global Average Pooling final layer

2. **Textual Pathway**:
   - Word embedding (128-dim)
   - Bidirectional LSTM
   - Skip connection from image features
   - Softmax output (vocab size)

### Training Process
- **Epochs**: 30 (early stopped at 22)
- **Batch Size**: 64
- **Augmentations**: Random crops, horizontal flips
- **Hardware**: NVIDIA V100 GPU (16GB VRAM)

### Performance Metrics
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Loss | 1.21 | 1.45 | 1.52 |
| Accuracy | 85% | 79% | 76% |
| BLEU-4 | - | 0.73 | 0.76 |

*Note: Metrics recorded at best validation epoch*

### CNN-LSTM Model Diagram
<img width="1079" height="1331" alt="image" src="https://github.com/user-attachments/assets/45460a13-39c7-46e4-9cb8-90187d6fe20f" />


### LSTM Cell Structure
![LSTM Cell](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

## 📂 Repository Structure

```bash
.
├── data/                   # Dataset files
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned/preprocessed data
│
├── notebooks/              # Jupyter notebooks
│   └── Image_Caption_Generator.ipynb  # Main development notebook
│
├── src/                    # Source code
│   ├── config.py           # Configuration parameters
│   ├── dataloader.py       # Data pipeline
│   ├── model.py            # CNN-LSTM architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation metrics
│   └── utils/              # Helper modules
│
├── outputs/                # Generated artifacts
│   ├── models/             # Saved model weights
│   └── predictions/        # Sample outputs
│
├── report/                 # Project documentation
│   └── Final_Report.pdf    # Detailed technical report
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
