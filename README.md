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
- [Team](#-team)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

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
| **Visual Encoder** | ResNet-50 (pretrained on ImageNet) |
| **Language Decoder** | 2-layer LSTM with attention |
| **Training** | Teacher forcing, early stopping |
| **Optimization** | AdamW, learning rate scheduling |
| **Evaluation** | BLEU-1 to BLEU-4, CIDEr |
| **Inference** | Beam search (size=3) |

## 🏗️ Architecture

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
