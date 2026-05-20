# Face Generation with GAN (CelebA)

This project implements a **Generative Adversarial Network (GAN)** from scratch using PyTorch to generate realistic human face images. The model is trained on the **CelebA** dataset, a large‑scale face attributes dataset containing over 200,000 celebrity images.

![Sample Generated Faces](https://via.placeholder.com/400?text=Generated+Faces)  
*Example of generated faces after 5 epochs (you will see actual outputs when you run the code).*

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Data Preprocessing](#data-preprocessing)
  - [Generator Network](#generator-network)
  - [Discriminator Network](#discriminator-network)
  - [Training Loop](#training-loop)
- [Usage](#usage)
- [Results](#results)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Generative Adversarial Networks consist of two competing neural networks:

- **Generator** – creates fake images from random noise.
- **Discriminator** – tries to distinguish real images from fake ones.

Through adversarial training, the generator learns to produce increasingly realistic images, while the discriminator becomes better at detecting fakes. After sufficient training, the generator can synthesize new faces that resemble the training distribution.

This implementation uses:
- **Fully connected (dense) layers** for both networks (a simple MLP architecture).
- **Binary cross‑entropy loss** (`BCELoss`).
- **Adam optimiser** with learning rate `0.0002` and beta values `(0.5, 0.999)`.
- Training on **64×64 RGB** images.

---

## Requirements

- Python 3.8+
- PyTorch (`torch`, `torchvision`)
- Pillow (PIL)
- Matplotlib
- NumPy

Install the required packages with:

```bash
pip install torch torchvision pillow matplotlib numpy
