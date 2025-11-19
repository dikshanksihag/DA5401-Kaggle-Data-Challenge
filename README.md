# DA5401 – Metric Learning for Conversational AI  
### End-Semester Data Challenge (2025)

Author: Dikshank Sihag

## Overview

This repository implements a full end-to-end solution for the DA5401 2025 Data Challenge.  
The task is to predict a **fitness/alignment score (0–10)** between:

- an *Evaluation Metric Definition* (provided as embeddings), and  
- a *Prompt–Response pair* (raw text)

The fitness score is produced by an LLM judge and represents how well the prompt-response pair aligns with the metric.

To solve this, we build a **metric-learning-style regression pipeline** using:

- SBERT embeddings for prompt & response
- Provided metric-definition embeddings
- A 3-layer MLP with residual connections
- Heavy synthetic augmentation to fix score imbalance
- Consistent training ↔ inference embedding logic

The final model produces strong RMSE performance even under heavy class imbalance.

---

## Repository Structure

```
.
├── train_nn_regressor_singleval.py      # Training script for the 2304-D MLP model
├── make_submission_single_nn.py         # Kaggle submission generator
├── data_preprocessing.ipynb             # (Optional) augmentation + inspection notebook
├── X_negative_augmented.npy             # Augmented training features
├── y_negative_augmented.npy             # Augmented labels
├── nn_single/
│   ├── model_final.pth                  # Trained model
│   └── single_val_metrics.json          # Validation RMSE
└── README.md
```

---

## Problem Description

The challenge requires learning the similarity relationship between:

1. A **metric definition** (text embedding only)
2. A **prompt + system prompt + response**
3. A **fitness score** in the [0, 10] range

The dataset is multilingual, containing Tamil, Hindi, Bengali, Assamese, Bodo, Sindhi, and English.

A major issue is **extreme score imbalance**: most training scores lie in the 7–10 region.  
To avoid model collapse, extensive augmentation is introduced.

---

## Data Inputs

### Provided by competition:

- `train_data.json`
- `test_data.json`
- `metric_names.json`
- `metric_name_embeddings.npy`

### Produced in this repository:

- Prompt embeddings (SBERT)
- Response embeddings (SBERT)
- Concatenated train matrices (2304-D)
- Synthetic augmentation for balanced training

---

## Model Architecture (MLP Regressor)

The model receives a **2304-dimensional vector**:

- 768-dim prompt embedding  
- 768-dim response embedding  
- 768-dim metric embedding  

Architecture:

- Hidden layers: **1024 → 512 → 256**
- Each block: `Linear → BatchNorm → ReLU → Dropout(0.2)`
- **Residual connection** from input → last hidden layer
- Output head: `LayerNorm → Linear → Score`
- Predictions **clamped to [0, 10]**

This model is efficient and well-suited for semantic similarity learning.

---

## Training Pipeline

Run training:

```bash
python train_nn_regressor_singleval.py
```

Training details:

- Train/val split: **validation = 3000**
- RMSE evaluation
- AdamW optimizer
- ReduceLROnPlateau learning-rate scheduling
- Early stopping (patience = 12)
- Optional label noise (`--label_noise_std`)
- Saves:
  - `nn_single/model_final.pth`
  - `nn_single/single_val_metrics.json`

---

## Data Augmentation Strategy

Because the dataset is extremely skewed, we generate:

- **~12,000 synthetic low-score samples (0–4)**
- **~5,000 synthetic mid-score samples (4–7)**

Augmentations include:

1. **Shuffled prompt/response pairs**
2. **Embedding-space noise injection**
3. **Metric-swapping (semantically inconsistent pairs)**

This balancing dramatically improves model stability and convergence.

---

## Inference & Submission

To generate a Kaggle submission:

```bash
python make_submission_single_nn.py
```

The script:

1. SBERT-embeds prompt and response (same model as training)
2. Loads metric embeddings
3. Builds `[PROMPT | RESPONSE | METRIC]` concatenated vectors
4. Loads `model_final.pth`
5. Predicts and **clamps to [0,10]**
6. Writes `submission_nn.csv` in the required format

---

## Requirements

```
torch
numpy
pandas
tqdm
sentence-transformers
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Key Design Principles

- Train ↔ Inference symmetry (zero embedding mismatch)
- Balanced score distribution through augmentation
- Simple but strong residual MLP regressor
- Fully reproducible with fixed seeds
- No metric-definition leakage (only embeddings used)

---

## Final Notes

This repository provides a robust, end-to-end solution for the DA5401 metric-learning task, from raw data to Kaggle submission.  
Everything is reproducible and structured for clarity and performance.
just ask.

