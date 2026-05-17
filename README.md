# Ethics Multi-task NLP

## Overview

This repository contains the code and experimental resources for a multi-task BERT framework for ethical text classification and interpretability analysis. The framework jointly models multiple ethical reasoning dimensions derived from the ETHICS benchmark, including commonsense morality, justice, deontology, virtue ethics, and utilitarianism. The project also incorporates attention visualization and SHAP-based interpretability analysis for transparent ethical prediction.

---

## Dataset Information

This project is based on the ETHICS benchmark proposed by Hendrycks et al.

Original ETHICS benchmark repository:
https://github.com/hendrycks/ethics

The reconstructed dataset used in this study follows the original task definitions, annotation schema, and label structure described in the official benchmark and associated publication.

---

## Environment

- Python 3.10
- PyTorch
- Transformers
- NumPy
- Pandas
- Scikit-learn
- SHAP
- Matplotlib

---

## Project Structure

- `scripts/` : training and evaluation scripts
- `models/` : saved model checkpoints
- `outputs/` : generated prediction outputs and figures
- `results/` : experimental results and evaluation metrics
- `plot_figure5_clean.py` : script for generating Figure 5
- `Figure5_clean.png` / `Figure5_clean.pdf` : generated figure files

---

## Installation

Install required dependencies using:

pip install -r requirements.txt

---

## Usage

Train the multi-task BERT model:

python train.py

Evaluate the trained model:

python evaluate.py

Generate SHAP interpretability analysis:

python shap_analysis.py

---

## Requirements

The complete dependency list is provided in `requirements.txt`.

Example dependencies include:

- torch
- transformers
- numpy
- pandas
- scikit-learn
- shap
- matplotlib

---

## Methodology

The proposed framework employs a shared BERT-base transformer encoder with multi-task learning for ethical text classification. Each ethical dimension is modeled as an independent Bernoulli prediction task using sigmoid-based multi-label classification.

The framework integrates:

- Shared transformer representation learning
- Multi-task classification
- Attention-based interpretability
- SHAP token attribution analysis
- Ablation experiments across ethical tasks

Performance evaluation includes accuracy, precision, recall, macro F1-score, and ROC-AUC.

---

## Reproducibility

All experiments were conducted using Python 3.10 and PyTorch under CUDA acceleration. Hyperparameter settings, preprocessing procedures, evaluation metrics, and ablation configurations are described in the manuscript and supplementary materials.

---

## Citation

If you use this repository in your research, please cite the corresponding article:

Yang B, Zhang X, Li C, Wu Y. A multi-task learning framework for ethical classification using transformer-based representations.

---

## License

This repository is provided for academic research and educational purposes.

