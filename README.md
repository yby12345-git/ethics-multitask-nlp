# Ethics Multi-task NLP

## Overview

This repository contains the official implementation of a transformer-based multi-task learning framework for ethical text classification and model interpretability analysis.

The proposed framework jointly learns multiple ethical reasoning tasks derived from the ETHICS benchmark while providing interpretable predictions through attention visualization and SHAP-based feature attribution.

The supported ethical reasoning tasks include:

- Commonsense Morality
- Justice
- Deontology
- Virtue Ethics
- Utilitarianism

The framework integrates:

- Multi-task learning
- Transformer-based language representations (BERT)
- Shared feature extraction
- Task-specific classification heads
- Attention visualization
- SHAP interpretability analysis

---

# Repository Structure

```text
.
├── data/
│   ├── README_DATASET.txt
│   ├── ethics_dataset_train.csv
│   ├── ethics_dataset_val.csv
│   └── ...
│
├── scripts/
│   ├── train_multitask_bert.py
│   ├── train_single_task_bert.py
│   ├── evaluate_model.py
│   ├── plot_figure5_clean.py
│   └── ...
│
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── ...
│
├── results/
│
├── models/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Dataset

The experiments are based on the publicly available **ETHICS benchmark** introduced by Hendrycks et al.

Official repository:

https://github.com/hendrycks/ethics

The dataset used in this project was reconstructed from the publicly available benchmark resources, annotation schema, and task definitions provided by the original authors.

The reconstructed dataset preserves:

- Ethical reasoning categories
- Annotation schema
- Label definitions
- Evaluation protocol

The original ETHICS dataset is **not redistributed** in this repository due to licensing and third-party data considerations.

Additional dataset information is available in

```text
data/README_DATASET.txt
```

---

# Requirements

The experiments were conducted using the following software environment.

| Package | Version |
|----------|---------|
| Python | 3.10 |
| PyTorch | 2.x |
| Transformers | 4.x |
| Datasets | latest |
| NumPy | latest |
| Pandas | latest |
| Scikit-learn | latest |
| Matplotlib | latest |
| SHAP | latest |

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# Training

Train the proposed multi-task BERT model

```bash
python train_multitask_bert.py
```

Train an individual single-task baseline

```bash
python train_single_task_bert.py
```

---

# Evaluation

Evaluate the trained model

```bash
python evaluate_model.py
```

---

# Figure Generation

Generate all figures reported in the manuscript

```bash
python plot_figure5_clean.py
```

The generated figures are automatically saved in

```text
outputs/
```

including

- PNG
- PDF
- SVG

formats.

---

# Methodology

The proposed framework consists of the following stages.

1. Text preprocessing
2. Tokenization using BERT tokenizer
3. Shared transformer encoding
4. Task-specific classification heads
5. Multi-task optimization
6. Performance evaluation
7. Attention visualization
8. SHAP-based interpretability analysis

Performance is evaluated using

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

# Reproducibility

To improve experimental reproducibility,

- fixed random seeds are used throughout training;
- standardized train/validation/test splits are adopted;
- all figures are generated directly from the experimental outputs;
- complete source code is provided;
- dependency versions are documented.

The repository contains

- source code
- preprocessing scripts
- training scripts
- evaluation scripts
- figure generation scripts
- dependency specifications

---

# Citation

If you use this repository, please cite both this repository and the original ETHICS benchmark.

```bibtex
@inproceedings{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Hendrycks, Dan and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

---

# License

This project is released under the **MIT License**.

See the LICENSE file for details.

---

# Acknowledgments

This work builds upon the ETHICS benchmark developed by Hendrycks et al.

The authors gratefully acknowledge the creators of the ETHICS benchmark for making the dataset and benchmark publicly available to the research community.

---

# Contact

For questions regarding this repository, please open an issue on GitHub or contact the corresponding author listed in the associated manuscript.
