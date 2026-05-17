# Ethics Multi-task NLP
## Overview
This repository contains the implementation of a multi-task transformer framework for ethical text classification and interpretability analysis.
The framework is designed to jointly model multiple ethical reasoning dimensions derived from the ETHICS benchmark, including:
* Commonsense morality
* Justice
* Deontology
* Virtue ethics
* Utilitarianism
The proposed framework integrates:
* Multi-task learning
* Transformer-based representations (BERT)
* Attention visualization
* SHAP interpretability analysis
---
## Dataset Information
The experiments are based on the ETHICS benchmark proposed by Hendrycks et al.
Original ETHICS dataset repository:
https://github.com/hendrycks/ethics
The dataset used in this study was reconstructed from the publicly available benchmark resources, annotation schema, and task definitions provided in the official ETHICS repository and publication.
The reconstructed dataset preserves:
* Ethical task categories
* Annotation schema
* Label structure
Due to licensing and third-party data considerations, the original dataset is not redistributed in this repository.
Additional dataset description is provided in:
```text
data/README_DATASET.txt
```
---
## Code Information
Main project structure:
```text
scripts/                  training and evaluation scripts
models/                   saved model checkpoints
outputs/                  generated predictions and figures
results/                  experimental results
data/                     reconstructed dataset information
```
Important files:
```text
plot_figure5_clean.py     Figure generation script
requirements.txt          Python dependency list
README.md                 Project description
```
---
## Requirements
The project was implemented using:
* Python 3.10
* PyTorch
* Transformers
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* SHAP
Install dependencies using:
```bash
pip install -r requirements.txt
```
---
## Usage Instructions
Example training command:
```bash
python train_multitask_bert.py
```
Example evaluation command:
```bash
python evaluate_model.py
```
Example figure generation command:
```bash
python plot_figure5_clean.py
```
---
## Methodology
The proposed framework uses a shared BERT encoder combined with task-specific classification heads for multi-task ethical classification.
The workflow includes:
1. Text preprocessing and tokenization
2. Transformer encoding using BERT
3. Multi-task prediction across ethical dimensions
4. Evaluation using Accuracy, F1-score, and ROC-AUC
5. Interpretability analysis using Attention and SHAP
---
## Reproducibility
All experiments were conducted using fixed random seeds and standardized train/validation/test splits to improve reproducibility.
The repository includes:
* Source code
* Figure generation scripts
* Experimental outputs
* Dependency specifications
---
## Citation
If you use this repository or the ETHICS benchmark, please cite:

```bibtex
@inproceedings{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Hendrycks, Dan and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
---
## License
This repository is provided for academic research purposes only.
---
## Acknowledgments
This work is based on the ETHICS benchmark developed by Hendrycks et al.
