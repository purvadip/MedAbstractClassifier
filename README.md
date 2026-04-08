# Medical Abstract Classifier

**Course:** SSIM916 — Problem Set #2: Using Text as Data  
**Student Number:** 750091800  
**Date:** 26 March 2026  
**Repository:** [github.com/purvadip/MedAbstractClassifier](https://github.com/purvadip/MedAbstractClassifier)

---

## Research Question

> *Can machine learning models trained on weakly-supervised medical abstracts reliably classify them by research type (Diagnosis, Treatment, Prevention), and which textual features emerge as the strongest interpretable predictors?*

---

## Project Overview

This project automates the categorisation of PubMed RCT abstracts into three research-type classes — **Diagnosis**, **Treatment**, and **Prevention** — using the PubMed 20k RCT dataset (Dernoncourt & Lee, 2017). Labels are derived through keyword-frequency heuristics rather than expert annotation, constituting a weak supervision framework. Two classical ML baselines (TF-IDF + Logistic Regression and Multinomial Naive Bayes) are evaluated and compared against a BioBERT external benchmark.

The motivation is practical: automated abstract triage can reduce manual screening workloads in evidence synthesis pipelines (e.g., Cochrane systematic reviews), where initial screening alone routinely exceeds 1,000 person-hours per review.

---

## Repository Structure

```
MedAbstractClassifier/
│
├── medical_classification.ipynb   # Main notebook (run this end-to-end)
├── biobert_colab.ipynb            # BioBERT fine-tuning (Google Colab / GPU)
├── create_notebooks.py            # Script that generates the .ipynb files
├── requirements.txt               # Python dependencies
├── split_indices.json             # Serialised train/test indices (auto-generated)
├── .gitignore
├── README.md
│
├── src/                           # Reusable Python modules
│   ├── __init__.py
│   ├── preprocess.py              # Data loading, labelling, keyword stripping
│   ├── baseline_model.py          # TF-IDF pipelines (LogReg + NB)
│   └── evaluate.py                # Metrics, confusion matrix, ROC curves
│
├── data/                          # Dataset split files (see §Dataset below)
│   ├── dev.csv
│   └── test.csv
│
└── results/                       # Output figures (auto-created at runtime)
    ├── class_distribution.png
    ├── wordclouds.png
    ├── abstract_length_dist.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_multinomial_nb.png
    ├── roc_curve_logistic_regression.png
    └── roc_curve_multinomial_nb.png
```

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | PubMed 20k RCT — [arxiv.org/abs/1710.06071](https://arxiv.org/abs/1710.06071) |
| **Size** | ~20,000 structured RCT abstracts |
| **Period** | MEDLINE 1976–2017 |
| **Download** | Place `train.csv`, `dev.csv`, `test.csv` in the `data/` folder |

### Label Assignment (Weak Supervision)

| Class | Signal Keywords |
|---|---|
| **Diagnosis** | sensitivity, imaging, screening, diagnostic, biopsy, accuracy |
| **Prevention** | vaccine, prophylaxis, immunisation, protective, incidence |
| **Treatment** | placebo, dose, intervention, trial, therapy, efficacy *(default)* |

Label noise: 11.5 % of abstracts had no matching keywords; 14.7 % had tied scores. Both groups defaulted to **Treatment**. Stratified undersampling brought all three classes to 6,667 abstracts each.

---

## Environment Setup

Requires **Python 3.8+**.

```bash
git clone https://github.com/purvadip/MedAbstractClassifier
cd MedAbstractClassifier
pip install -r requirements.txt
```

---

## Running the Baseline Notebook

```bash
jupyter notebook
```

Open `medical_classification.ipynb` and select **Cell → Run All**.  
The notebook runs the full preprocessing pipeline, trains both baseline models, and saves all figures to `results/`. No manual steps are required once the `data/` directory is populated.

Split indices are serialised to `split_indices.json` (72/8/20 train/validation/test split) to ensure deterministic evaluation across runs.

---

## Running BioBERT (Google Colab)

BioBERT fine-tuning requires a GPU and is isolated in `biobert_colab.ipynb`.

1. Upload `biobert_colab.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Enable GPU: **Runtime → Change Runtime Type → Hardware Accelerator → GPU**.
3. Upload your dataset CSVs to `/content/drive/MyDrive/medical_classification/data/`.
4. Select **Run All**.
5. Outputs and model checkpoints save automatically to `/content/drive/MyDrive/medical_classification/`.
6. After training completes (~5–10 minutes), copy the final metrics from the last evaluation cell into your report's comparison table.

---

## Results

All metrics are computed on the held-out test set under leakage-free conditions. Label keywords were **removed** from the corpus before vectorisation to prevent models from reverse-engineering the heuristic.

| Model | Accuracy | Macro F1 | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Logistic Regression (C = 10.0) | 81.50 % | 0.6105 | 0.7833 | 0.5500 | 0.87 |
| Multinomial NB (α = 0.05) | 81.22 % | 0.6370 | 0.6823 | 0.5867 | 0.97 |
| **BioBERT v1.2 (benchmark)** | **93.67 %** | **0.9362** | **0.9365** | **0.9360** | — |

*BioBERT result is an external benchmark; it was not trained or evaluated in this repository.*

**Class-level breakdown — Logistic Regression:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Treatment | 0.79 | 0.97 | 0.89 |
| Diagnosis | 0.79 | 0.37 | 0.50 |
| Prevention | 0.76 | 0.31 | 0.44 |

---

## Feature Interpretability

Top Logistic Regression coefficients per class align with clinical intuition:

- **Diagnosis:** resonance, magnetic resonance, accuracy, contrast, colonoscopy, hypersensitivity
- **Prevention:** vaccine, vaccination, incidences of, infection, influenza, programs
- **Treatment:** interventions, controlled, trials, doses, chemotherapy, double blind

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Keyword-heuristic label noise (11.5 % + 14.7 %) | Hard performance ceiling unrelated to model architecture |
| Sentence-role information lost during reconstruction | Methods and Results sentences processed identically |
| No semantic synonymy in TF-IDF | *renal failure* and *kidney failure* are orthogonal vectors |
| Single-label constraint | Multi-domain abstracts consistently misclassified |

---

## Replication

All code, preprocessing scripts, and trained model artefacts are in this public repository. The notebook is fully self-contained once the dataset folder is in place.

```bash
git clone https://github.com/purvadip/MedAbstractClassifier
cd MedAbstractClassifier
pip install -r requirements.txt
jupyter notebook   # open medical_classification.ipynb → Run All
```

Split indices in `split_indices.json` guarantee the same held-out test set across every run.

---

## References

Dernoncourt, F., & Lee, J. Y. (2017). PubMed 200k RCT: A dataset for sequential sentence classification in medical abstracts. *arXiv:1710.06071*.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv:1810.04805*.

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284.

Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics, 36*(4), 1234–1240.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.

Ratner, A. J., De Sa, C., Wu, S., Selsam, D., & Ré, C. (2017). Snorkel: Rapid training data creation with weak supervision. *VLDB Endowment, 11*(3), 269–282.
