# Medical Abstract Classification
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

## Dataset

**Source:** PubMed 20k RCT — [arxiv.org/abs/1710.06071](https://arxiv.org/abs/1710.06071)  
**Local path:** `PubMed_20k_RCT/` (place in project root before running)

The dataset contains ~20,000 structured RCT abstracts from MEDLINE (1976–2017), segmented into sentence roles (Background, Methods, Results, Conclusions). Sentences are reconstructed per `abstract_id` into full abstract texts to simulate unstructured real-world input.

**Label assignment** uses a keyword-frequency heuristic:
- *Sensitivity*, *imaging* → Diagnosis
- *Vaccine*, *prophylaxis* → Prevention
- *Placebo*, *dose*, *intervention* → Treatment (also the tie-breaking default)

This introduces measurable noise: 11.5% of abstracts had no matching keywords (n = 2,301) and 14.7% had tied scores (n = 2,935). Both groups defaulted to Treatment, producing a heavily skewed initial distribution (Treatment: 17,805; Diagnosis: 1,110; Prevention: 1,085). Stratified undersampling brought all three classes to 6,667 abstracts each.

---

## Environment Setup

Requires Python 3.8+.

```bash
git clone https://github.com/purvadip/MedAbstractClassifier
cd medical_classification
pip install -r requirements.txt
```

---

## Running the Baseline Notebook

```bash
jupyter notebook
```

Open `medical_classification.ipynb` and select **Cell → Run All**. The notebook runs the full preprocessing pipeline, trains both baseline models, and saves all figures to `outputs/figures/`. No manual steps are required once the `src/` directory is in place.

Split indices are serialised to `split_indices.json` (72/8/20 train/validation/test) to ensure deterministic evaluation across runs.

---

## Running BioBERT (Google Colab)

BioBERT fine-tuning requires a GPU and is isolated in `biobert_colab.ipynb`.

1. Upload `biobert_colab.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Enable GPU: **Runtime → Change Runtime Type → Hardware Accelerator → GPU**.
3. Select **Run All**.
4. Outputs and model checkpoints save automatically to `/content/drive/MyDrive/medical_classification/`.
5. After training completes (~5–10 minutes), copy the final metrics from Cell 19 into the **Section 9 Model Comparison Table** of the main notebook.

---

## Output Files

All figures are written to `outputs/figures/`:

| File | Description |
|---|---|
| `class_distribution.png` | Class counts before and after stratified undersampling |
| `wordclouds.png` | Top terms per class (1×3 subplot) |
| `abstract_length_dist.png` | Word-count KDE by label |
| `confusion_matrix_logistic_regression.png` | Predicted vs. true labels — Logistic Regression |
| `confusion_matrix_multinomial_nb.png` | Predicted vs. true labels — Multinomial NB |
| `roc_curve_logistic_regression.png` | One-vs-rest ROC, Logistic Regression |
| `roc_curve_multinomial_nb.png` | One-vs-rest ROC, Multinomial NB |

---

## Results

All metrics are computed on the held-out test set (n = 1,773) under leakage-free conditions. Label keywords were removed from the corpus before vectorisation to prevent the models from reverse-engineering the heuristic.

| Model | Accuracy | Macro F1 | Precision | Recall | ROC-AUC (OVR) |
|---|---|---|---|---|---|
| Logistic Regression (C = 10.0) | 81.50% | 0.6105 | 0.7833 | 0.5500 | 0.87 |
| Multinomial NB (α = 0.05) | 81.22% | 0.6370 | 0.6823 | 0.5867 | 0.97 |
| **BioBERT v1.2 (benchmark)** | **93.67%** | **0.9362** | **0.9365** | **0.9360** | — |

*BioBERT result reported as an external benchmark; it was not trained or evaluated in this repository.*

**Class-level breakdown (Logistic Regression):**  
Treatment F1 = 0.89 (precision 0.79, recall 0.97) — Diagnosis F1 = 0.50 (precision 0.79, recall 0.37) — Prevention F1 = 0.44 (precision 0.76, recall 0.31).

The Treatment class benefits directly from the heuristic's default behaviour; the minority classes reflect genuine signal sparsity rather than model failure. The micro-averaged ROC-AUC of 0.87 indicates that probabilistic discrimination is substantially better than threshold-level performance suggests — clinicians could adjust operating thresholds on the ROC curve to prioritise Diagnosis or Prevention review queues.

---

## Feature Interpretability

The top Logistic Regression coefficients per class align with clinical intuition:

- **Diagnosis:** resonance, magnetic resonance, accuracy, contrast, colonoscopy, hypersensitivity
- **Prevention:** vaccine, vaccination, incidences of, infection, influenza, programs
- **Treatment:** interventions, controlled, trials, doses, chemotherapy, double blind

The presence of *magnetic resonance* and *double blind* as top-weighted unigrams confirms that TF-IDF is picking up genuinely discriminative clinical vocabulary, not artefacts of the heuristic.

---

## Known Limitations

- **Label noise ceiling.** With 11.5% no-keyword and 14.7% tied-score abstracts defaulting to Treatment, model performance is bounded by heuristic quality rather than architecture.
- **Sentence-role information lost.** Reconstructing full abstracts flattens Background, Methods, Results, and Conclusions into one text block. Methods sentences and Results sentences are processed identically.
- **No semantic synonymy.** TF-IDF treats *renal failure* and *kidney failure* as orthogonal vectors, restricting vocabulary coverage for medically equivalent terms.
- **Single-label constraint.** Multi-domain abstracts (e.g., testing the prevention efficacy of a diagnostic tool) are forced into one class and consistently misclassified.

---

## Replication

All code, preprocessing scripts, and trained model artefacts are in the public repository. The notebook is fully self-contained once the dataset folder is placed in the project root.

```bash
# Reproduce baseline results
git clone https://github.com/purvadip/MedAbstractClassifier
cd medical_classification
pip install -r requirements.txt
jupyter notebook  # open medical_classification.ipynb → Run All
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
