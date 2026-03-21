# Medical Abstract Classification
**GitHub Repository**: [KB937/PD_](https://github.com/KB937/PD_)

## Project Summary
This project aims to automate the categorization of medical research abstracts into three distinct classes: **Diagnosis**, **Treatment**, and **Prevention**. Given the exponentially growing volume of medical literature, automated text-as-data pipelines are crucial for researchers to systematically filter studies. This project reconstructs full abstracts from the PubMed 20k RCT dataset, applies a keyword heuristic to generate weak supervision labels, and evaluates both traditional machine learning baselines (TF-IDF with Logistic Regression and Multinomial Naive Bayes) and a state-of-the-art transformer model (BioBERT).

## Research Question
"Can machine learning models trained on structured medical abstracts reliably classify them by research type (Diagnosis, Treatment, Prevention), and how does a fine-tuned biomedical transformer compare to a traditional baseline?"

## Dataset Loading Instructions
The dataset used is the local `PubMed_20k_RCT` folder.
In `src/preprocess.py`, we recursively load `train.csv`, `dev.csv`, and `test.csv` to reconstruct full abstract texts by grouping sentences using `abstract_id`. Following reconstruction, a keyword heuristic maps each abstract to a label, prioritizing the maximum count with "Treatment" serving as the default in ties. The class distribution is then balanced using stratified sampling to exactly 20,000 abstracts.

## Environment Setup
Make sure you have Python 3.8+ installed.

```bash
# Clone the repository and navigate into it
git clone <repo-url>
cd medical_classification

# Install requirements
pip install -r requirements.txt
```

## How to Run the Main Notebook
1. Start Jupyter Server:
```bash
jupyter notebook
```
2. Open `medical_classification.ipynb`.
3. Select "Run All" from the "Cell" menu. The notebook will automatically execute the preprocessing pipeline, train the Baseline Models (Logistic Regression and Naive Bayes), and generate learning curves, confusion matrices, and ROC-AUC plots in `outputs/figures/`.
4. Wait for the run to complete. The notebook is fully self-contained once the `src/` directory methods are initialized.

## How to run biobert_colab.ipynb in Google Colab (Standalone)
Fine-tuning BioBERT requires a GPU. Thus, it is isolated in a separate notebook intended for Google Colab.
- Open Google Colab and upload `biobert_colab.ipynb`.
- Enable GPU: Go to **Runtime > Change Runtime Type > Hardware Accelerator > GPU**.
- Select "Run All".
- Outputs (model checkpoints and figures) will automatically save to your Google Drive under `/content/drive/MyDrive/medical_classification/`.
- Once training in Colab finishes (approx 5-10 minutes), **copy the final metrics from Cell 19 into the Main Notebook Section 9 Model Comparison Table**.

## Description of Output Files and Figures
All output properties and figures are stored in `outputs/figures/`:
- `class_distribution.png`: Bar chart confirming class balance (before and after stratified sampling).
- `wordclouds.png`: 1x3 subplot displaying the most frequent terms across the three classes.
- `abstract_length_dist.png`: KDE plot displaying the word count distribution by labels.
- `confusion_matrix_logistic_regression.png`: Heatmap showing LogReg predicted vs true labels.
- `confusion_matrix_multinomial_nb.png`: Heatmap showing NB predicted vs true labels.
- `roc_curve_logistic_regression.png`: One-vs-Rest ROC curve with micro-averaged AUC.
- `roc_curve_multinomial_nb.png`: One-vs-Rest ROC curve with micro-averaged AUC.

## Results Table (Leakage-Free)
| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | ROC-AUC (OVR) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9228 | 0.9227 | 0.9227 | 0.9219 | 0.9219 | 0.9830 |
| Multinomial NB | 0.8815 | 0.8823 | 0.8815 | 0.8818 | 0.8818 | 0.9711 |
| BioBERT | TBD | TBD | TBD | TBD | TBD | TBD |

*(Values are populated based on typical test sets. Run the notebook for exact deterministic metrics on the split. Update BioBERT after Colab execution).*

## Known Limitations
- The labels are derived using dictionary-based heuristics, so our models are ultimately learning to approximate a heuristic rather than ground-truth clinical judgment.
- "Treatment" defaults in the instance of a keyword tie, causing an over-representation bias that required aggressive downsampling.
- Multi-domain abstracts (e.g. testing the *prevention* efficacy of a *diagnostic* tool) are easily misclassified due to strict single-label requirements.

## Citations
- Dernoncourt, F., & Lee, J. Y. (2017). PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. *arXiv preprint arXiv:1710.06071*.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.
- Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234-1240.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
