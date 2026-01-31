# UnfairClausesNLP

> The life of the law has not been logic; it has been experience. ~Oliver Wendell Holmes Jr.

UnfairClausesNLP is a notebook-first NLP project focused on detecting potentially unfair clauses in legal text / Terms of Service in the Polish language.  
The repository is kept lightweight: one main notebook + a small helper module for data downloading, loading and preparation.

---

## Dataset

The dataset is available on [Hugginface](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl/blob/main/abusive-clauses-pl.py). The dataset is pre-split into train / validation / test.
The `data/` folder in this repository is empty by default.  
Use the dedicated notebook cell to download and prepare the dataset locally.

---

## Repository structure

- `UnfairClauses.ipynb` - main notebook (EDA → preprocessing → modeling → evaluation)
- `unfair_clauses_data_provider.py` - utilities for downloading / loading / preparing the dataset
- `data/` - prepared data folder (created/populated by the notebook)

---

## Getting started

### 1. Create environment

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn
pip install matplotlib seaborn
pip install tqdm umap-learn optuna
pip install torch transformers datasets accelerate bitsandbytes
pip install google-genai
```
### 3. Gwmini setup (API key)
To use Gemini-related cells, you must provide your API key (that can be genereted here) in:
```
API_KEY.json
```

### 4. Hardware note (bielik)
Runing Bielik 11B with 4bit quanirzed weights requires minimum 5GiB of VRAM

## Result
<p align="center">
  <img src="https://github.com/user-attachments/assets/76b26e09-cea1-4055-a047-e8a8f35ef8f7" />
</p>
