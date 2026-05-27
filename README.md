# Criteo CTR Prediction — Multi-Model Benchmark & Optimization

Click-Through Rate (CTR) prediction on the [Criteo](https://ailab.criteo.com/criteo-attribution-modeling-benchmark-dataset/) dataset, comparing Logistic Regression, LightGBM, and DeepFM baselines, followed by a GBDT+DeepFM hybrid with systematic ablation studies.

## Dataset

- **Source**: Criteo 1TB Click Logs (sampled to 6M records)
- **Split**: 5M training / 1M test
- **Features**: 1 label + 13 numerical (I1–I13) + 26 categorical (C1–C26)
- **CTR**: ~25% (train), ~24.6% (test)
- **Data notes**: Numerical features contain missing values (encoded as `< 0`); categorical features have high cardinality (some with 1M+ unique values)

## Project Structure

```
criteo/
├── src/
│   ├── EDA/
│   │   ├── new_eda.py          # Dataset statistics (missing rate, cardinality, distribution)
│   │   └── 3_26_eda.py         # Earlier EDA version
│   └── 500w_news0/
│       ├── lr/
│       │   └── run_lr.py       # Logistic Regression (SGDClassifier) baseline
│       ├── lightgbm/
│       │   └── run_lightgbm.py # LightGBM baseline with 5-fold stacking
│       ├── deepfm/
│       │   └── run_deepfm.py   # DeepFM baseline (deepctr-torch)
│       └── run_gbdt_deepfm/
│           ├── xiaorong.py     # GBDT+DeepFM hybrid with ablation experiments
│           └── auc_xiaorong.py # Optimized model with ROC plotting
├── auczong.py                  # Multi-model ROC comparison
└── *.png / *.npy               # Plots and prediction outputs
```

## Models

### 1. Logistic Regression (LR)

- **Library**: `sklearn.linear_model.SGDClassifier` (log_loss)
- **Preprocessing**: log1p + KBinsDiscretizer(20 bins) for numerical; top-20K frequency encoding + OneHot for categorical
- **Grid Search**: alpha ∈ {1e-6, 5e-6, 1e-5, 5e-5, 1e-4}, penalty ∈ {l1, l2, elasticnet}
- **Validation**: 5-fold StratifiedKFold
- **Test AUC**: ~0.7723 | **LogLoss**: ~0.4646

### 2. LightGBM

- **Library**: `lightgbm`
- **Preprocessing**: log1p for numerical; top-20K frequency encoding for categorical
- **Parameters**: num_leaves=31, learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.8
- **Training**: 5-fold stacking, early stopping (50 rounds), max 2000 rounds
- **Test AUC**: ~0.7906 | **LogLoss**: ~0.4502

### 3. DeepFM (Baseline)

- **Library**: `deepctr_torch`
- **Preprocessing**: log1p + MinMaxScaler for numerical; top-20K frequency encoding for categorical
- **Architecture**: embedding_dim=10, DNN(512→256), dropout=0.5, batch_size=4096, 2 epochs
- **Validation**: 5-fold StratifiedKFold
- **Test AUC**: ~0.7952 | **LogLoss**: ~0.4467

### 4. GBDT+DeepFM Hybrid (Optimized)

- **Concept**: LightGBM leaf embeddings → concatenated with original features → fed into DeepFM
- **GBDT**: 100 trees, 64 leaves each, extracts leaf indices as sparse features
- **DeepFM**: Shared embedding layer, dual FM + DNN paths, multi-field input

## Ablation Study

Five optimization techniques are systematically evaluated via on/off toggling:

| # | Optimization | Description | AUC Gain |
|---|-------------|-------------|----------|
| 1 | **Missing Indicator** | Binary flags for NaN/negative in numerical features | +0.0003 |
| 2 | **Continuous Binning** | KBinsDiscretizer(16 bins) → treat as categorical | -0.0000 |
| 3 | **Target Encoding** | 5-fold cross-validated smoothed target encoding | +0.0006 |
| 4 | **Scaled FM** | Learnable per-field scaling on FM embeddings | +0.0004 |
| 5 | **DNN BatchNorm** | Batch Normalization in Deep MLP layers | +0.0005 |

**Key finding**: Target Encoding and Scaled FM contribute the most. Continuous binning adds complexity without clear gain. The best configuration (all except binning) achieves **Test AUC 0.7989** with **LogLoss 0.4434**, stable across random seeds.

## Results Summary

| Model | Test AUC | Test LogLoss |
|-------|----------|-------------|
| Logistic Regression | 0.7723 | 0.4646 |
| LightGBM | 0.7906 | 0.4502 |
| DeepFM (baseline) | 0.7952 | 0.4467 |
| Base (no optimizations) | 0.7974 | 0.4448 |
| **Final (GBDT+DeepFM)** | **0.7989** | **0.4434** |

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- deepctr-torch
- lightgbm
- scikit-learn, pandas, numpy, matplotlib

Install dependencies:

```bash
pip install torch deepctr-torch lightgbm scikit-learn pandas numpy matplotlib
```

## Usage

**EDA**:
```bash
python criteo/src/EDA/new_eda.py
```

**Train baselines**:
```bash
python criteo/src/500w_news0/lr/run_lr.py
python criteo/src/500w_news0/lightgbm/run_lightgbm.py
python criteo/src/500w_news0/deepfm/run_deepfm.py
```

**Run ablation (toggle flags in script)**:
```bash
python criteo/src/500w_news0/run_gbdt_deepfm/xiaorong.py
```

**Plot comparison**:
```bash
python criteo/auczong.py
```
