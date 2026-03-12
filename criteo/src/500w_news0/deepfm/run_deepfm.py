import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import time
import gc
import os

TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBEDDING_DIM = 10
BATCH_SIZE = 4096
EPOCHS = 2
DNN_DROPOUT = 0.5        
N_FOLDS = 5              
SEED = 2026

VOCAB_SIZE = 20000

COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]

def main():
    print(f"[DeepFM Baseline] 开始 5-Fold 时序验证 (Device: {DEVICE})...")
    start_total = time.time()

    print("正在读取数据...")
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
    test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)
    
    test_labels_true = test_df['label'].values
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_df['is_train'] = 1
    test_df['is_train'] = 0
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    del train_df, test_df
    gc.collect()

    print("正在进行数据预处理...")
    
    all_df[DENSE_FEATURES] = all_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')
    all_df[DENSE_FEATURES] = all_df[DENSE_FEATURES].fillna(0).clip(lower=0)
    for col in DENSE_FEATURES:
        all_df[col] = np.log1p(all_df[col])

    print(f"正在处理类别特征 (保留训练集Top{VOCAB_SIZE}高频值)...")
    for feat in SPARSE_FEATURES:
        all_df[feat] = all_df[feat].fillna('unk').astype(str)
        train_feat_series = all_df[all_df['is_train'] == 1][feat]
        top_values = train_feat_series.value_counts().index[:VOCAB_SIZE].tolist()
        vocab_map = {v: i+1 for i, v in enumerate(top_values)}
        all_df[feat] = all_df[feat].map(vocab_map).fillna(0).astype('int32')

    print("正在生成DeepFM特征列...")
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=all_df[feat].max() + 1, embedding_dim=EMBEDDING_DIM)
        for feat in SPARSE_FEATURES
    ] + [
        DenseFeat(feat, 1) 
        for feat in DENSE_FEATURES
    ]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train = all_df[all_df['is_train'] == 1].reset_index(drop=True)
    test = all_df[all_df['is_train'] == 0].reset_index(drop=True)
    
    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}
    
    y = train['label'].values
    
    del all_df, train, test
    gc.collect()

    oof_preds = np.zeros(y.shape[0])
    test_preds = np.zeros(len(test_labels_true))
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,random_state=SEED)
    
    print(f"\n开始 {N_FOLDS} 折训练 (Batch: {BATCH_SIZE}, Epochs: {EPOCHS})...")
    
    for fold, (train_idx, val_idx) in enumerate(folds.split(train_model_input[feature_names[0]], y)):
        fold_start = time.time()
        print(f"\n--- Fold {fold + 1} / {N_FOLDS} ---")
        
        X_train_input = {name: train_model_input[name][train_idx] for name in feature_names}
        y_train_fold = y[train_idx]
        X_val_input = {name: train_model_input[name][val_idx] for name in feature_names}
        y_val_fold = y[val_idx]
        
        model = DeepFM(
            linear_feature_columns, dnn_feature_columns, 
            task='binary',
            l2_reg_embedding=1e-5, 
            l2_reg_dnn=1e-5,
            dnn_dropout=DNN_DROPOUT,
            dnn_hidden_units=(512, 256), 
            dnn_use_bn=True,
            device=DEVICE
        )
        
        model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
        
        model.fit(
            X_train_input, y_train_fold,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(X_val_input, y_val_fold)
        )
        
        val_pred = model.predict(X_val_input, batch_size=BATCH_SIZE*2)
        oof_preds[val_idx] = val_pred.flatten()
        
        test_pred_fold = model.predict(test_model_input, batch_size=BATCH_SIZE*2)
        test_preds += test_pred_fold.flatten() / folds.n_splits
        
        cur_auc = roc_auc_score(y_val_fold, val_pred)
        print(f"Fold {fold+1} AUC: {cur_auc:.5f} (耗时: {time.time()-fold_start:.1f}s)")
        
        del model, X_train_input, y_train_fold, X_val_input, y_val_fold
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*40)
    print("DeepFM 基线最终报告")
    print("="*40)
    
    oof_auc = roc_auc_score(y, oof_preds)
    oof_loss = log_loss(y, oof_preds)
    print(f"Train OOF AUC  : {oof_auc:.5f}")
    print(f"Train OOF Loss : {oof_loss:.5f}")
    
    test_auc = roc_auc_score(test_labels_true, test_preds)
    test_loss = log_loss(test_labels_true, test_preds)
    print(f"Test Score AUC : {test_auc:.5f}")
    print(f"Test Score Loss: {test_loss:.5f}")
    print("="*40)
    
    print(f"\n流程结束！总耗时: {time.time() - start_total:.2f} s")

if __name__ == '__main__':
    main()

# ========================================
# DeepFM 基线最终报告
# ========================================
# Train OOF AUC  : 0.79634
# Train OOF Loss : 0.44964
# Test Score AUC : 0.79534
# Test Score Loss: 0.44638
# ========================================