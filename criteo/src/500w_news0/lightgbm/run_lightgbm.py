import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import gc


TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

VOCAB_SIZE = 20000
N_FOLDS = 5
SEED = 2026


COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
CAT_COLS = [f'C{i}' for i in range(1, 27)]
NUM_COLS = [f'I{i}' for i in range(1, 14)]

def process_data(df, encoders=None):
    df[NUM_COLS] = df[NUM_COLS].fillna(0).astype('float32')
    df[NUM_COLS] = np.log1p(df[NUM_COLS])
    if encoders is None:
        encoders = {}
        for col in CAT_COLS:
            df[col] = df[col].fillna('unk').astype(str)
            value_counts = df[col].value_counts()
            top_values = value_counts.index[:VOCAB_SIZE].tolist()
            vocab_map = {v: k+1 for k, v in enumerate(top_values)}
            encoders[col] = vocab_map
            df[col] = df[col].map(vocab_map).fillna(0).astype('int32')
    else:
        for col in CAT_COLS:
            df[col] = df[col].fillna('unk').astype(str)
            vocab_map = encoders[col]
            df[col] = df[col].map(vocab_map).fillna(0).astype('int32')
    return df, encoders

def main():
    print(f"[{time.strftime('%H:%M:%S')}] 开始 LightGBM 5-Fold Stacking")
    start_total = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] 读取训练集...")
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
    print(f"[{time.strftime('%H:%M:%S')}] 训练集特征编码...")
    train_df, encoders = process_data(train_df, encoders=None)
    y = train_df['label'].values
    X = train_df.drop(['label'], axis=1)
    del train_df
    gc.collect()

    print(f"[{time.strftime('%H:%M:%S')}] 读取测试集...")
    test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)
    test_labels_true = test_df['label'].values
    print(f"[{time.strftime('%H:%M:%S')}] 应用编码到测试集...")
    test_df, _ = process_data(test_df, encoders=encoders)
    X_test = test_df.drop(['label'], axis=1)
    del test_df
    gc.collect()

    print(f"[{time.strftime('%H:%M:%S')}] 初始化存储矩阵...")
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': SEED
    }

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"[{time.strftime('%H:%M:%S')}] 开始 {N_FOLDS} 折交叉验证...")

    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        fold_start = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] --- Fold {fold + 1} / {N_FOLDS} ---")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]

        lgb_train = lgb.Dataset(X_train_fold, y_train_fold, categorical_feature=CAT_COLS)
        lgb_val = lgb.Dataset(X_val_fold, y_val_fold, categorical_feature=CAT_COLS, reference=lgb_train)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=callbacks
        )

        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

        fold_auc = roc_auc_score(y_val_fold, val_pred)
        fold_time = time.time() - fold_start
        print(f"[{time.strftime('%H:%M:%S')}] Fold {fold+1} AUC: {fold_auc:.5f}, 耗时: {fold_time:.2f}秒")

        del X_train_fold, y_train_fold, X_val_fold, y_val_fold, lgb_train, lgb_val, model
        gc.collect()

    print("\n" + "="*40)
    print("LightGBM Stacking 最终报告")
    print("="*40)

    oof_auc = roc_auc_score(y, oof_preds)
    oof_loss = log_loss(y, oof_preds)
    test_auc = roc_auc_score(test_labels_true, test_preds)
    test_loss = log_loss(test_labels_true, test_preds)

    print(f"Train OOF AUC  : {oof_auc:.5f}")
    print(f"Train OOF Loss : {oof_loss:.5f}")
    print("-" * 20)
    print(f"Test Score AUC : {test_auc:.5f}")
    print(f"Test Score Loss: {test_loss:.5f}")
    print("="*40)

    total_time = time.time() - start_total
    print(f"\n全部完成！总耗时: {total_time:.2f} 秒")

if __name__ == '__main__':
    main()

# # Train OOF AUC  : 0.79319
# # Train OOF Loss : 0.45203
# # Test Score AUC : 0.78974
# # Test Score Loss: 0.45088