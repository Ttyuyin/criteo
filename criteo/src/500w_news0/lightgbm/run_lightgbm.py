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
    df[NUM_COLS] = df[NUM_COLS].clip(lower=0)  
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





# 先算全局正负样本比例，算出统一的 F₀
# 全部数据用同一个 F₀ 做 Sigmoid，得到初始概率 P₀
# 每条数据用 (g=P_0-y) 算出各自误差梯度
# 把成千上万条数据 + 各自梯度 整体拿来分裂、生成第一棵树（分裂是看看能不能一刀把某属性里面的某个地方分开，让左右两边数据的梯度g之和差值增大，更容易分开）
# 所有数据顺着树走，各自落到对应叶子节点
# 每个叶子用内部所有数据的梯度和算 叶子权重 w
# 每条数据按自己所在叶子：(F_1 = F_0 + lr * w)每个人算出属于自己的 F₁（从此大家 F 不再统一）（w是所有梯度的所有梯度之和等等算出的结果）
# 下一轮：每条数据用自己独有的 F₁ 做 Sigmoid → 自己新概率 → 自己新梯度差值
# 再把全体数据 + 各自新梯度整体拿去生成下一棵树
# 循环往复直到 2000 轮结束





# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc as sk_auc
# from sklearn.model_selection import StratifiedKFold
# import time
# import gc
# import matplotlib.pyplot as plt

# TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
# TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

# VOCAB_SIZE = 20000
# N_FOLDS = 5
# SEED = 2026

# COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
# CAT_COLS = [f'C{i}' for i in range(1, 27)]
# NUM_COLS = [f'I{i}' for i in range(1, 14)]

# def process_data(df, encoders=None):
#     df[NUM_COLS] = df[NUM_COLS].fillna(0).astype('float32')
#     df[NUM_COLS] = np.log1p(df[NUM_COLS])
#     if encoders is None:
#         encoders = {}
#         for col in CAT_COLS:
#             df[col] = df[col].fillna('unk').astype(str)
#             value_counts = df[col].value_counts()
#             top_values = value_counts.index[:VOCAB_SIZE].tolist()
#             vocab_map = {v: k+1 for k, v in enumerate(top_values)}
#             encoders[col] = vocab_map
#             df[col] = df[col].map(vocab_map).fillna(0).astype('int32')
#     else:
#         for col in CAT_COLS:
#             df[col] = df[col].fillna('unk').astype(str)
#             vocab_map = encoders[col]
#             df[col] = df[col].map(vocab_map).fillna(0).astype('int32')
#     return df, encoders

# def main():
#     print(f"[{time.strftime('%H:%M:%S')}] 开始 LightGBM 5-Fold")
#     start_total = time.time()

#     print(f"[{time.strftime('%H:%M:%S')}] 读取训练集...")
#     train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
#     train_df, encoders = process_data(train_df, encoders=None)
#     y = train_df['label'].values
#     X = train_df.drop(['label'], axis=1)
#     del train_df
#     gc.collect()

#     print(f"[{time.strftime('%H:%M:%S')}] 读取测试集...")
#     test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)
#     test_labels_true = test_df['label'].values
#     test_df, _ = process_data(test_df, encoders=encoders)
#     X_test = test_df.drop(['label'], axis=1)
#     del test_df
#     gc.collect()

#     oof_preds = np.zeros(X.shape[0])
#     test_preds = np.zeros(X_test.shape[0])

#     params = {
#         'objective': 'binary',
#         'metric': 'auc',
#         'boosting_type': 'gbdt',
#         'num_leaves': 31,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.9,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 5,
#         'verbose': -1,
#         'n_jobs': -1,
#         'seed': SEED
#     }

#     folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

#     for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
#         fold_start = time.time()
#         print(f"\n--- Fold {fold + 1} / {N_FOLDS} ---")

#         X_train_fold = X.iloc[train_idx]
#         y_train_fold = y[train_idx]
#         X_val_fold = X.iloc[val_idx]
#         y_val_fold = y[val_idx]

#         lgb_train = lgb.Dataset(X_train_fold, y_train_fold, categorical_feature=CAT_COLS)
#         lgb_val = lgb.Dataset(X_val_fold, y_val_fold, categorical_feature=CAT_COLS, reference=lgb_train)

#         model = lgb.train(
#             params,
#             lgb_train,
#             num_boost_round=2000,
#             valid_sets=[lgb_val],
#             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
#         )

#         val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
#         oof_preds[val_idx] = val_pred
#         test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS

#         fold_auc = roc_auc_score(y_val_fold, val_pred)
#         print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

#     # ===================== 最终指标 =====================
#     oof_auc = roc_auc_score(y, oof_preds)
#     test_auc = roc_auc_score(test_labels_true, test_preds)
#     test_loss = log_loss(test_labels_true, test_preds)

#     print("\n===== LightGBM 最终结果 =====")
#     print(f"Test AUC : {test_auc:.5f}")
#     print(f"Test Loss: {test_loss:.5f}")

#     # ===================== 画 ROC 图 =====================
#     fpr, tpr, _ = roc_curve(test_labels_true, test_preds)
#     roc_auc = sk_auc(fpr, tpr)

#     plt.figure(figsize=(7,6))
#     plt.plot(fpr, tpr, lw=2, color='#0066ff', label=f'AUC = {roc_auc:.4f}')
#     plt.plot([0,1],[0,1], '--', color='gray')
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.title('LightGBM Test ROC Curve')
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.savefig('lgb_single_roc.png', dpi=300)
#     plt.show()

#     # ===================== 保存数据 =====================
#     np.save("y_true.npy", test_labels_true)
#     np.save("pred_lgb.npy", test_preds)

#     print(f"\n总耗时：{time.time() - start_total:.2f}s")

# if __name__ == '__main__':
#     main()


































#SEED=2026
# ========================================
# LightGBM Stacking 最终报告
# ========================================
# Train OOF AUC  : 0.79471
# Train OOF Loss : 0.45071
# --------------------
# Test Score AUC : 0.79061
# Test Score Loss: 0.45018
# ========================================


# SEED=2025
# ========================================
# LightGBM Stacking 最终报告
# ========================================
# Train OOF AUC  : 0.79461
# Train OOF Loss : 0.45080
# --------------------
# Test Score AUC : 0.79044
# Test Score Loss: 0.45030
# ========================================

# 全部完成！总耗时: 2516.52 秒


# SEED=2024
# ========================================
# LightGBM Stacking 最终报告
# ========================================
# Train OOF AUC  : 0.79461
# Train OOF Loss : 0.45080
# --------------------
# Test Score AUC : 0.79065
# Test Score Loss: 0.45016
# ========================================

# 全部完成！总耗时: 2545.21 秒