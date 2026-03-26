import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM
from deepctr_torch.callbacks import EarlyStopping
import time
import gc
import warnings
import torch.nn as nn
from deepctr_torch.models.deepfm import DeepFM as BaseDeepFM
warnings.filterwarnings('ignore')

# ================= 参数配置=================
TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS = 5
SEED = 2026

EMBEDDING_DIM = 10
BATCH_SIZE = 4096
EPOCHS = 2
DNN_DROPOUT = 0.5
DNN_HIDDEN_UNITS = (512, 256)
DNN_USE_BN = True

GBDT_NUM_LEAVES = 64
GBDT_TREES = 100
FREQ_FILTER_THRESHOLD = 5
N_BINS = 16
TARGET_ENCODING_SMOOTH = 10

COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]


class ScaledDeepFM(DeepFM):
    def __init__(self, linear_feature_columns, dnn_feature_columns, **kwargs):
        super().__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        
        self.n_fields = len(self.embedding_dict)
        
        self.field_scale = nn.Parameter(torch.ones(self.n_fields))

    def fm(self, fm_input):
        """
        fm_input shape: (batch_size, field_num, embed_dim)
        """
        scale = self.field_scale.to(fm_input.device).view(1, -1, 1)
        
        scaled = fm_input * scale
        
        square_of_sum = torch.sum(scaled, dim=1) ** 2
        sum_of_square = torch.sum(scaled ** 2, dim=1)
        
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        
        return torch.sum(cross_term, dim=1, keepdim=True)
# ================= 数据处理 =================
def process_data_hybrid():
    print(f"[{time.strftime('%H:%M:%S')}] 读取数据...")
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
    test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)

    train_labels = train_df['label'].values
    test_labels = test_df['label'].values
    train_df['is_train'] = 1
    test_df['is_train'] = 0

    train_df[DENSE_FEATURES] = train_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')
    test_df[DENSE_FEATURES] = test_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')

    for col in DENSE_FEATURES:
        train_df[col + '_miss'] = ((train_df[col] < 0) | (train_df[col].isna())).astype(int)
        test_df[col + '_miss'] = ((test_df[col] < 0) | (test_df[col].isna())).astype(int)

    train_df[DENSE_FEATURES] = train_df[DENSE_FEATURES].fillna(0).clip(lower=0)
    test_df[DENSE_FEATURES] = test_df[DENSE_FEATURES].fillna(0).clip(lower=0)

    print(f"[{time.strftime('%H:%M:%S')}] 分桶离散化...")
    BIN_FEATURES = [f'{col}_bin' for col in DENSE_FEATURES]
    est = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')
    est.fit(train_df[DENSE_FEATURES])

    train_bins = est.transform(train_df[DENSE_FEATURES])
    test_bins = est.transform(test_df[DENSE_FEATURES])

    train_df[BIN_FEATURES] = train_bins.astype(int).astype(str)
    test_df[BIN_FEATURES] = test_bins.astype(int).astype(str)

    print(f"[{time.strftime('%H:%M:%S')}] 数值特征 log+归一化...")
    for col in DENSE_FEATURES:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])

    mms = MinMaxScaler()
    train_df[DENSE_FEATURES] = mms.fit_transform(train_df[DENSE_FEATURES])
    test_df[DENSE_FEATURES] = mms.transform(test_df[DENSE_FEATURES])

    print(f"[{time.strftime('%H:%M:%S')}] 类别特征低频过滤...")
    for feat in SPARSE_FEATURES:
        train_df[feat] = train_df[feat].fillna('-1').astype(str)
        test_df[feat] = test_df[feat].fillna('-1').astype(str)

        cnt = train_df[feat].value_counts()
        valid = set(cnt[cnt >= FREQ_FILTER_THRESHOLD].index)
        train_df.loc[~train_df[feat].isin(valid), feat] = 'UNK'
        test_df.loc[~test_df[feat].isin(valid), feat] = 'UNK'

    return train_df, test_df, train_labels, test_labels, SPARSE_FEATURES, DENSE_FEATURES, BIN_FEATURES

# ================= 目标编码=================
def target_encoding_cv(train_df, test_df, cat_cols, target, n_folds=5, smooth=10):
    print(f"[{time.strftime('%H:%M:%S')}] 目标编码...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = train_df[target].mean()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for col in cat_cols:
        enc_col = col + '_target_enc'
        train_df[enc_col] = np.nan
        test_df[enc_col] = np.nan

        for tr_idx, val_idx in skf.split(train_df, train_df[target]):
            tr_means = train_df.iloc[tr_idx].groupby(col)[target].agg(['count', 'mean']).reset_index()
            tr_means['enc'] = (tr_means['count'] * tr_means['mean'] + global_mean * smooth) / (tr_means['count'] + smooth)
            val_map = tr_means.set_index(col)['enc'].to_dict()
            train_df.loc[val_idx, enc_col] = train_df.loc[val_idx, col].map(val_map)

        train_df[enc_col].fillna(global_mean, inplace=True)

        full_means = train_df.groupby(col)[target].agg(['count', 'mean']).reset_index()
        full_means['enc'] = (full_means['count'] * full_means['mean'] + global_mean * smooth) / (full_means['count'] + smooth)
        test_map = full_means.set_index(col)['enc'].to_dict()
        test_df[enc_col] = test_df[col].map(test_map)
        test_df[enc_col].fillna(global_mean, inplace=True)

    return train_df, test_df

# ================= GBDT叶子特征提取 =================
def get_leaf_indices(X_train, y_train, X_val, X_test):
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, reference=lgb_train, free_raw_data=False)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': GBDT_NUM_LEAVES,
        'learning_rate': 0.1,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=GBDT_TREES)

    batch_size = 100000  
    n_trees = GBDT_TREES
    block_size = 20
    n_blocks = (n_trees + block_size - 1) // block_size

    def predict_leaf_blocks(X):
        n_samples = len(X)
        block_lists = [[] for _ in range(n_blocks)]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            if hasattr(X, 'iloc'):
                X_batch = X.iloc[start:end]
            else:
                X_batch = X[start:end]
            leaf_batch = model.predict(X_batch, pred_leaf=True)  
            for b in range(n_blocks):
                tree_start = b * block_size
                tree_end = min((b+1) * block_size, n_trees)
                block_data = leaf_batch[:, tree_start:tree_end]
                block_lists[b].append(block_data)
        result = []
        for b in range(n_blocks):
            if len(block_lists[b]) == 0:
                result.append(np.empty((0, block_size), dtype=np.uint8))
            else:
                result.append(np.vstack(block_lists[b]).astype(np.uint8))  
        return result

    tr_leaf_blocks = predict_leaf_blocks(X_train)
    val_leaf_blocks = predict_leaf_blocks(X_val)
    te_leaf_blocks = predict_leaf_blocks(X_test)
    return tr_leaf_blocks, val_leaf_blocks, te_leaf_blocks

# ================= 主函数 =================
def main():
    print("代码运行中...请耐心等待结果输出")
    start_total = time.time()

    # 1. 数据处理
    train_df, test_df, y_train_full, y_test_true, sparse_cols, dense_cols, bin_cols = process_data_hybrid()

    # 2. 目标编码
    cat_cols_for_te = sparse_cols + bin_cols
    train_df, test_df = target_encoding_cv(
        train_df, test_df, cat_cols_for_te, target='label',
        n_folds=N_FOLDS, smooth=TARGET_ENCODING_SMOOTH
    )
    te_features = [col + '_target_enc' for col in cat_cols_for_te]

    # 3. Label Encoding（分别处理）
    # print(f"[{time.strftime('%H:%M:%S')}] Label Encoding...")
    # all_cat_features = sparse_cols + bin_cols + [col+'_miss' for col in dense_cols]
    # for feat in all_cat_features:
    #     le = LabelEncoder()
    #     train_vals = train_df[feat].astype(str).values
    #     test_vals = test_df[feat].astype(str).values
    #     le.fit(train_vals)
    #     train_df[feat] = le.transform(train_vals).astype(np.int32)
    #     test_df[feat] = le.transform(test_vals).astype(np.int32)


    print(f"[{time.strftime('%H:%M:%S')}] Label Encoding...")
    all_cat_features = sparse_cols + bin_cols + [col+'_miss' for col in dense_cols]
    for feat in all_cat_features:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[feat], test_df[feat]], axis=0).astype(str)
        le.fit(all_vals)

        train_df[feat] = le.transform(train_df[feat].astype(str)).astype(np.int32)
        test_df[feat] = le.transform(test_df[feat].astype(str)).astype(np.int32)

    # 4. 准备特征列定义
    sparse_features = all_cat_features
    vocab_sizes = {feat: train_df[feat].max() + 1 for feat in sparse_features}
    all_dense_cols = dense_cols + te_features

    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=vocab_sizes[feat], embedding_dim=EMBEDDING_DIM)
        for feat in sparse_features
    ]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in all_dense_cols]
    gbdt_leaf_cols = [
        SparseFeat(f'gbdt_tree_{i}', vocabulary_size=GBDT_NUM_LEAVES+1, embedding_dim=EMBEDDING_DIM)
        for i in range(GBDT_TREES)
    ]
    linear_feature_columns = fixlen_feature_columns + gbdt_leaf_cols + dense_feature_columns
    dnn_feature_columns = linear_feature_columns

    # 5. 交叉验证
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(folds.split(train_df.drop(['label','is_train'], axis=1), y_train_full)):
        print(f"\n>>> Fold {fold+1}/{N_FOLDS}")

        # 分割数据
        X_tr_lgb = train_df.drop(['label','is_train'], axis=1).iloc[tr_idx]
        y_tr = y_train_full[tr_idx]
        X_val_lgb = train_df.drop(['label','is_train'], axis=1).iloc[val_idx]
        y_val = y_train_full[val_idx]
        X_test_lgb = test_df.drop(['label','is_train'], axis=1)

        # 获取叶子特征块
        tr_leaf, val_leaf, te_leaf = get_leaf_indices(X_tr_lgb, y_tr, X_val_lgb, X_test_lgb)

        # 构建输入字典
        def build_input(df, leaves):
            d = {}
            # 数值特征用 float16
            for name in all_dense_cols:
                d[name] = df[name].values.astype(np.float16)
            # 稀疏特征用 int32
            for name in sparse_features:
                d[name] = df[name].values.astype(np.int32)
            # 叶子特征用 uint8
            leaf_idx = 0
            for leaf_block in leaves:
                block = leaf_block  
                for i in range(block.shape[1]):
                    d[f'gbdt_tree_{leaf_idx}'] = block[:, i]
                    leaf_idx += 1
            return d

        tr_inp = build_input(train_df.drop(['label','is_train'], axis=1).iloc[tr_idx], tr_leaf)
        val_inp = build_input(train_df.drop(['label','is_train'], axis=1).iloc[val_idx], val_leaf)
        te_inp = build_input(test_df.drop(['label','is_train'], axis=1), te_leaf)

        # 释放不再需要的大对象
        del X_tr_lgb, X_val_lgb, X_test_lgb, tr_leaf, val_leaf, te_leaf
        gc.collect()

        # 创建模型
        model = ScaledDeepFM(
            linear_feature_columns, dnn_feature_columns,
            task='binary',
            l2_reg_embedding=1e-5,
            l2_reg_dnn=1e-5,
            dnn_dropout=DNN_DROPOUT,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_use_bn=DNN_USE_BN,
            device=DEVICE
        )
        model.compile('adam', 'binary_crossentropy', metrics=['auc'])
        es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=0, patience=1, mode='max')

        model.fit(
            tr_inp, y_tr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(val_inp, y_val),
            callbacks=[es]
        )

        # 预测
        val_pred = model.predict(val_inp, batch_size=BATCH_SIZE).flatten()
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(te_inp, batch_size=BATCH_SIZE).flatten() / N_FOLDS

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_logloss = log_loss(y_val, val_pred)
        print(f"    Fold AUC: {fold_auc:.6f}, Fold LogLoss: {fold_logloss:.6f}")

        # 清理
        del model, tr_inp, val_inp, te_inp
        torch.cuda.empty_cache()
        gc.collect()

    # 6. 最终评估
    print("\n" + "="*50)
    print(" 最终的实验结果")
    print("="*50)
    train_auc = roc_auc_score(y_train_full, oof_preds)
    train_logloss = log_loss(y_train_full, oof_preds)
    print(f"Train OOF AUC     : {train_auc:.6f}")
    print(f"Train OOF LogLoss : {train_logloss:.6f}")
    print("-" * 20)
    test_auc = roc_auc_score(y_test_true, test_preds)
    test_logloss = log_loss(y_test_true, test_preds)
    print(f"Test Score AUC    : {test_auc:.6f}")
    print(f"Test Score LogLoss: {test_logloss:.6f}")
    print("="*50)

    print(f" 总耗时: {(time.time()-start_total)/60:.1f} 分钟")

if __name__ == '__main__':
    main()







# ==================================================
# 最终的实验结果（2024）
# ==================================================
# Train OOF AUC     : 0.795449
# Train OOF LogLoss : 0.451546
# --------------------
# Test Score AUC    : 0.798547
# Test Score LogLoss: 0.443932
# ==================================================
# 总耗时: 56.0 分钟
# PS D:\code\Python\dachuang\criteo> 


# ==================================================
# 最终的实验结果（2025）
# ==================================================
# Train OOF AUC     : 0.795466
# Train OOF LogLoss : 0.451387
# --------------------
# Test Score AUC    : 0.798785
# Test Score LogLoss: 0.443585
# ==================================================
# 总耗时: 56.5 分钟


# ==================================================
# 最终的实验结果（2026）
# ==================================================
# Train OOF AUC     : 0.795637
# Train OOF LogLoss : 0.451175
# --------------------
# Test Score AUC    : 0.798589
# Test Score LogLoss: 0.443661
# ==================================================
# 总耗时: 57.4 分钟

# ==================================================
# 最终的实验结果（2027）
# ==================================================
# Train OOF AUC     : 0.795540
# Train OOF LogLoss : 0.451083
# --------------------
# Test Score AUC    : 0.798680
# Test Score LogLoss: 0.443531
# ==================================================


# ==================================================
# 最终的实验结果（2028）
# ==================================================
# Train OOF AUC     : 0.795632
# Train OOF LogLoss : 0.451193
# --------------------
# Test Score AUC    : 0.798545
# Test Score LogLoss: 0.443775
# ==================================================
# 总耗时: 55.5 分钟

