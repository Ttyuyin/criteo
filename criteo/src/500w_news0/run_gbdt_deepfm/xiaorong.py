import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM as BaseDeepFM
from deepctr_torch.callbacks import EarlyStopping
import time
import gc
import warnings

warnings.filterwarnings('ignore')

# ================= ⚙️ 核心优化点消融实验配置区 =================
# 通过修改以下的 True / False 进行消融实验
USE_MISSING_INDICATOR = False   # 优化点1：是否添加连续特征的缺失/异常值指示特征 (_miss)
USE_BINNING           = True  # 优化点2：是否对连续特征进行分桶离散化 (_bin)
USE_TARGET_ENCODING   = True     # 优化点3：是否使用交叉验证的目标编码 (_target_enc)
USE_SCALED_FM         = False     # 优化点4：是否使用改进版的 ScaledDeepFM (带自适应特征权重)
USE_DNN_BN            = True     # 优化点5：DNN 层是否使用 Batch Normalization

# ================= 🔧 基础参数配置 =================
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

GBDT_NUM_LEAVES = 64
GBDT_TREES = 100
FREQ_FILTER_THRESHOLD = 5
N_BINS = 16
TARGET_ENCODING_SMOOTH = 10

COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]

# ================= 🌟 改进版模型定义 =================
class ScaledDeepFM(BaseDeepFM):
    """带自适应Field权重的Scaled FM"""
    def __init__(self, linear_feature_columns, dnn_feature_columns, **kwargs):
        super().__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        self.n_fields = len(self.embedding_dict)
        self.field_scale = nn.Parameter(torch.ones(self.n_fields))

    def fm(self, fm_input):
        scale = self.field_scale.to(fm_input.device).view(1, -1, 1)
        scaled = fm_input * scale
        square_of_sum = torch.sum(scaled, dim=1) ** 2
        sum_of_square = torch.sum(scaled ** 2, dim=1)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return torch.sum(cross_term, dim=1, keepdim=True)

# ================= 📊 数据处理 (支持动态开关) =================
def process_data_ablation():
    print(f"[{time.strftime('%H:%M:%S')}] 读取数据...")
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
    test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)

    y_train = train_df['label'].values
    y_test = test_df['label'].values
    train_df['is_train'] = 1
    test_df['is_train'] = 0

    train_df[DENSE_FEATURES] = train_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')
    test_df[DENSE_FEATURES] = test_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')

    final_sparse_cols = SPARSE_FEATURES.copy()
    final_dense_cols = DENSE_FEATURES.copy()

    # --- [消融] 优化点1：缺失值指示 ---
    if USE_MISSING_INDICATOR:
        print(f"[{time.strftime('%H:%M:%S')}] 添加缺失值指示特征...")
        for col in DENSE_FEATURES:
            miss_col = col + '_miss'
            train_df[miss_col] = ((train_df[col] < 0) | (train_df[col].isna())).astype(int).astype(str)
            test_df[miss_col] = ((test_df[col] < 0) | (test_df[col].isna())).astype(int).astype(str)
            final_sparse_cols.append(miss_col)

    # 填充与截断
    train_df[DENSE_FEATURES] = train_df[DENSE_FEATURES].fillna(0).clip(lower=0)
    test_df[DENSE_FEATURES] = test_df[DENSE_FEATURES].fillna(0).clip(lower=0)

    # --- [消融] 优化点2：连续特征分桶 ---
    if USE_BINNING:
        print(f"[{time.strftime('%H:%M:%S')}] 连续特征分桶离散化...")
        BIN_FEATURES = [f'{col}_bin' for col in DENSE_FEATURES]
        est = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')
        
        train_bins = est.fit_transform(train_df[DENSE_FEATURES])
        test_bins = est.transform(test_df[DENSE_FEATURES])

        train_df[BIN_FEATURES] = train_bins.astype(int).astype(str)
        test_df[BIN_FEATURES] = test_bins.astype(int).astype(str)
        final_sparse_cols.extend(BIN_FEATURES)

    # 数值特征 Log + 归一化 (无论哪个版本都做)
    print(f"[{time.strftime('%H:%M:%S')}] 数值特征 log1p & 归一化...")
    for col in DENSE_FEATURES:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])

    mms = MinMaxScaler()
    train_df[DENSE_FEATURES] = mms.fit_transform(train_df[DENSE_FEATURES])
    test_df[DENSE_FEATURES] = mms.transform(test_df[DENSE_FEATURES])

    # 类别特征低频过滤
    print(f"[{time.strftime('%H:%M:%S')}] 类别特征低频过滤...")
    for feat in SPARSE_FEATURES: # 只过滤原始稀疏特征
        train_df[feat] = train_df[feat].fillna('-1').astype(str)
        test_df[feat] = test_df[feat].fillna('-1').astype(str)

        cnt = train_df[feat].value_counts()
        valid = set(cnt[cnt >= FREQ_FILTER_THRESHOLD].index)
        train_df.loc[~train_df[feat].isin(valid), feat] = 'UNK'
        test_df.loc[~test_df[feat].isin(valid), feat] = 'UNK'

    return train_df, test_df, y_train, y_test, final_sparse_cols, final_dense_cols

# ================= 🎯 目标编码函数 =================
def target_encoding_cv(train_df, test_df, cat_cols, target, n_folds=5, smooth=10):
    print(f"[{time.strftime('%H:%M:%S')}] 执行交叉验证目标编码...")
    global_mean = train_df[target].mean()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    te_features = []

    for col in cat_cols:
        enc_col = col + '_target_enc'
        te_features.append(enc_col)
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

    return train_df, test_df, te_features

# ================= 🌲 GBDT 提取特征 =================
def get_leaf_indices(X_train, y_train, X_val, X_test):
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, reference=lgb_train, free_raw_data=False)
    params = {
        'objective': 'binary', 'metric': 'auc', 'num_leaves': GBDT_NUM_LEAVES,
        'learning_rate': 0.1, 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
        'verbose': -1, 'n_jobs': -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=GBDT_TREES)

    batch_size, n_trees, block_size = 100000, GBDT_TREES, 20
    n_blocks = (n_trees + block_size - 1) // block_size

    def predict_leaf_blocks(X):
        n_samples = len(X)
        block_lists = [[] for _ in range(n_blocks)]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X.iloc[start:end] if hasattr(X, 'iloc') else X[start:end]
            leaf_batch = model.predict(X_batch, pred_leaf=True)
            for b in range(n_blocks):
                tree_start, tree_end = b * block_size, min((b+1) * block_size, n_trees)
                block_lists[b].append(leaf_batch[:, tree_start:tree_end])
        return [np.vstack(block_lists[b]).astype(np.uint8) if block_lists[b] else np.empty((0, block_size), dtype=np.uint8) for b in range(n_blocks)]

    return predict_leaf_blocks(X_train), predict_leaf_blocks(X_val), predict_leaf_blocks(X_test)

# ================= 🚀 主函数 =================
def main():
    print("="*50)
    print("🔬 核心优化点消融实验配置状态:")
    print(f"[-] 缺失值指示 (USE_MISSING_INDICATOR): {USE_MISSING_INDICATOR}")
    print(f"[-] 连续特征分桶 (USE_BINNING)        : {USE_BINNING}")
    print(f"[-] 目标编码 (USE_TARGET_ENCODING)    : {USE_TARGET_ENCODING}")
    print(f"[-] Scaled FM机制 (USE_SCALED_FM)     : {USE_SCALED_FM}")
    print(f"[-] DNN Batch Norm (USE_DNN_BN)       : {USE_DNN_BN}")
    print("="*50)

    start_total = time.time()

    # 1. 基础数据处理 (动态)
    train_df, test_df, y_train_full, y_test_true, sparse_cols, dense_cols = process_data_ablation()

    # --- [消融] 优化点3：目标编码 ---
    if USE_TARGET_ENCODING:
        # 针对原始稀疏特征和分桶特征进行 Target Encoding (不含缺失指示列)
        cat_cols_for_te = SPARSE_FEATURES.copy()
        if USE_BINNING:
            cat_cols_for_te.extend([f'{col}_bin' for col in DENSE_FEATURES])
        
        train_df, test_df, te_features = target_encoding_cv(
            train_df, test_df, cat_cols_for_te, target='label',
            n_folds=N_FOLDS, smooth=TARGET_ENCODING_SMOOTH
        )
        dense_cols.extend(te_features) # 把编码后的特征当做连续值输入 DNN

    # 2. Label Encoding (所有稀疏特征统一编码)
    print(f"[{time.strftime('%H:%M:%S')}] Label Encoding...")
    for feat in sparse_cols:
        le = LabelEncoder()
        train_vals = train_df[feat].values
        test_vals = test_df[feat].values
        # 合并 fit 防止 unseen labels 报错
        le.fit(np.concatenate([train_vals, test_vals])) 
        train_df[feat] = le.transform(train_vals).astype(np.int32)
        test_df[feat] = le.transform(test_vals).astype(np.int32)

    # 3. 构建 Feature Columns
    vocab_sizes = {feat: train_df[feat].max() + 1 for feat in sparse_cols}
    
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocab_sizes[feat], embedding_dim=EMBEDDING_DIM) for feat in sparse_cols]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_cols]
    gbdt_leaf_cols = [SparseFeat(f'gbdt_tree_{i}', vocabulary_size=GBDT_NUM_LEAVES+1, embedding_dim=EMBEDDING_DIM) for i in range(GBDT_TREES)]
    
    linear_feature_columns = fixlen_feature_columns + gbdt_leaf_cols + dense_feature_columns
    dnn_feature_columns = linear_feature_columns

    # 4. 交叉验证训练
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(folds.split(train_df.drop(['label','is_train'], axis=1), y_train_full)):
        print(f"\n>>> Fold {fold+1}/{N_FOLDS}")

        X_tr_lgb = train_df.drop(['label','is_train'], axis=1).iloc[tr_idx]
        y_tr = y_train_full[tr_idx]
        X_val_lgb = train_df.drop(['label','is_train'], axis=1).iloc[val_idx]
        y_val = y_train_full[val_idx]
        X_test_lgb = test_df.drop(['label','is_train'], axis=1)

        print("    [LGBM] 提取叶子节点特征...")
        tr_leaf, val_leaf, te_leaf = get_leaf_indices(X_tr_lgb, y_tr, X_val_lgb, X_test_lgb)

        def build_input(df, leaves):
            d = {}
            for name in dense_cols: d[name] = df[name].values.astype(np.float16)
            for name in sparse_cols: d[name] = df[name].values.astype(np.int32)
            leaf_idx = 0
            for block in leaves:
                for i in range(block.shape[1]):
                    d[f'gbdt_tree_{leaf_idx}'] = block[:, i]
                    leaf_idx += 1
            return d

        tr_inp = build_input(X_tr_lgb, tr_leaf)
        val_inp = build_input(X_val_lgb, val_leaf)
        te_inp = build_input(X_test_lgb, te_leaf)

        del X_tr_lgb, X_val_lgb, X_test_lgb, tr_leaf, val_leaf, te_leaf
        gc.collect()

        # --- [消融] 优化点4：动态切换 ScaledDeepFM / BaseDeepFM ---
        ModelClass = ScaledDeepFM if USE_SCALED_FM else BaseDeepFM
        
        print(f"    [DeepFM] 训练 ({ModelClass.__name__})...")
        model = ModelClass(
            linear_feature_columns, dnn_feature_columns,
            task='binary',
            l2_reg_embedding=1e-5, l2_reg_dnn=1e-5,
            dnn_dropout=DNN_DROPOUT,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_use_bn=USE_DNN_BN,  # --- [消融] 优化点5：BatchNorm开关 ---
            device=DEVICE
        )
        model.compile('adam', 'binary_crossentropy', metrics=['auc'])
        es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=0, patience=1, mode='max')

        model.fit(tr_inp, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(val_inp, y_val), callbacks=[es])

        val_pred = model.predict(val_inp, batch_size=BATCH_SIZE*2).flatten()
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(te_inp, batch_size=BATCH_SIZE*2).flatten() / N_FOLDS

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_logloss = log_loss(y_val, val_pred)
        print(f"    >>> Fold AUC: {fold_auc:.6f}, Fold LogLoss: {fold_logloss:.6f}")

        del model, tr_inp, val_inp, te_inp
        torch.cuda.empty_cache()
        gc.collect()

    # 5. 最终评估
    print("\n" + "="*50)
    print("🔥 消融实验结果汇总")
    print("="*50)
    train_auc = roc_auc_score(y_train_full, oof_preds)
    train_logloss = log_loss(y_train_full, oof_preds)
    test_auc = roc_auc_score(y_test_true, test_preds)
    test_logloss = log_loss(y_test_true, test_preds)

    print(f"Train OOF AUC     : {train_auc:.6f}")
    print(f"Train OOF LogLoss : {train_logloss:.6f}")
    print("-" * 20)
    print(f"Test Score AUC    : {test_auc:.6f}")
    print(f"Test Score LogLoss: {test_logloss:.6f}")
    print(f"⏱️ 总耗时: {(time.time()-start_total)/60:.1f} 分钟")

if __name__ == '__main__':
    main()



# USE_MISSING_INDICATOR = True   # 优化点1：是否添加连续特征的缺失/异常值指示特征 (_miss)
# ==================================================
# 🔥 消融实验结果汇总
# ==================================================
# Train OOF AUC     : 0.795785
# Train OOF LogLoss : 0.451294
# --------------------
# Test Score AUC    : 0.797343
# Test Score LogLoss: 0.445028
# ⏱️ 总耗时: 36.2 分钟





# USE_BINNING           = True   # 优化点2：是否对连续特征进行分桶离散化 (_bin)
# ==================================================
# 🔥 消融实验结果汇总
# ==================================================
# Train OOF AUC     : 0.795849
# Train OOF LogLoss : 0.451278
# --------------------
# Test Score AUC    : 0.797439
# Test Score LogLoss: 0.444872
# ⏱️ 总耗时: 35.4 分钟



# USE_TARGET_ENCODING   = True     # 优化点3：是否使用交叉验证的目标编码 (_target_enc)
# ==================================================
# 🔥 消融实验结果汇总
# ==================================================
# Train OOF AUC     : 0.794149
# Train OOF LogLoss : 0.452902
# --------------------
# Test Score AUC    : 0.797589
# Test Score LogLoss: 0.444664
# ⏱️ 总耗时: 38.5 分钟





# USE_SCALED_FM         = True     # 优化点4：是否使用改进版的 ScaledDeepFM (带自适应特征权重)
# ==================================================
# 🔥 消融实验结果汇总
# ==================================================
# Train OOF AUC     : 0.797280
# Train OOF LogLoss : 0.449926
# --------------------
# Test Score AUC    : 0.797409
# Test Score LogLoss: 0.445148
# ⏱️ 总耗时: 36.5 分钟



# USE_DNN_BN            = True     # 优化点5：DNN 层是否使用 Batch Normalization
# ==================================================
# 🔥 消融实验结果汇总
# ==================================================
# Train OOF AUC     : 0.796174
# Train OOF LogLoss : 0.450805
# --------------------
# Test Score AUC    : 0.797377
# Test Score LogLoss: 0.444988
# ⏱️ 总耗时: 33.5 分钟



# USE_MISSING_INDICATOR = False
# USE_BINNING = True
# USE_TARGET_ENCODING = True
# USE_SCALED_FM = False
# USE_DNN_BN = True
# ==================================================
# 🔥 消融实验结果汇总
# ==================================================
# Train OOF AUC     : 0.793847
# Train OOF LogLoss : 0.452792
# --------------------
# Test Score AUC    : 0.798223
# Test Score LogLoss: 0.444116
# ⏱️ 总耗时: 46.0 分钟