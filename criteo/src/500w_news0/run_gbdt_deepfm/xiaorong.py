# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import lightgbm as lgb
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler
# from sklearn.metrics import log_loss, roc_auc_score
# from sklearn.model_selection import StratifiedKFold
# from deepctr_torch.inputs import SparseFeat, DenseFeat
# from deepctr_torch.models import DeepFM as BaseDeepFM
# from deepctr_torch.callbacks import EarlyStopping
# import time
# import gc
# import warnings

# warnings.filterwarnings('ignore')

# # ================= ⚙️ 核心优化点消融实验配置区 =================
# # 通过修改以下的 True / False 进行消融实验
# USE_MISSING_INDICATOR = False   # 优化点1：是否添加连续特征的缺失/异常值指示特征 (_miss)
# USE_BINNING           = True  # 优化点2：是否对连续特征进行分桶离散化 (_bin)
# USE_TARGET_ENCODING   = True     # 优化点3：是否使用交叉验证的目标编码 (_target_enc)
# USE_SCALED_FM         = False     # 优化点4：是否使用改进版的 ScaledDeepFM (带自适应特征权重)
# USE_DNN_BN            = True     # 优化点5：DNN 层是否使用 Batch Normalization

# # ================= 🔧 基础参数配置 =================
# TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
# TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# N_FOLDS = 5
# SEED = 2026

# EMBEDDING_DIM = 10
# BATCH_SIZE = 4096
# EPOCHS = 2
# DNN_DROPOUT = 0.5
# DNN_HIDDEN_UNITS = (512, 256)

# GBDT_NUM_LEAVES = 64
# GBDT_TREES = 100
# FREQ_FILTER_THRESHOLD = 5
# N_BINS = 16
# TARGET_ENCODING_SMOOTH = 10

# COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
# SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]
# DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]

# # ================= 🌟 改进版模型定义 =================
# class ScaledDeepFM(BaseDeepFM):
#     """带自适应Field权重的Scaled FM"""
#     def __init__(self, linear_feature_columns, dnn_feature_columns, **kwargs):
#         super().__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
#         self.n_fields = len(self.embedding_dict)
#         self.field_scale = nn.Parameter(torch.ones(self.n_fields))

#     def fm(self, fm_input):
#         scale = self.field_scale.to(fm_input.device).view(1, -1, 1)
#         scaled = fm_input * scale
#         square_of_sum = torch.sum(scaled, dim=1) ** 2
#         sum_of_square = torch.sum(scaled ** 2, dim=1)
#         cross_term = 0.5 * (square_of_sum - sum_of_square)
#         return torch.sum(cross_term, dim=1, keepdim=True)

# # ================= 📊 数据处理 (支持动态开关) =================
# def process_data_ablation():
#     print(f"[{time.strftime('%H:%M:%S')}] 读取数据...")
#     train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
#     test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)

#     y_train = train_df['label'].values
#     y_test = test_df['label'].values
#     train_df['is_train'] = 1
#     test_df['is_train'] = 0

#     train_df[DENSE_FEATURES] = train_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')
#     test_df[DENSE_FEATURES] = test_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')

#     final_sparse_cols = SPARSE_FEATURES.copy()
#     final_dense_cols = DENSE_FEATURES.copy()

#     # --- [消融] 优化点1：缺失值指示 ---
#     if USE_MISSING_INDICATOR:
#         print(f"[{time.strftime('%H:%M:%S')}] 添加缺失值指示特征...")
#         for col in DENSE_FEATURES:
#             miss_col = col + '_miss'
#             train_df[miss_col] = ((train_df[col] < 0) | (train_df[col].isna())).astype(int).astype(str)
#             test_df[miss_col] = ((test_df[col] < 0) | (test_df[col].isna())).astype(int).astype(str)
#             final_sparse_cols.append(miss_col)

#     # 填充与截断
#     train_df[DENSE_FEATURES] = train_df[DENSE_FEATURES].fillna(0).clip(lower=0)
#     test_df[DENSE_FEATURES] = test_df[DENSE_FEATURES].fillna(0).clip(lower=0)

#     # --- [消融] 优化点2：连续特征分桶 ---
#     if USE_BINNING:
#         print(f"[{time.strftime('%H:%M:%S')}] 连续特征分桶离散化...")
#         BIN_FEATURES = [f'{col}_bin' for col in DENSE_FEATURES]
#         est = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')
        
#         train_bins = est.fit_transform(train_df[DENSE_FEATURES])
#         test_bins = est.transform(test_df[DENSE_FEATURES])

#         train_df[BIN_FEATURES] = train_bins.astype(int).astype(str)
#         test_df[BIN_FEATURES] = test_bins.astype(int).astype(str)
#         final_sparse_cols.extend(BIN_FEATURES)

#     # 数值特征 Log + 归一化 (无论哪个版本都做)
#     print(f"[{time.strftime('%H:%M:%S')}] 数值特征 log1p & 归一化...")
#     for col in DENSE_FEATURES:
#         train_df[col] = np.log1p(train_df[col])
#         test_df[col] = np.log1p(test_df[col])

#     mms = MinMaxScaler()
#     train_df[DENSE_FEATURES] = mms.fit_transform(train_df[DENSE_FEATURES])
#     test_df[DENSE_FEATURES] = mms.transform(test_df[DENSE_FEATURES])

#     # 类别特征低频过滤
#     print(f"[{time.strftime('%H:%M:%S')}] 类别特征低频过滤...")
#     for feat in SPARSE_FEATURES: # 只过滤原始稀疏特征
#         train_df[feat] = train_df[feat].fillna('-1').astype(str)
#         test_df[feat] = test_df[feat].fillna('-1').astype(str)

#         cnt = train_df[feat].value_counts()
#         valid = set(cnt[cnt >= FREQ_FILTER_THRESHOLD].index)
#         train_df.loc[~train_df[feat].isin(valid), feat] = 'UNK'
#         test_df.loc[~test_df[feat].isin(valid), feat] = 'UNK'

#     return train_df, test_df, y_train, y_test, final_sparse_cols, final_dense_cols

# # ================= 🎯 目标编码函数 =================
# def target_encoding_cv(train_df, test_df, cat_cols, target, n_folds=5, smooth=10):
#     print(f"[{time.strftime('%H:%M:%S')}] 执行交叉验证目标编码...")
#     global_mean = train_df[target].mean()
#     skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
#     te_features = []

#     for col in cat_cols:
#         enc_col = col + '_target_enc'
#         te_features.append(enc_col)
#         train_df[enc_col] = np.nan
#         test_df[enc_col] = np.nan

#         for tr_idx, val_idx in skf.split(train_df, train_df[target]):
#             tr_means = train_df.iloc[tr_idx].groupby(col)[target].agg(['count', 'mean']).reset_index()
#             tr_means['enc'] = (tr_means['count'] * tr_means['mean'] + global_mean * smooth) / (tr_means['count'] + smooth)
#             val_map = tr_means.set_index(col)['enc'].to_dict()
#             train_df.loc[val_idx, enc_col] = train_df.loc[val_idx, col].map(val_map)

#         train_df[enc_col].fillna(global_mean, inplace=True)

#         full_means = train_df.groupby(col)[target].agg(['count', 'mean']).reset_index()
#         full_means['enc'] = (full_means['count'] * full_means['mean'] + global_mean * smooth) / (full_means['count'] + smooth)
#         test_map = full_means.set_index(col)['enc'].to_dict()
#         test_df[enc_col] = test_df[col].map(test_map)
#         test_df[enc_col].fillna(global_mean, inplace=True)

#     return train_df, test_df, te_features

# # ================= 🌲 GBDT 提取特征 =================
# def get_leaf_indices(X_train, y_train, X_val, X_test):
#     lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
#     lgb_val = lgb.Dataset(X_val, reference=lgb_train, free_raw_data=False)
#     params = {
#         'objective': 'binary', 'metric': 'auc', 'num_leaves': GBDT_NUM_LEAVES,
#         'learning_rate': 0.1, 'bagging_fraction': 0.8, 'feature_fraction': 0.8,
#         'verbose': -1, 'n_jobs': -1
#     }
#     model = lgb.train(params, lgb_train, num_boost_round=GBDT_TREES)

#     batch_size, n_trees, block_size = 100000, GBDT_TREES, 20
#     n_blocks = (n_trees + block_size - 1) // block_size

#     def predict_leaf_blocks(X):
#         n_samples = len(X)
#         block_lists = [[] for _ in range(n_blocks)]
#         for start in range(0, n_samples, batch_size):
#             end = min(start + batch_size, n_samples)
#             X_batch = X.iloc[start:end] if hasattr(X, 'iloc') else X[start:end]
#             leaf_batch = model.predict(X_batch, pred_leaf=True)
#             for b in range(n_blocks):
#                 tree_start, tree_end = b * block_size, min((b+1) * block_size, n_trees)
#                 block_lists[b].append(leaf_batch[:, tree_start:tree_end])
#         return [np.vstack(block_lists[b]).astype(np.uint8) if block_lists[b] else np.empty((0, block_size), dtype=np.uint8) for b in range(n_blocks)]

#     return predict_leaf_blocks(X_train), predict_leaf_blocks(X_val), predict_leaf_blocks(X_test)

# # ================= 🚀 主函数 =================
# def main():
#     print("="*50)
#     print("🔬 核心优化点消融实验配置状态:")
#     print(f"[-] 缺失值指示 (USE_MISSING_INDICATOR): {USE_MISSING_INDICATOR}")
#     print(f"[-] 连续特征分桶 (USE_BINNING)        : {USE_BINNING}")
#     print(f"[-] 目标编码 (USE_TARGET_ENCODING)    : {USE_TARGET_ENCODING}")
#     print(f"[-] Scaled FM机制 (USE_SCALED_FM)     : {USE_SCALED_FM}")
#     print(f"[-] DNN Batch Norm (USE_DNN_BN)       : {USE_DNN_BN}")
#     print("="*50)

#     start_total = time.time()

#     # 1. 基础数据处理 (动态)
#     train_df, test_df, y_train_full, y_test_true, sparse_cols, dense_cols = process_data_ablation()

#     # --- [消融] 优化点3：目标编码 ---
#     if USE_TARGET_ENCODING:
#         # 针对原始稀疏特征和分桶特征进行 Target Encoding (不含缺失指示列)
#         cat_cols_for_te = SPARSE_FEATURES.copy()
#         if USE_BINNING:
#             cat_cols_for_te.extend([f'{col}_bin' for col in DENSE_FEATURES])
        
#         train_df, test_df, te_features = target_encoding_cv(
#             train_df, test_df, cat_cols_for_te, target='label',
#             n_folds=N_FOLDS, smooth=TARGET_ENCODING_SMOOTH
#         )
#         dense_cols.extend(te_features) # 把编码后的特征当做连续值输入 DNN

#     # 2. Label Encoding (所有稀疏特征统一编码)
#     print(f"[{time.strftime('%H:%M:%S')}] Label Encoding...")
#     for feat in sparse_cols:
#         le = LabelEncoder()
#         train_vals = train_df[feat].values
#         test_vals = test_df[feat].values
#         # 合并 fit 防止 unseen labels 报错
#         le.fit(np.concatenate([train_vals, test_vals])) 
#         train_df[feat] = le.transform(train_vals).astype(np.int32)
#         test_df[feat] = le.transform(test_vals).astype(np.int32)

#     # 3. 构建 Feature Columns
#     vocab_sizes = {feat: train_df[feat].max() + 1 for feat in sparse_cols}
    
#     fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocab_sizes[feat], embedding_dim=EMBEDDING_DIM) for feat in sparse_cols]
#     dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_cols]
#     gbdt_leaf_cols = [SparseFeat(f'gbdt_tree_{i}', vocabulary_size=GBDT_NUM_LEAVES+1, embedding_dim=EMBEDDING_DIM) for i in range(GBDT_TREES)]
    
#     linear_feature_columns = fixlen_feature_columns + gbdt_leaf_cols + dense_feature_columns
#     dnn_feature_columns = linear_feature_columns

#     # 4. 交叉验证训练
#     oof_preds = np.zeros(len(train_df), dtype=np.float32)
#     test_preds = np.zeros(len(test_df), dtype=np.float32)
#     folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

#     for fold, (tr_idx, val_idx) in enumerate(folds.split(train_df.drop(['label','is_train'], axis=1), y_train_full)):
#         print(f"\n>>> Fold {fold+1}/{N_FOLDS}")

#         X_tr_lgb = train_df.drop(['label','is_train'], axis=1).iloc[tr_idx]
#         y_tr = y_train_full[tr_idx]
#         X_val_lgb = train_df.drop(['label','is_train'], axis=1).iloc[val_idx]
#         y_val = y_train_full[val_idx]
#         X_test_lgb = test_df.drop(['label','is_train'], axis=1)

#         print("    [LGBM] 提取叶子节点特征...")
#         tr_leaf, val_leaf, te_leaf = get_leaf_indices(X_tr_lgb, y_tr, X_val_lgb, X_test_lgb)

#         def build_input(df, leaves):
#             d = {}
#             for name in dense_cols: d[name] = df[name].values.astype(np.float16)
#             for name in sparse_cols: d[name] = df[name].values.astype(np.int32)
#             leaf_idx = 0
#             for block in leaves:
#                 for i in range(block.shape[1]):
#                     d[f'gbdt_tree_{leaf_idx}'] = block[:, i]
#                     leaf_idx += 1
#             return d

#         tr_inp = build_input(X_tr_lgb, tr_leaf)
#         val_inp = build_input(X_val_lgb, val_leaf)
#         te_inp = build_input(X_test_lgb, te_leaf)

#         del X_tr_lgb, X_val_lgb, X_test_lgb, tr_leaf, val_leaf, te_leaf
#         gc.collect()

#         # --- [消融] 优化点4：动态切换 ScaledDeepFM / BaseDeepFM ---
#         ModelClass = ScaledDeepFM if USE_SCALED_FM else BaseDeepFM
        
#         print(f"    [DeepFM] 训练 ({ModelClass.__name__})...")
#         model = ModelClass(
#             linear_feature_columns, dnn_feature_columns,
#             task='binary',
#             l2_reg_embedding=1e-5, l2_reg_dnn=1e-5,
#             dnn_dropout=DNN_DROPOUT,
#             dnn_hidden_units=DNN_HIDDEN_UNITS,
#             dnn_use_bn=USE_DNN_BN,  # --- [消融] 优化点5：BatchNorm开关 ---
#             device=DEVICE
#         )
#         model.compile('adam', 'binary_crossentropy', metrics=['auc'])
#         es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=0, patience=1, mode='max')

#         model.fit(tr_inp, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(val_inp, y_val), callbacks=[es])

#         val_pred = model.predict(val_inp, batch_size=BATCH_SIZE*2).flatten()
#         oof_preds[val_idx] = val_pred
#         test_preds += model.predict(te_inp, batch_size=BATCH_SIZE*2).flatten() / N_FOLDS

#         fold_auc = roc_auc_score(y_val, val_pred)
#         fold_logloss = log_loss(y_val, val_pred)
#         print(f"    >>> Fold AUC: {fold_auc:.6f}, Fold LogLoss: {fold_logloss:.6f}")

#         del model, tr_inp, val_inp, te_inp
#         torch.cuda.empty_cache()
#         gc.collect()

#     # 5. 最终评估
#     print("\n" + "="*50)
#     print("🔥 消融实验结果汇总")
#     print("="*50)
#     train_auc = roc_auc_score(y_train_full, oof_preds)
#     train_logloss = log_loss(y_train_full, oof_preds)
#     test_auc = roc_auc_score(y_test_true, test_preds)
#     test_logloss = log_loss(y_test_true, test_preds)

#     print(f"Train OOF AUC     : {train_auc:.6f}")
#     print(f"Train OOF LogLoss : {train_logloss:.6f}")
#     print("-" * 20)
#     print(f"Test Score AUC    : {test_auc:.6f}")
#     print(f"Test Score LogLoss: {test_logloss:.6f}")
#     print(f"⏱️ 总耗时: {(time.time()-start_total)/60:.1f} 分钟")

# if __name__ == '__main__':
#     main()



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

# ================= ⚙️ 消融实验开关 =================
USE_MISSING_INDICATOR = False   # 优化点1：连续特征缺失/异常值指示特征 (_miss)
USE_BINNING           = False    # 优化点2：连续特征分桶离散化 (_bin)
USE_TARGET_ENCODING   = False    # 优化点3：交叉验证目标编码 (_target_enc)
USE_SCALED_FM         = True   # 优化点4：ScaledDeepFM (自适应 field 权重)
USE_DNN_BN            = False    # 优化点5：DNN 是否使用 BatchNorm

# ================= 🔧 基础参数 =================
TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH  = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

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
    """带自适应 Field 权重的 Scaled FM"""
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


# ================= 📥 读取原始数据 =================
def load_raw_data():
    print(f"[{time.strftime('%H:%M:%S')}] 正在读取原始数据...")
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
    test_df  = pd.read_csv(TEST_PATH,  sep='\t', names=COL_NAMES, header=None)

    train_df['is_train'] = 1
    test_df['is_train'] = 0
    return train_df, test_df


# ================= 🧰 工具函数 =================
def get_valid_stratified_splits(y, max_splits=5):
    """保证 StratifiedKFold 的 n_splits 合法"""
    y = np.asarray(y).astype(int)
    class_counts = np.bincount(y)
    class_counts = class_counts[class_counts > 0]
    if len(class_counts) < 2:
        return None
    min_class_count = class_counts.min()
    if min_class_count < 2:
        return None
    return min(max_splits, min_class_count)


def safe_label_encode(train_series, val_series, test_series):
    """
    只在 train 上 fit；
    val/test 中未见类别统一映射到 __UNK__，避免 train+test 一起 fit
    """
    train_vals = train_series.astype(str).values
    val_vals   = val_series.astype(str).values
    test_vals  = test_series.astype(str).values

    le = LabelEncoder()
    le.fit(np.concatenate([train_vals, np.array(['__UNK__'], dtype=object)]))

    known_classes = set(le.classes_)

    val_vals  = np.array([x if x in known_classes else '__UNK__' for x in val_vals], dtype=object)
    test_vals = np.array([x if x in known_classes else '__UNK__' for x in test_vals], dtype=object)

    train_encoded = le.transform(train_vals).astype(np.int32)
    val_encoded   = le.transform(val_vals).astype(np.int32)
    test_encoded  = le.transform(test_vals).astype(np.int32)

    vocab_size = len(le.classes_)
    return train_encoded, val_encoded, test_encoded, vocab_size


# ================= 🎯 Fold 内 Target Encoding =================
def add_target_encoding_in_fold(train_df, val_df, test_df, cat_cols, target_col='label'):
    """
    [更正核心]
    1) 只使用 outer-fold 的训练子集拟合统计量
    2) train 子集自身使用 inner OOF target encoding，避免行内泄漏
    3) val/test 使用 outer-fold train 的全量统计映射
    """
    print(f"[{time.strftime('%H:%M:%S')}] Fold 内执行 Target Encoding...")
    te_features = []
    global_mean = train_df[target_col].mean()
    y_train = train_df[target_col].values

    inner_splits = get_valid_stratified_splits(y_train, N_FOLDS)

    for col in cat_cols:
        enc_col = col + '_target_enc'
        te_features.append(enc_col)

        train_df[enc_col] = np.nan

        # --- train: inner OOF target encoding ---
        if inner_splits is not None and inner_splits >= 2:
            inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)

            for inner_tr_idx, inner_val_idx in inner_skf.split(train_df, y_train):
                tr_part = train_df.iloc[inner_tr_idx][[col, target_col]].copy()

                stats = tr_part.groupby(col)[target_col].agg(['count', 'mean'])
                stats['enc'] = (
                    stats['count'] * stats['mean'] + TARGET_ENCODING_SMOOTH * global_mean
                ) / (stats['count'] + TARGET_ENCODING_SMOOTH)

                enc_map = stats['enc'].to_dict()
                val_index = train_df.index[inner_val_idx]
                train_df.loc[val_index, enc_col] = train_df.loc[val_index, col].map(enc_map)

        train_df[enc_col] = train_df[enc_col].fillna(global_mean).astype(np.float32)

        # --- val/test: outer train full stats ---
        full_stats = train_df.groupby(col)[target_col].agg(['count', 'mean'])
        full_stats['enc'] = (
            full_stats['count'] * full_stats['mean'] + TARGET_ENCODING_SMOOTH * global_mean
        ) / (full_stats['count'] + TARGET_ENCODING_SMOOTH)

        full_map = full_stats['enc'].to_dict()

        val_df[enc_col] = val_df[col].map(full_map).fillna(global_mean).astype(np.float32)
        test_df[enc_col] = test_df[col].map(full_map).fillna(global_mean).astype(np.float32)

    return train_df, val_df, test_df, te_features


# ================= 📊 Fold 内预处理（核心修正版） =================
def preprocess_one_fold(train_fold_df, val_fold_df, test_df):
    """
    [更正核心]
    所有需要 fit 的预处理都只在当前 outer-fold 的 train 子集上拟合：
    - KBinsDiscretizer.fit
    - MinMaxScaler.fit
    - 低频过滤统计
    - LabelEncoder.fit
    - Target Encoding
    """
    tr_df = train_fold_df.copy().reset_index(drop=True)
    val_df = val_fold_df.copy().reset_index(drop=True)
    te_df = test_df.copy().reset_index(drop=True)

    final_sparse_cols = SPARSE_FEATURES.copy()
    final_dense_cols = DENSE_FEATURES.copy()

    # ---------- 1) 连续特征转数值 ----------
    for df in [tr_df, val_df, te_df]:
        df[DENSE_FEATURES] = df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')

    # ---------- 2) 缺失/异常值指示 ----------
    if USE_MISSING_INDICATOR:
        print(f"[{time.strftime('%H:%M:%S')}] 添加缺失/异常值指示特征...")
        for col in DENSE_FEATURES:
            miss_col = col + '_miss'
            tr_df[miss_col] = ((tr_df[col].isna()) | (tr_df[col] < 0)).astype(int).astype(str)
            val_df[miss_col] = ((val_df[col].isna()) | (val_df[col] < 0)).astype(int).astype(str)
            te_df[miss_col] = ((te_df[col].isna()) | (te_df[col] < 0)).astype(int).astype(str)
            final_sparse_cols.append(miss_col)

    # ---------- 3) 缺失填充 + 截断 ----------
    for df in [tr_df, val_df, te_df]:
        df[DENSE_FEATURES] = df[DENSE_FEATURES].fillna(0).clip(lower=0)

    # ---------- 4) 连续特征分桶（只在 train fold 上 fit） ----------
    bin_features = []
    if USE_BINNING:
        print(f"[{time.strftime('%H:%M:%S')}] 连续特征分桶离散化（fold 内 fit）...")
        bin_features = [f'{col}_bin' for col in DENSE_FEATURES]
        est = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')

        tr_bins = est.fit_transform(tr_df[DENSE_FEATURES])
        val_bins = est.transform(val_df[DENSE_FEATURES])
        te_bins = est.transform(te_df[DENSE_FEATURES])

        tr_df[bin_features] = tr_bins.astype(int).astype(str)
        val_df[bin_features] = val_bins.astype(int).astype(str)
        te_df[bin_features] = te_bins.astype(int).astype(str)

        final_sparse_cols.extend(bin_features)

    # ---------- 5) log1p + MinMaxScaler（只在 train fold 上 fit） ----------
    print(f"[{time.strftime('%H:%M:%S')}] 数值特征 log1p + 归一化（fold 内 fit）...")
    for df in [tr_df, val_df, te_df]:
        for col in DENSE_FEATURES:
            df[col] = np.log1p(df[col])

    scaler = MinMaxScaler()
    tr_df[DENSE_FEATURES] = scaler.fit_transform(tr_df[DENSE_FEATURES]).astype(np.float32)
    val_df[DENSE_FEATURES] = scaler.transform(val_df[DENSE_FEATURES]).astype(np.float32)
    te_df[DENSE_FEATURES] = scaler.transform(te_df[DENSE_FEATURES]).astype(np.float32)

    # ---------- 6) 原始类别特征低频过滤（只根据 train fold 统计） ----------
    print(f"[{time.strftime('%H:%M:%S')}] 类别特征低频过滤（fold 内统计）...")
    for feat in SPARSE_FEATURES:
        for df in [tr_df, val_df, te_df]:
            df[feat] = df[feat].fillna('-1').astype(str)

        cnt = tr_df[feat].value_counts()
        valid_cats = set(cnt[cnt >= FREQ_FILTER_THRESHOLD].index)

        tr_df.loc[~tr_df[feat].isin(valid_cats), feat] = 'Rare'
        val_df.loc[~val_df[feat].isin(valid_cats), feat] = 'Rare'
        te_df.loc[~te_df[feat].isin(valid_cats), feat] = 'Rare'

    # ---------- 7) Target Encoding（fold 内严格生成） ----------
    if USE_TARGET_ENCODING:
        cat_cols_for_te = SPARSE_FEATURES.copy()
        if USE_BINNING:
            cat_cols_for_te.extend(bin_features)

        tr_df, val_df, te_df, te_features = add_target_encoding_in_fold(
            tr_df, val_df, te_df, cat_cols_for_te, target_col='label'
        )
        final_dense_cols.extend(te_features)

    # ---------- 8) 所有稀疏特征统一 Label Encoding（只在 train fold 上 fit） ----------
    print(f"[{time.strftime('%H:%M:%S')}] 稀疏特征 Label Encoding（fold 内 fit）...")
    vocab_sizes = {}
    for feat in final_sparse_cols:
        tr_enc, val_enc, te_enc, vocab_size = safe_label_encode(
            tr_df[feat], val_df[feat], te_df[feat]
        )
        tr_df[feat] = tr_enc
        val_df[feat] = val_enc
        te_df[feat] = te_enc
        vocab_sizes[feat] = vocab_size

    return tr_df, val_df, te_df, final_sparse_cols, final_dense_cols, vocab_sizes


# ================= 🌲 GBDT 提取叶子特征 =================
def get_leaf_indices(X_train, y_train, X_val, X_test):
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': GBDT_NUM_LEAVES,
        'learning_rate': 0.1,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': -1,
        'seed': SEED
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
            X_batch = X.iloc[start:end] if hasattr(X, 'iloc') else X[start:end]
            leaf_batch = model.predict(X_batch, pred_leaf=True)

            for b in range(n_blocks):
                tree_start = b * block_size
                tree_end = min((b + 1) * block_size, n_trees)
                block_lists[b].append(leaf_batch[:, tree_start:tree_end])

        result = []
        for b in range(n_blocks):
            if len(block_lists[b]) == 0:
                result.append(np.empty((0, block_size), dtype=np.uint8))
            else:
                result.append(np.vstack(block_lists[b]).astype(np.uint8))
        return result

    tr_leaf = predict_leaf_blocks(X_train)
    val_leaf = predict_leaf_blocks(X_val)
    te_leaf = predict_leaf_blocks(X_test)
    return tr_leaf, val_leaf, te_leaf


# ================= 🚀 主函数 =================
def main():
    print("=" * 60)
    print("🔬 消融实验配置")
    print(f"[-] 缺失值指示 (USE_MISSING_INDICATOR): {USE_MISSING_INDICATOR}")
    print(f"[-] 连续特征分桶 (USE_BINNING)        : {USE_BINNING}")
    print(f"[-] 目标编码 (USE_TARGET_ENCODING)    : {USE_TARGET_ENCODING}")
    print(f"[-] Scaled FM机制 (USE_SCALED_FM)     : {USE_SCALED_FM}")
    print(f"[-] DNN BatchNorm (USE_DNN_BN)        : {USE_DNN_BN}")
    print("=" * 60)

    start_total = time.time()

    # [更正] 这里只读取原始数据，不在全局先做 fit 型预处理
    train_raw_df, test_raw_df = load_raw_data()
    y_train_full = train_raw_df['label'].values
    y_test_true = test_raw_df['label'].values

    oof_preds = np.zeros(len(train_raw_df), dtype=np.float32)
    test_preds = np.zeros(len(test_raw_df), dtype=np.float32)

    outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(outer_skf.split(train_raw_df, y_train_full)):
        print(f"\n{'>' * 15} Fold {fold + 1}/{N_FOLDS} {'<' * 15}")

        train_fold_raw = train_raw_df.iloc[tr_idx].copy()
        val_fold_raw = train_raw_df.iloc[val_idx].copy()

        # [更正] 所有预处理都在 fold 内拟合
        tr_df, val_df, te_df, sparse_cols, dense_cols, vocab_sizes = preprocess_one_fold(
            train_fold_raw, val_fold_raw, test_raw_df
        )

        # 构建给 LGBM 的输入
        X_tr_lgb = tr_df.drop(['label', 'is_train'], axis=1)
        y_tr = tr_df['label'].values
        X_val_lgb = val_df.drop(['label', 'is_train'], axis=1)
        y_val = val_df['label'].values
        X_te_lgb = te_df.drop(['label', 'is_train'], axis=1)

        # 提取 GBDT 叶子特征
        print(f"[{time.strftime('%H:%M:%S')}] [LGBM] 提取叶子节点特征...")
        tr_leaf, val_leaf, te_leaf = get_leaf_indices(X_tr_lgb, y_tr, X_val_lgb, X_te_lgb)

        # 每个 fold 单独构建 Feature Columns（因为 vocab 可能不同）
        fixlen_feature_columns = [
            SparseFeat(feat, vocabulary_size=vocab_sizes[feat], embedding_dim=EMBEDDING_DIM)
            for feat in sparse_cols
        ]
        dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_cols]
        gbdt_leaf_cols = [
            SparseFeat(f'gbdt_tree_{i}', vocabulary_size=GBDT_NUM_LEAVES + 1, embedding_dim=EMBEDDING_DIM)
            for i in range(GBDT_TREES)
        ]

        linear_feature_columns = fixlen_feature_columns + gbdt_leaf_cols + dense_feature_columns
        dnn_feature_columns = linear_feature_columns

        def build_input(df, leaf_blocks):
            d = {}

            for name in dense_cols:
                d[name] = df[name].values.astype(np.float32)

            for name in sparse_cols:
                d[name] = df[name].values.astype(np.int32)

            leaf_idx = 0
            for block in leaf_blocks:
                for i in range(block.shape[1]):
                    d[f'gbdt_tree_{leaf_idx}'] = block[:, i]
                    leaf_idx += 1
            return d

        tr_inp = build_input(tr_df, tr_leaf)
        val_inp = build_input(val_df, val_leaf)
        te_inp = build_input(te_df, te_leaf)

        del X_tr_lgb, X_val_lgb, X_te_lgb, tr_leaf, val_leaf, te_leaf
        gc.collect()

        # 动态选择模型
        ModelClass = ScaledDeepFM if USE_SCALED_FM else BaseDeepFM

        print(f"[{time.strftime('%H:%M:%S')}] [DeepFM] 训练 {ModelClass.__name__}...")
        model = ModelClass(
            linear_feature_columns,
            dnn_feature_columns,
            task='binary',
            l2_reg_embedding=1e-5,
            l2_reg_dnn=1e-5,
            dnn_dropout=DNN_DROPOUT,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_use_bn=USE_DNN_BN,
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

        val_pred = model.predict(val_inp, batch_size=BATCH_SIZE * 2).flatten()
        test_pred = model.predict(te_inp, batch_size=BATCH_SIZE * 2).flatten()

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / N_FOLDS

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_logloss = log_loss(y_val, val_pred)
        print(f"Fold {fold + 1} >>> AUC: {fold_auc:.6f}, LogLoss: {fold_logloss:.6f}")

        del model, tr_inp, val_inp, te_inp, tr_df, val_df, te_df
        torch.cuda.empty_cache()
        gc.collect()

    # 最终结果
    print("\n" + "=" * 60)
    print("🔥 消融实验最终结果")
    print("=" * 60)

    train_auc = roc_auc_score(y_train_full, oof_preds)
    train_logloss = log_loss(y_train_full, oof_preds)
    test_auc = roc_auc_score(y_test_true, test_preds)
    test_logloss = log_loss(y_test_true, test_preds)

    print(f"Train OOF AUC     : {train_auc:.6f}")
    print(f"Train OOF LogLoss : {train_logloss:.6f}")
    print("-" * 25)
    print(f"Test Score AUC    : {test_auc:.6f}")
    print(f"Test Score LogLoss: {test_logloss:.6f}")
    print(f"⏱️ 总耗时: {(time.time() - start_total) / 60:.1f} 分钟")


if __name__ == '__main__':
    main()

# ============================================================
# 🔬 消融实验配置
# [-] 缺失值指示 (USE_MISSING_INDICATOR): True
# [-] 连续特征分桶 (USE_BINNING)        : True
# [-] 目标编码 (USE_TARGET_ENCODING)    : True
# [-] Scaled FM机制 (USE_SCALED_FM)     : True
# [-] DNN BatchNorm (USE_DNN_BN)        : True
# ============================================================
# ============================================================
# 🔥 消融实验最终结果
# ============================================================
# Train OOF AUC     : 0.797787
# Train OOF LogLoss : 0.448982
# -------------------------
# Test Score AUC    : 0.798956
# Test Score LogLoss: 0.443298
# ⏱️ 总耗时: 91.0 分钟







# ============================================================
# 🔬 消融实验配置
# [-] 缺失值指示 (USE_MISSING_INDICATOR): False
# [-] 连续特征分桶 (USE_BINNING)        : False
# [-] 目标编码 (USE_TARGET_ENCODING)    : False
# [-] Scaled FM机制 (USE_SCALED_FM)     : False
# [-] DNN BatchNorm (USE_DNN_BN)        : False
# ============================================================
# ============================================================
# 🔥 消融实验最终结果
# ============================================================
# Train OOF AUC     : 0.796195
# Train OOF LogLoss : 0.450809
# -------------------------
# Test Score AUC    : 0.797426
# Test Score LogLoss: 0.444775
# ⏱️ 总耗时: 37.0 分钟






# ============================================================
# 🔬 消融实验配置
# [-] 缺失值指示 (USE_MISSING_INDICATOR): True
# [-] 连续特征分桶 (USE_BINNING)        : False
# [-] 目标编码 (USE_TARGET_ENCODING)    : False
# [-] Scaled FM机制 (USE_SCALED_FM)     : False
# [-] DNN BatchNorm (USE_DNN_BN)        : False
# ============================================================
# ============================================================
# 🔥 消融实验最终结果
# ============================================================
# Train OOF AUC     : 0.796410
# Train OOF LogLoss : 0.450450
# -------------------------
# Test Score AUC    : 0.797770
# Test Score LogLoss: 0.444447
# ⏱️ 总耗时: 42.8 分钟





# ============================================================
# 🔬 消融实验配置
# [-] 缺失值指示 (USE_MISSING_INDICATOR): False
# [-] 连续特征分桶 (USE_BINNING)        : True
# [-] 目标编码 (USE_TARGET_ENCODING)    : False
# [-] Scaled FM机制 (USE_SCALED_FM)     : False
# [-] DNN BatchNorm (USE_DNN_BN)        : False
# ============================================================
# ============================================================
# 🔥 消融实验最终结果
# ============================================================
# Train OOF AUC     : 0.796227
# Train OOF LogLoss : 0.450810
# -------------------------
# Test Score AUC    : 0.797422
# Test Score LogLoss: 0.444855
# ⏱️ 总耗时: 41.1 分钟















# ============================================================
# 🔬 消融实验配置
# [-] 缺失值指示 (USE_MISSING_INDICATOR): False
# [-] 连续特征分桶 (USE_BINNING)        : False
# [-] 目标编码 (USE_TARGET_ENCODING)    : True
# [-] Scaled FM机制 (USE_SCALED_FM)     : False
# [-] DNN BatchNorm (USE_DNN_BN)        : False
# ============================================================
# ============================================================
# 🔥 消融实验最终结果
# ============================================================
# Train OOF AUC     : 0.795837
# Train OOF LogLoss : 0.451414
# -------------------------
# Test Score AUC    : 0.798033
# Test Score LogLoss: 0.444288
# ⏱️ 总耗时: 61.4 分钟