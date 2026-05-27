# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import lightgbm as lgb
# import random
# import os
# import gc
# import warnings
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler
# from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc as sk_auc
# from sklearn.model_selection import StratifiedKFold

# from deepctr_torch.inputs import SparseFeat, DenseFeat
# from deepctr_torch.models import DeepFM as BaseDeepFM
# from deepctr_torch.callbacks import EarlyStopping

# warnings.filterwarnings('ignore')

# # 修复 Matplotlib 中文显示乱码问题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用黑体
# plt.rcParams['axes.unicode_minus'] = False

# # ================= ⚙️ 【一键切换版本】只改这里 =================
# # 版本1：基础版 Base —— 全部关闭
# # USE_MISSING_INDICATOR = False
# # USE_BINNING           = False
# # USE_TARGET_ENCODING   = False
# # USE_SCALED_FM         = False
# # USE_DNN_BN            = False

# # 版本2：最终版 Final —— 除分桶外全开
# USE_MISSING_INDICATOR = True
# USE_BINNING           = False
# USE_TARGET_ENCODING   = True
# USE_SCALED_FM         = True
# USE_DNN_BN            = True
# # ============================================================

# # ================= 🔧 基础参数 =================
# TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
# TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"
# SAVE_BASE_PATH = r"D:\code\Python\criteo"

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

# # ================= 🌱 固定随机种子 =================
# def seed_everything(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

# # ================= 🌟 Scaled FM =================
# class ScaledFM(nn.Module):
#     def __init__(self, n_fields):
#         super().__init__()
#         self.n_fields = n_fields
#         self.field_scale = nn.Parameter(torch.ones(n_fields))

#     def forward(self, fm_input):
#         scale = self.field_scale.view(1, -1, 1).to(fm_input.device)
#         scaled = fm_input * scale
#         square_of_sum = torch.sum(scaled, dim=1) ** 2
#         sum_of_square = torch.sum(scaled ** 2, dim=1)
#         cross_term = 0.5 * (square_of_sum - sum_of_square)
#         return torch.sum(cross_term, dim=1, keepdim=True)

# class ScaledDeepFM(BaseDeepFM):
#     def __init__(self, linear_feature_columns, dnn_feature_columns, **kwargs):
#         super().__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
#         n_fields = sum(1 for feat in dnn_feature_columns if isinstance(feat, SparseFeat))
#         self.fm = ScaledFM(n_fields)

# # ================= 🧰 工具函数 =================
# def get_valid_stratified_splits(y, max_splits=5):
#     y = np.asarray(y).astype(int)
#     class_counts = np.bincount(y)
#     class_counts = class_counts[class_counts > 0]
#     if len(class_counts) < 2: return None
#     min_class_count = class_counts.min()
#     if min_class_count < 2: return None
#     return min(max_splits, min_class_count)

# def reduce_mem_usage(df):
#     """🔥 内存优化神器：自动降低数据类型精度"""
#     for col in df.columns:
#         col_type = df[col].dtype
#         if col_type != object:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#             else:
#                 df[col] = df[col].astype(np.float32)
#     return df

# # ================= 📥 读取数据 =================
# def load_raw_data():
#     # 读取时直接减少初始占用
#     train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
#     test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)
#     train_df['is_train'] = np.int8(1)
#     test_df['is_train'] = np.int8(0)
#     train_df = reduce_mem_usage(train_df)
#     test_df = reduce_mem_usage(test_df)
#     return train_df, test_df

# # ================= 📊 预处理 =================
# def preprocess_one_fold(train_fold_df, val_fold_df, test_df):
#     tr_df = train_fold_df.copy().reset_index(drop=True)
#     val_df = val_fold_df.copy().reset_index(drop=True)
#     te_df = test_df.copy().reset_index(drop=True)

#     final_sparse = SPARSE_FEATURES.copy()
#     final_dense = DENSE_FEATURES.copy()

#     # 数值转数字
#     for df in [tr_df, val_df, te_df]:
#         df[DENSE_FEATURES] = df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')

#     # 缺失指示 (使用 int8)
#     if USE_MISSING_INDICATOR:
#         for c in DENSE_FEATURES:
#             mc = c + '_miss'
#             tr_df[mc] = ((tr_df[c].isna()) | (tr_df[c] < 0)).astype(np.int8)
#             val_df[mc] = ((val_df[c].isna()) | (val_df[c] < 0)).astype(np.int8)
#             te_df[mc] = ((te_df[c].isna()) | (te_df[c] < 0)).astype(np.int8)
#             final_sparse.append(mc)

#     # 填充+截断
#     for df in [tr_df, val_df, te_df]:
#         df[DENSE_FEATURES] = df[DENSE_FEATURES].fillna(0).clip(lower=0)

#     # log + 归一化
#     for df in [tr_df, val_df, te_df]:
#         for c in DENSE_FEATURES:
#             df[c] = np.log1p(df[c])
#     scaler = MinMaxScaler()
#     tr_df[DENSE_FEATURES] = scaler.fit_transform(tr_df[DENSE_FEATURES]).astype(np.float32)
#     val_df[DENSE_FEATURES] = scaler.transform(val_df[DENSE_FEATURES]).astype(np.float32)
#     te_df[DENSE_FEATURES] = scaler.transform(te_df[DENSE_FEATURES]).astype(np.float32)

#     # 低频过滤 + 编码
#     le_dict = {}
#     for feat in SPARSE_FEATURES:
#         for df in [tr_df, val_df, te_df]:
#             df[feat] = df[feat].fillna('-1').astype(str)
#         cnt = tr_df[feat].value_counts()
#         valid = cnt[cnt >= FREQ_FILTER_THRESHOLD].index
#         for df in [tr_df, val_df, te_df]:
#             df.loc[~df[feat].isin(valid), feat] = 'Rare'
#         le = LabelEncoder()
#         # 稀疏特征类别一般不会超过 int32
#         tr_df[feat] = le.fit_transform(tr_df[feat]).astype(np.int32)
#         val_df[feat] = le.transform(val_df[feat]).astype(np.int32)
#         te_df[feat] = le.transform(te_df[feat]).astype(np.int32)
#         le_dict[feat] = le

#     # 分桶 (使用 int8/int16)
#     bin_feats = []
#     if USE_BINNING:
#         bin_feats = [c + '_bin' for c in DENSE_FEATURES]
#         est = KBinsDiscretizer(N_BINS, encode='ordinal', strategy='quantile')
#         tr_b = est.fit_transform(tr_df[DENSE_FEATURES])
#         val_b = est.transform(val_df[DENSE_FEATURES])
#         te_b = est.transform(te_df[DENSE_FEATURES])
#         tr_df[bin_feats] = tr_b.astype(np.int16)
#         val_df[bin_feats] = val_b.astype(np.int16)
#         te_df[bin_feats] = te_b.astype(np.int16)
#         final_sparse.extend(bin_feats)

#     # Target Encoding
#     te_feats = []
#     if USE_TARGET_ENCODING:
#         te_cols = SPARSE_FEATURES + bin_feats
#         gmean = tr_df['label'].mean()
#         for c in te_cols:
#             nc = c + '_te'
#             te_feats.append(nc)
#             tr_df[nc] = np.nan
#             inner = get_valid_stratified_splits(tr_df['label'].values, 5)
#             if inner and inner >= 2:
#                 skf = StratifiedKFold(inner, shuffle=True, random_state=SEED)
#                 for itr, ival in skf.split(tr_df, tr_df['label']):
#                     tmp = tr_df.iloc[itr].groupby(c)['label'].mean()
#                     tr_df.loc[tr_df.index[ival], nc] = tr_df.iloc[ival][c].map(tmp)
#             tr_df[nc] = tr_df[nc].fillna(gmean).astype(np.float32)
#             m = tr_df.groupby(c)['label'].mean()
#             val_df[nc] = val_df[c].map(m).fillna(gmean).astype(np.float32)
#             te_df[nc] = te_df[c].map(m).fillna(gmean).astype(np.float32)
#         final_dense.extend(te_feats)

#     return tr_df, val_df, te_df, final_sparse, final_dense, le_dict

# # ================= 🌲 GBDT =================
# def get_leaf_indices(X_tr, y_tr, X_val, X_te):
#     lgb_train = lgb.Dataset(X_tr, y_tr)
#     params = {
#         'objective': 'binary', 'metric': 'auc', 'num_leaves': GBDT_NUM_LEAVES,
#         'learning_rate': 0.1, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
#         'verbose': -1, 'n_jobs': -1, 'seed': SEED
#     }
#     model = lgb.train(params, lgb_train, num_boost_round=GBDT_TREES)

#     def batch_predict_leaf(data, batch_size=200000):
#         leaves = []
#         for i in range(0, len(data), batch_size):
#             batch = data.iloc[i:i+batch_size]
#             leaf_batch = model.predict(batch, pred_leaf=True)
#             # 🔥 GBDT 最大 num_leaves 只有 64，完美存入 int8 (范围 -128~127)，此举节约 75% 内存！
#             leaves.append(leaf_batch.astype(np.int8)) 
#             del leaf_batch, batch
#         return np.vstack(leaves)

#     tr_lf = batch_predict_leaf(X_tr)
#     vl_lf = batch_predict_leaf(X_val)
#     te_lf = batch_predict_leaf(X_te)

#     del model
#     gc.collect()
#     return tr_lf, vl_lf, te_lf

# # ================= 🚀 主函数 =================
# def safe_predict(model, data, batch_size=50000):
#     """分批预测，避免一次性拼接爆内存"""
#     preds = []
#     total = len(next(iter(data.values())))
#     for i in range(0, total, batch_size):
#         batch = {k: v[i:i+batch_size] for k, v in data.items()}
#         p = model.predict(batch, batch_size=batch_size).reshape(-1)
#         preds.append(p)
#         del batch, p
#     return np.concatenate(preds)

# def main():
#     seed_everything(SEED)
#     print("🔥 终极内存安全版启动 (数据类型降维 + 严苛垃圾回收)...")

#     train_raw, test_raw = load_raw_data()
#     y_true = test_raw['label'].values

#     oof = np.zeros(len(train_raw), dtype=np.float32)
#     test_probs = np.zeros(len(test_raw), dtype=np.float32)

#     skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

#     for fold, (tr_idx, val_idx) in enumerate(skf.split(train_raw, train_raw['label'])):
#         print(f"\n>>>>>>>>>>>>>>> Fold {fold+1}/{N_FOLDS} <<<<<<<<<<<<<<<")

#         tr_fold = train_raw.iloc[tr_idx].copy()
#         val_fold = train_raw.iloc[val_idx].copy()

#         tr, val, te, sparse_cols, dense_cols, le_dict = preprocess_one_fold(
#             tr_fold, val_fold, test_raw
#         )
        
#         # 提取完数据立即删除不再使用的 fold 数据
#         del tr_fold, val_fold
#         gc.collect()

#         X_tr = tr.drop(['label', 'is_train'], axis=1)
#         y_tr = tr['label'].values.astype(np.float32)
#         X_val = val.drop(['label', 'is_train'], axis=1)
#         y_val = val['label'].values.astype(np.float32)
#         X_te = te.drop(['label', 'is_train'], axis=1)

#         tr_leaf, val_leaf, te_leaf = get_leaf_indices(X_tr, y_tr, X_val, X_te)

#         # 🔥 关键内存清理节点：获得叶子后，立刻干掉 X_tr, X_val, X_te
#         del X_tr, X_val, X_te
#         gc.collect()

#         def make_input(df, leaf):
#             inp = {}
#             for c in dense_cols:
#                 inp[c] = df[c].values.astype(np.float32)
#             for c in sparse_cols:
#                 inp[c] = df[c].values.astype(np.int32)
#             for i in range(leaf.shape[1]):
#                 inp[f'gbdt_{i}'] = leaf[:, i] # 在预测函数里已经是 int8 了
#             return inp

#         # 构建完字典后，立刻删除原 DF 和 叶子矩阵！严格控制内存共存时间。
#         tr_in = make_input(tr, tr_leaf)
#         del tr, tr_leaf
#         gc.collect()

#         vl_in = make_input(val, val_leaf)
#         del val, val_leaf
#         gc.collect()

#         te_in = make_input(te, te_leaf)
#         del te, te_leaf
#         gc.collect()

#         # ================= 模型 =================
#         # DeepCTR 特征设置
#         fix = [SparseFeat(c, int(np.max(tr_in[c])) + 1, EMBEDDING_DIM) for c in sparse_cols]
#         dns = [DenseFeat(c, 1) for c in dense_cols]
#         gbdt = [SparseFeat(f'gbdt_{i}', GBDT_NUM_LEAVES + 1, EMBEDDING_DIM) for i in range(GBDT_TREES)]

#         linear = fix + dns + gbdt
#         dnn = linear

#         Model = ScaledDeepFM if USE_SCALED_FM else BaseDeepFM
#         model = Model(
#             linear, dnn,
#             task='binary',
#             dnn_use_bn=USE_DNN_BN,
#             dnn_hidden_units=DNN_HIDDEN_UNITS,
#             dnn_dropout=DNN_DROPOUT,
#             device=DEVICE
#         )

#         es = EarlyStopping(monitor='val_auc', patience=1, mode='max', verbose=0)
#         model.compile('adam', 'binary_crossentropy', metrics=['auc'])

#         model.fit(
#             tr_in, y_tr,
#             BATCH_SIZE, EPOCHS,
#             verbose=1,
#             validation_data=(vl_in, y_val),
#             callbacks=[es]
#         )

#         # 预测时清理显存
#         torch.cuda.empty_cache()
        
#         v_pred = safe_predict(model, vl_in)
#         t_pred = safe_predict(model, te_in)

#         oof[val_idx] = v_pred
#         test_probs += t_pred / N_FOLDS

#         print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, v_pred):.5f}")

#         # 折叠结束，彻底释放所有相关内存
#         del model, tr_in, vl_in, te_in, y_tr, y_val, v_pred, t_pred
#         torch.cuda.empty_cache()
#         gc.collect()

#     # ================= 保存结果 =================
#     all_off = not any([
#         USE_MISSING_INDICATOR,
#         USE_BINNING,
#         USE_TARGET_ENCODING,
#         USE_SCALED_FM,
#         USE_DNN_BN
#     ])

#     name = "pred_base" if all_off else "pred_final"

#     np.save(f"{SAVE_BASE_PATH}\\{name}.npy", test_probs)

#     if not os.path.exists(f"{SAVE_BASE_PATH}\\y_true.npy"):
#         np.save(f"{SAVE_BASE_PATH}\\y_true.npy", y_true)

#     # ================= ROC =================
#     fpr, tpr, _ = roc_curve(y_true, test_probs)
#     auc = sk_auc(fpr, tpr)

#     plt.figure(figsize=(7, 6))
#     plt.plot(fpr, tpr, label=f'{name} AUC={auc:.4f}')
#     plt.plot([0, 1], [0, 1], '--k')
#     plt.title(f"{name} ROC Curve")
#     plt.legend()
#     plt.grid(alpha=0.3)

#     plt.savefig(f"{SAVE_BASE_PATH}\\{name}.png", dpi=300, bbox_inches='tight')
#     plt.show()

#     print(f"\n✅ 全部跑完！已保存：{name}.npy + {name}.png")

# if __name__ == '__main__':
#     main()











import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
import random
import os
import gc
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc as sk_auc
from sklearn.model_selection import StratifiedKFold

from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM as BaseDeepFM
from deepctr_torch.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 修复画图中文乱码
plt.rcParams['axes.unicode_minus'] = False

# ================= ⚙️ 【一键切换版本】 =================
USE_MISSING_INDICATOR = True
USE_BINNING           = False
USE_TARGET_ENCODING   = True
USE_SCALED_FM         = True
USE_DNN_BN            = True
# ============================================================

TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"
SAVE_BASE_PATH = r"D:\code\Python\criteo"

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

COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ScaledFM(nn.Module):
    def __init__(self, n_fields):
        super().__init__()
        self.n_fields = n_fields
        self.field_scale = nn.Parameter(torch.ones(n_fields))

    def forward(self, fm_input):
        scale = self.field_scale.view(1, -1, 1).to(fm_input.device)
        scaled = fm_input * scale
        square_of_sum = torch.sum(scaled, dim=1) ** 2
        sum_of_square = torch.sum(scaled ** 2, dim=1)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return torch.sum(cross_term, dim=1, keepdim=True)

class ScaledDeepFM(BaseDeepFM):
    def __init__(self, linear_feature_columns, dnn_feature_columns, **kwargs):
        super().__init__(linear_feature_columns, dnn_feature_columns, **kwargs)
        n_fields = sum(1 for feat in dnn_feature_columns if isinstance(feat, SparseFeat))
        self.fm = ScaledFM(n_fields)

def get_valid_stratified_splits(y, max_splits=5):
    y = np.asarray(y).astype(int)
    class_counts = np.bincount(y)
    class_counts = class_counts[class_counts > 0]
    if len(class_counts) < 2: return None
    min_class_count = class_counts.min()
    if min_class_count < 2: return None
    return min(max_splits, min_class_count)

# ================= 📥 读取数据 (恢复原版) =================
def load_raw_data():
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
    test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    return train_df, test_df

# ================= 📊 预处理 (完全恢复原版精度) =================
def preprocess_one_fold(train_fold_df, val_fold_df, test_df):
    tr_df = train_fold_df.copy().reset_index(drop=True)
    val_df = val_fold_df.copy().reset_index(drop=True)
    te_df = test_df.copy().reset_index(drop=True)

    final_sparse = SPARSE_FEATURES.copy()
    final_dense = DENSE_FEATURES.copy()

    for df in [tr_df, val_df, te_df]:
        df[DENSE_FEATURES] = df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')

    if USE_MISSING_INDICATOR:
        for c in DENSE_FEATURES:
            mc = c + '_miss'
            tr_df[mc] = ((tr_df[c].isna()) | (tr_df[c] < 0)).astype(np.int32)
            val_df[mc] = ((val_df[c].isna()) | (val_df[c] < 0)).astype(np.int32)
            te_df[mc] = ((te_df[c].isna()) | (te_df[c] < 0)).astype(np.int32)
            final_sparse.append(mc)

    for df in [tr_df, val_df, te_df]:
        df[DENSE_FEATURES] = df[DENSE_FEATURES].fillna(0).clip(lower=0)

    for df in [tr_df, val_df, te_df]:
        for c in DENSE_FEATURES:
            df[c] = np.log1p(df[c])
            
    scaler = MinMaxScaler()
    tr_df[DENSE_FEATURES] = scaler.fit_transform(tr_df[DENSE_FEATURES]).astype(np.float32)
    val_df[DENSE_FEATURES] = scaler.transform(val_df[DENSE_FEATURES]).astype(np.float32)
    te_df[DENSE_FEATURES] = scaler.transform(te_df[DENSE_FEATURES]).astype(np.float32)

    le_dict = {}
    for feat in SPARSE_FEATURES:
        for df in [tr_df, val_df, te_df]:
            df[feat] = df[feat].fillna('-1').astype(str)
        cnt = tr_df[feat].value_counts()
        valid = cnt[cnt >= FREQ_FILTER_THRESHOLD].index
        for df in [tr_df, val_df, te_df]:
            df.loc[~df[feat].isin(valid), feat] = 'Rare'
        le = LabelEncoder()
        tr_df[feat] = le.fit_transform(tr_df[feat]).astype(np.int32)
        val_df[feat] = le.transform(val_df[feat]).astype(np.int32)
        te_df[feat] = le.transform(te_df[feat]).astype(np.int32)
        le_dict[feat] = le

    bin_feats = []
    if USE_BINNING:
        bin_feats = [c + '_bin' for c in DENSE_FEATURES]
        est = KBinsDiscretizer(N_BINS, encode='ordinal', strategy='quantile')
        tr_b = est.fit_transform(tr_df[DENSE_FEATURES])
        val_b = est.transform(val_df[DENSE_FEATURES])
        te_b = est.transform(te_df[DENSE_FEATURES])
        tr_df[bin_feats] = tr_b.astype(np.int32)
        val_df[bin_feats] = val_b.astype(np.int32)
        te_df[bin_feats] = te_b.astype(np.int32)
        final_sparse.extend(bin_feats)

    te_feats = []
    if USE_TARGET_ENCODING:
        te_cols = SPARSE_FEATURES + bin_feats
        gmean = tr_df['label'].mean()
        for c in te_cols:
            nc = c + '_te'
            te_feats.append(nc)
            tr_df[nc] = np.nan
            inner = get_valid_stratified_splits(tr_df['label'].values, 5)
            if inner and inner >= 2:
                skf = StratifiedKFold(inner, shuffle=True, random_state=SEED)
                for itr, ival in skf.split(tr_df, tr_df['label']):
                    tmp = tr_df.iloc[itr].groupby(c)['label'].mean()
                    tr_df.loc[tr_df.index[ival], nc] = tr_df.iloc[ival][c].map(tmp)
            tr_df[nc] = tr_df[nc].fillna(gmean).astype(np.float32)
            m = tr_df.groupby(c)['label'].mean()
            val_df[nc] = val_df[c].map(m).fillna(gmean).astype(np.float32)
            te_df[nc] = te_df[c].map(m).fillna(gmean).astype(np.float32)
        final_dense.extend(te_feats)

    return tr_df, val_df, te_df, final_sparse, final_dense, le_dict

# ================= 🌲 GBDT =================
def get_leaf_indices(X_tr, y_tr, X_val, X_te):
    lgb_train = lgb.Dataset(X_tr, y_tr)
    params = {
        'objective': 'binary', 'metric': 'auc', 'num_leaves': GBDT_NUM_LEAVES,
        'learning_rate': 0.1, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'verbose': -1, 'n_jobs': -1, 'seed': SEED
    }
    model = lgb.train(params, lgb_train, num_boost_round=GBDT_TREES)

    def batch_predict_leaf(data, batch_size=200000):
        leaves = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            leaf_batch = model.predict(batch, pred_leaf=True)
            # 这里保存为int8省内存（完全不损失精度，因为最大值只有63）
            leaves.append(leaf_batch.astype(np.int8)) 
            del leaf_batch, batch
        return np.vstack(leaves)

    tr_lf = batch_predict_leaf(X_tr)
    vl_lf = batch_predict_leaf(X_val)
    te_lf = batch_predict_leaf(X_te)

    del model
    gc.collect()
    return tr_lf, vl_lf, te_lf

# ================= 🚀 主函数 =================
def safe_predict(model, data, batch_size=50000):
    preds = []
    total = len(next(iter(data.values())))
    for i in range(0, total, batch_size):
        batch = {k: v[i:i+batch_size] for k, v in data.items()}
        p = model.predict(batch, batch_size=batch_size).reshape(-1)
        preds.append(p)
        del batch, p
    return np.concatenate(preds)

def main():
    seed_everything(SEED)
    print("🔥 保精度+防爆内存版：AUC将完全对齐原版！")

    train_raw, test_raw = load_raw_data()
    y_true = test_raw['label'].values

    oof = np.zeros(len(train_raw))
    test_probs = np.zeros(len(test_raw))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_raw, train_raw['label'])):
        print(f"\n>>>>>>>>>>>>>>> Fold {fold+1}/{N_FOLDS} <<<<<<<<<<<<<<<")

        tr_fold = train_raw.iloc[tr_idx].copy()
        val_fold = train_raw.iloc[val_idx].copy()

        tr, val, te, sparse_cols, dense_cols, le_dict = preprocess_one_fold(
            tr_fold, val_fold, test_raw
        )
        
        del tr_fold, val_fold
        gc.collect()

        X_tr = tr.drop(['label', 'is_train'], axis=1)
        y_tr = tr['label'].values
        X_val = val.drop(['label', 'is_train'], axis=1)
        y_val = val['label'].values
        X_te = te.drop(['label', 'is_train'], axis=1)

        tr_leaf, val_leaf, te_leaf = get_leaf_indices(X_tr, y_tr, X_val, X_te)

        # 核心防爆点1：得到叶子立刻删除训练集冗余部分
        del X_tr, X_val, X_te
        gc.collect()

        def make_input(df, leaf):
            inp = {}
            for c in dense_cols:
                inp[c] = df[c].values.astype(np.float32)
            for c in sparse_cols:
                inp[c] = df[c].values.astype(np.int32)
            for i in range(leaf.shape[1]):
                # 恢复原版的 int32 输入，保证 DeepCTR 内部 Embedding 处理完全一致
                inp[f'gbdt_{i}'] = leaf[:, i].astype(np.int32)
            return inp

        # 核心防爆点2：每次构建完一个字典，立马删除其原始 DataFrame 和叶子矩阵
        tr_in = make_input(tr, tr_leaf)
        # DeepCTR 需要知道 max 特征维度，提前算好
        sparse_vocab = {c: int(tr[c].max()) + 1 for c in sparse_cols}
        del tr, tr_leaf
        gc.collect()

        vl_in = make_input(val, val_leaf)
        del val, val_leaf
        gc.collect()

        te_in = make_input(te, te_leaf)
        del te, te_leaf
        gc.collect()

        # ================= 模型 =================
        fix = [SparseFeat(c, sparse_vocab[c], EMBEDDING_DIM) for c in sparse_cols]
        dns = [DenseFeat(c, 1) for c in dense_cols]
        gbdt = [SparseFeat(f'gbdt_{i}', GBDT_NUM_LEAVES + 1, EMBEDDING_DIM) for i in range(GBDT_TREES)]

        linear = fix + dns + gbdt
        dnn = linear

        Model = ScaledDeepFM if USE_SCALED_FM else BaseDeepFM
        model = Model(
            linear, dnn,
            task='binary',
            dnn_use_bn=USE_DNN_BN,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_dropout=DNN_DROPOUT,
            device=DEVICE
        )

        es = EarlyStopping(monitor='val_auc', patience=1, mode='max', verbose=0)
        model.compile('adam', 'binary_crossentropy', metrics=['auc'])

        model.fit(
            tr_in, y_tr,
            BATCH_SIZE, EPOCHS,
            verbose=1,
            validation_data=(vl_in, y_val),
            callbacks=[es]
        )

        v_pred = safe_predict(model, vl_in)
        t_pred = safe_predict(model, te_in)

        oof[val_idx] = v_pred
        test_probs += t_pred / N_FOLDS

        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, v_pred):.5f}")

        # 核心防爆点3：折叠结束，全部清理
        del model, tr_in, vl_in, te_in, y_tr, y_val, v_pred, t_pred
        torch.cuda.empty_cache()
        gc.collect()

    # ================= 保存与出图 =================
    all_off = not any([USE_MISSING_INDICATOR, USE_BINNING, USE_TARGET_ENCODING, USE_SCALED_FM, USE_DNN_BN])
    name = "pred_base" if all_off else "pred_final123"

    np.save(f"{SAVE_BASE_PATH}\\{name}.npy", test_probs)
    if not os.path.exists(f"{SAVE_BASE_PATH}\\y_true.npy"):
        np.save(f"{SAVE_BASE_PATH}\\y_true.npy", y_true)

    fpr, tpr, _ = roc_curve(y_true, test_probs)
    auc = sk_auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f'{name} AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], '--k')
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"{SAVE_BASE_PATH}\\{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ 全部跑完！已保存：{name}.npy + {name}.png")

if __name__ == '__main__':
    main()