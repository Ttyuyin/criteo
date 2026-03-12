import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM
from deepctr_torch.callbacks import EarlyStopping
import time
import gc
import warnings
warnings.filterwarnings('ignore')


TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS = 5
SEED = 2026


FREQ_FILTER_THRESHOLD = 5


GBDT_NUM_LEAVES = 64
GBDT_TREES = 100


EMBEDDING_DIM = 10
BATCH_SIZE = 4096
EPOCHS = 2
DNN_DROPOUT = 0.5
DNN_HIDDEN_UNITS = (512, 256)


COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]

def process_base_data_optimized():

    print(f"[{time.strftime('%H:%M:%S')}] 正在读取数据...")
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
    test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    train_df['is_train'] = 1
    test_df['is_train'] = 0


    print(f"[{time.strftime('%H:%M:%S')}] 正在处理数值特征...")
  
    train_df[DENSE_FEATURES] = train_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)
    test_df[DENSE_FEATURES] = test_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)
    

    for col in DENSE_FEATURES:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])
    

    mms = MinMaxScaler(feature_range=(0, 1))
    train_df[DENSE_FEATURES] = mms.fit_transform(train_df[DENSE_FEATURES]).astype(np.float16)
    test_df[DENSE_FEATURES] = mms.transform(test_df[DENSE_FEATURES]).astype(np.float16)


    print(f"[{time.strftime('%H:%M:%S')}] 正在处理类别特征 (低频过滤)...")
    for feat in SPARSE_FEATURES:

        train_df[feat] = train_df[feat].fillna('-1').astype(str)
        test_df[feat] = test_df[feat].fillna('-1').astype(str)
        

        cnt = train_df[feat].value_counts()
        valid_cats = set(cnt[cnt >= FREQ_FILTER_THRESHOLD].index)
        

        train_df.loc[~train_df[feat].isin(valid_cats), feat] = 'Rare'

        test_df.loc[~test_df[feat].isin(valid_cats), feat] = 'Rare'
        

        le = LabelEncoder()
        le.fit(train_df[feat])
        train_df[feat] = le.transform(train_df[feat]).astype(np.int32)

        test_df[feat] = le.transform(test_df[feat]).astype(np.int32)
        
    return train_df, test_df, y_train, y_test


def get_leaf_indices_batched(X_train, y_train, X_val, X_test):
    """训练 LGBM 并提取叶子节点（分批预测，内存友好）"""
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

def main():
    print(f"[Optimized GBDT+DeepFM] Start! (Filter={FREQ_FILTER_THRESHOLD}, Epochs={EPOCHS})")
    start_total = time.time()


    train_df, test_df, y_train_full, y_test_true = process_base_data_optimized()
    
    X_train_full_df = train_df.drop(['label', 'is_train'], axis=1)
    X_test_df = test_df.drop(['label', 'is_train'], axis=1)


    print(f"[{time.strftime('%H:%M:%S')}] 构建 Feature Columns...")
    
    original_fixlen_features = []
    for feat in SPARSE_FEATURES:

        max_val = train_df[feat].max()
        original_fixlen_features.append(
            SparseFeat(feat, vocabulary_size=int(max_val) + 1, embedding_dim=EMBEDDING_DIM)
        )

    dense_features = [DenseFeat(feat, 1) for feat in DENSE_FEATURES]

    gbdt_leaf_features = [
        SparseFeat(f'gbdt_tree_{i}', vocabulary_size=GBDT_NUM_LEAVES + 1, embedding_dim=EMBEDDING_DIM)
        for i in range(GBDT_TREES)
    ]
    
    dnn_feature_columns = original_fixlen_features + gbdt_leaf_features + dense_features
    linear_feature_columns = original_fixlen_features + gbdt_leaf_features + dense_features


    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(folds.split(X_train_full_df, y_train_full)):
        print(f"\n⚡ Fold {fold + 1} / {N_FOLDS}")
        
        X_tr = X_train_full_df.iloc[train_idx]
        y_tr = y_train_full[train_idx]
        X_val = X_train_full_df.iloc[val_idx]
        y_val = y_train_full[val_idx]
        

        print("   > [LGBM] Extracting leaves (batched)...")
        tr_leaf_blocks, val_leaf_blocks, te_leaf_blocks = get_leaf_indices_batched(X_tr, y_tr, X_val, X_test_df)
        

        def build_input(df, leaf_blocks):
            d = {}

            for name in DENSE_FEATURES:
                d[name] = df[name].values.astype(np.float16)
  
            for name in SPARSE_FEATURES:
                d[name] = df[name].values.astype(np.int32)

            leaf_idx = 0
            for leaf_block in leaf_blocks:
                block = leaf_block  
                for i in range(block.shape[1]):
                    d[f'gbdt_tree_{leaf_idx}'] = block[:, i]
                    leaf_idx += 1
            return d

        tr_inp = build_input(X_tr, tr_leaf_blocks)
        val_inp = build_input(X_val, val_leaf_blocks)
        te_inp = build_input(test_df, te_leaf_blocks)


        del X_tr, X_val, tr_leaf_blocks, val_leaf_blocks, te_leaf_blocks
        gc.collect()


        print("   > [DeepFM] Training...")
        model = DeepFM(linear_feature_columns, dnn_feature_columns,
                       task='binary',
                       dnn_hidden_units=DNN_HIDDEN_UNITS,
                       l2_reg_embedding=1e-5,
                       dnn_dropout=DNN_DROPOUT,
                       device=DEVICE)
        
        model.compile("adam", "binary_crossentropy", metrics=["auc"])
        es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=0, patience=1, mode='max')
        
        model.fit(tr_inp, y_tr,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=1,
                  validation_data=(val_inp, y_val),
                  callbacks=[es])

        val_pred_fold = model.predict(val_inp, batch_size=BATCH_SIZE*2).flatten()
        oof_preds[val_idx] = val_pred_fold
        test_preds += model.predict(te_inp, batch_size=BATCH_SIZE*2).flatten() / N_FOLDS
        
        fold_auc = roc_auc_score(y_val, val_pred_fold)
        fold_logloss = log_loss(y_val, val_pred_fold)
        print(f"   >>> Fold {fold + 1} Result: AUC={fold_auc:.6f}, LogLoss={fold_logloss:.6f}")
        
        del model, tr_inp, val_inp, te_inp
        torch.cuda.empty_cache()
        gc.collect()


    print("\n" + "="*40)
    print("GBDT+DeepFM 最终结果")
    print("="*40)
    
    train_auc = roc_auc_score(y_train_full, oof_preds)
    train_logloss = log_loss(y_train_full, oof_preds)
    test_auc = roc_auc_score(y_test_true, test_preds)
    test_logloss = log_loss(y_test_true, test_preds)

    print(f"Train OOF AUC     : {train_auc:.6f}")
    print(f"Train OOF LogLoss : {train_logloss:.6f}")
    print("-" * 20)
    print(f"Test Score AUC    : {test_auc:.6f}")
    print(f"Test Score LogLoss: {test_logloss:.6f}")
    print(f"耗时: {(time.time() - start_total)/60:.1f} mins")

if __name__ == '__main__':
    main()


# ========================================
# GBDT+DeepFM 最终结果（2024）
# ========================================
# Train OOF AUC     : 0.796061
# Train OOF LogLoss : 0.451378
# --------------------
# Test Score AUC    : 0.797254
# Test Score LogLoss: 0.445469
# 耗时: 56.3 mins


# ========================================
# GBDT+DeepFM 最终结果 2025
# ========================================
# Train OOF AUC     : 0.796146
# Train OOF LogLoss : 0.450767
# --------------------
# Test Score AUC    : 0.797263
# Test Score LogLoss: 0.444870
# 耗时: 56.3 mins

# ========================================
# GBDT+DeepFM 最终结果（2026）
# ========================================
# Train OOF AUC     : 0.795804
# Train OOF LogLoss : 0.451486
# --------------------
# Test Score AUC    : 0.796995
# Test Score LogLoss: 0.445516
# 耗时: 56.1 mins

# ========================================
# GBDT+DeepFM 最终结果 2027
# ========================================
# Train OOF AUC     : 0.795695
# Train OOF LogLoss : 0.451044
# --------------------
# Test Score AUC    : 0.797318
# Test Score LogLoss: 0.444780
# 耗时: 56.0 min

# ========================================
# GBDT+DeepFM 最终结果 2028
# ========================================
# Train OOF AUC     : 0.795888
# Train OOF LogLoss : 0.451202
# --------------------
# Test Score AUC    : 0.797303
# Test Score LogLoss: 0.444893
# 耗时: 56.0 mins