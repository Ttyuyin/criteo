import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import time
import gc
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

VOCAB_SIZE = 20000  
N_FOLDS = 5
SEED = 2026

DTYPES = {
    'label': np.float32, 
    **{f'I{i}': np.float32 for i in range(1, 14)},
    **{f'C{i}': str for i in range(1, 27)}
}

COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
NUM_COLS = [f'I{i}' for i in range(1, 14)]
CAT_COLS = [f'C{i}' for i in range(1, 27)]

def process_data_strict(train_df, test_df):
    print("开始无泄漏数据预处理...")
    
    train_rows = len(train_df)
    y_train = train_df['label'].fillna(0).astype(np.int8).values
    y_test_true = test_df['label'].fillna(0).astype(np.int8).values 
    
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    del train_df, test_df
    gc.collect()

    for col in NUM_COLS:
        all_df[col] = all_df[col].fillna(0)
        all_df.loc[all_df[col] < 0, col] = 0
        all_df[col] = np.log1p(all_df[col])
    
    X_num_train = all_df.iloc[:train_rows][NUM_COLS].values
    est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile', subsample=200000)
    est.fit(X_num_train)
    X_bin_all = est.transform(all_df[NUM_COLS].values)
    bin_cols = [f'bin_{i}' for i in range(13)]
    bin_df = pd.DataFrame(X_bin_all, columns=bin_cols, dtype=np.int32)
    
    for col in CAT_COLS:
        all_df[col] = all_df[col].fillna('unk')
        train_series = all_df[col].iloc[:train_rows]
        counts = train_series.value_counts()
        top_vocab = counts.index[:VOCAB_SIZE]
        vocab_map = {val: i+1 for i, val in enumerate(top_vocab)}
        all_df[col] = all_df[col].map(vocab_map).fillna(0).astype(np.int32)

    all_df = pd.concat([all_df[CAT_COLS], bin_df], axis=1)
    
    enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
    train_feat = all_df.iloc[:train_rows]
    test_feat = all_df.iloc[train_rows:]
    X_train = enc.fit_transform(train_feat).tocsr()
    X_test = enc.transform(test_feat).tocsr()
    
    print(f"Train维度: {X_train.shape}, Test维度: {X_test.shape}")
    del all_df, bin_df, train_feat, test_feat
    gc.collect()
    
    return X_train, y_train, X_test, y_test_true

def main():
    start_total = time.time()
    print("1. 读取数据")
    train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None, dtype=DTYPES)
    test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None, dtype=DTYPES)
    X_train, y_train, X_test, y_test_true = process_data_strict(train_df, test_df)

    print("\n2. 网格搜索最优参数")
    param_grid = {
        'alpha': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        'penalty': ['l2', 'l1', 'elasticnet']
    }
    sample_size = min(500000, X_train.shape[0])
    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_sample, y_sample = X_train[indices], y_train[indices]
    X_s, X_v, y_s, y_v = train_test_split(X_sample, y_sample, test_size=0.2, random_state=SEED)
    
    best_score = 0
    best_params = None
    all_param_results = []
    
    for alpha in param_grid['alpha']:
        for penalty in param_grid['penalty']:
            if penalty == 'elasticnet':
                model = SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, 
                                      l1_ratio=0.5, n_jobs=-1, random_state=SEED, max_iter=1000)
            else:
                model = SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, 
                                      n_jobs=-1, random_state=SEED, max_iter=1000)
            
            model.fit(X_s, y_s)
            y_v_pred = model.predict_proba(X_v)[:, 1]
            auc = roc_auc_score(y_v, y_v_pred)
            all_param_results.append((alpha, penalty, auc))
            
            if auc > best_score:
                best_score = auc
                best_params = (alpha, penalty)
    
    print(f"\n最优参数组合：alpha={best_params[0]}, penalty={best_params[1]} | 对应AUC={best_score:.5f}")

    print("\n3. 5折交叉验证（计算AUC/LogLoss）")
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    test_preds = np.zeros(X_test.shape[0])
    fold_metrics = []
    
    for fold, (t_idx, v_idx) in enumerate(folds.split(X_train, y_train)):
        X_t, X_v = X_train[t_idx], X_train[v_idx]
        y_t, y_v = y_train[t_idx], y_train[v_idx]
        
        if best_params[1] == 'elasticnet':
            model = SGDClassifier(loss='log_loss', penalty=best_params[1], alpha=best_params[0],
                                  l1_ratio=0.5, max_iter=1000, tol=1e-4, n_jobs=-1, random_state=SEED)
        else:
            model = SGDClassifier(loss='log_loss', penalty=best_params[1], alpha=best_params[0],
                                  max_iter=1000, tol=1e-4, n_jobs=-1, random_state=SEED)
        
        model.fit(X_t, y_t)
        y_v_pred = model.predict_proba(X_v)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
        
        fold_auc = roc_auc_score(y_v, y_v_pred)
        fold_logloss = log_loss(y_v, np.clip(y_v_pred, 1e-7, 1-1e-7))
        fold_metrics.append((fold_auc, fold_logloss))
        
        test_preds += y_test_pred / N_FOLDS
    
    avg_auc = np.mean([m[0] for m in fold_metrics])
    avg_logloss = np.mean([m[1] for m in fold_metrics])
    print(f"\n折内验证集平均指标 | AUC: {avg_auc:.5f} | LogLoss: {avg_logloss:.5f}")

    print("\n4. 测试集指标计算")
    test_auc = roc_auc_score(y_test_true, test_preds)
    test_logloss = log_loss(y_test_true, np.clip(test_preds, 1e-7, 1-1e-7))
    print(f"测试集最终指标 | AUC: {test_auc:.5f} | LogLoss: {test_logloss:.5f}")

    gc.collect()
    print(f"\n任务完成，总耗时：{time.time() - start_total:.2f} 秒")

if __name__ == '__main__':
    main()



# 折内验证集平均指标 | AUC: 0.78717 | LogLoss: 0.45730
# 4. 测试集指标计算 
# 测试集最终指标 | AUC: 0.78076 | LogLoss: 0.45838

# ===== 任务完成，总耗时：398.79 秒 =====
# PS D:\code\Python\dachuang\criteo> 