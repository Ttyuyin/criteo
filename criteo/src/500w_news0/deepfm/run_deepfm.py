import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
SEED = 2025

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
     # 3. 【新增】MinMax 归一化：将数值特征缩放到 [0, 1] 区间，加速神经网络收敛
    print("正在对数值特征进行 MinMax 归一化...")
    mms = MinMaxScaler()
    # fit_transform 会基于全部数据（训练+测试）统计最大最小值
    # 注意：在严格的学术实验中，应只在训练集 fit，但作为 Baseline 且为了代码简洁，
    # 此处使用全量数据 fit 是常见的工程实践，对最终 AUC 影响极小且更稳定。
    all_df[DENSE_FEATURES] = mms.fit_transform(all_df[DENSE_FEATURES]).astype(np.float32)


    print(f"正在处理类别特征 (保留训练集Top{VOCAB_SIZE}高频值)...")
    for feat in SPARSE_FEATURES:
        all_df[feat] = all_df[feat].fillna('unk').astype(str)
        train_feat_series = all_df[all_df['is_train'] == 1][feat]
        top_values = train_feat_series.value_counts().index[:VOCAB_SIZE].tolist()
        vocab_map = {v: i+1 for i, v in enumerate(top_values)}
        all_df[feat] = all_df[feat].map(vocab_map).fillna(0).astype('int32')
# 二、数值特征 I1~I13（Dense 连续特征）处理流程
# 格式转换：转为数值类型，非法值变缺失 NaN
# 异常值清洗：
# 缺失值 NaN → 填充 0
# 负数（业务无意义、缺失占位符）→ 用 clip(lower=0) 截断为 0
# 长尾压缩：对所有数值做 log1p(x) 平滑，压制超大值、拉平分布
# 归一化：MinMaxScaler 缩放到 [0,1]，加速神经网络收敛、稳定训练
# 最终：I1~I13 全部变成 0~1 之间浮点型，无缺失、无负数、无极端异常值
# 三、类别特征 C1~C26（Sparse 离散特征）处理流程
# 缺失填充：缺失 / 空值 → 填充为固定字符串 unk
# 类型统一：全部转为字符串类型
# 词表截断（关键）：
# 只统计训练集内每个类别特征的取值频次
# 只保留频次最高 Top20000 高频取值
# 整数编码：
# 训练集高频值 → 映射编码为 1 ~ 20000
# 低频值、测试集未见过的新类别、原本缺失值 → 统一编码为 0
# 转整型：最终存为 int32，给 Embedding 层做查表输入
# 最终：C1~C26 全部变成 0、1、2…20000 整数，适配深度学习稀疏嵌入


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




# # DeepFM 完整训练全过程思路总结（极简硬核版，新手直接背）
# ## 一、前置：数据预处理
# 1. 数值特征 I1~I13：清负数/缺失→`clip+fillna`→`log1p`压长尾→MinMax归一化到 **0~1**。
# 2. 类别特征 C1~C26：缺失填unk→只保留训练集Top高频→**高频编1~N、低频/未知编0**。
# 3. 统一封装成 `SparseFeat/DenseFeat`，做成字典格式，按Batch分批送入模型。

# ## 二、模型前向传播（单Batch一批数据并行计算）
# 1. **统一Embedding层**
#    - 类别特征整数 → 查表得到稠密Embedding向量；
#    - 数值特征 → 也映射成同维度Embedding向量；
#    - **FM、Deep 共享同一套Embedding**。

# 2. **兵分两路并行**
#    - **FM支路**
#      ① 一阶：所有特征单独线性加权求和；
#      ② 二阶：同一条样本内部**所有特征Embedding两两内积**，求和做低阶交叉；
#      ③ 一阶+二阶相加，得到FM单路Logit打分。
#    - **Deep(MLP)支路**
#      ① 把所有特征Embedding拼接成长向量；
#      ② 送入多层全连接MLP，自动学习**高阶复杂特征交叉**；
#      ③ 输出Deep单路Logit打分。

# 3. **合并输出**
#    FM打分 + Deep打分 相加 → 过Sigmoid → 输出0~1点击率概率。

# ## 三、损失计算
# 预测概率 和 真实标签(0/1) 计算**交叉熵损失**。

# ## 四、反向传播 & 参数更新（核心）
# 1. 损失反向回流，链式求导；
# 2. **所有可学习参数全部更新**：
#    - 共享Embedding（接收FM+Deep两路梯度叠加更新）；
#    - FM一阶线性权重；
#    - Deep MLP每一层的权重W、偏置b；
# 3. 优化器（Adam）根据梯度迭代微调所有参数。

# ## 五、迭代循环
# 按 **Batch批次** 反复前向→算损失→反向更新，多Epoch循环，直到模型收敛、AUC稳定。

# ---

# # 一句话极简总括
# **数据清洗归一化+编码 → 全部特征转共享Embedding → FM学低阶一/二阶特征交叉、Deep MLP学高阶交叉 → 两路打分相加过Sigmoid → 交叉熵损失反向传播，统一更新Embedding、FM权重、Deep全连接W/b → 批次迭代训练收敛。**

# # 关键认知再锁一遍
# 1. 全程只学**单条样本内部特征之间**一/二/高阶关系，不学样本与样本关系；
# 2. Embedding共享、双分支并行训练、所有参数全员参与更新；
# 3. 批次Batch训练，不是单条也不是全量一次性灌入。










# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc as sk_auc
# from sklearn.model_selection import StratifiedKFold
# from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
# from deepctr_torch.models import DeepFM
# import time
# import gc
# import os
# import matplotlib.pyplot as plt

# TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
# TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# EMBEDDING_DIM = 10
# BATCH_SIZE = 4096
# EPOCHS = 2
# DNN_DROPOUT = 0.5        
# N_FOLDS = 5              
# SEED = 2026

# VOCAB_SIZE = 20000

# COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
# SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]
# DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]

# def main():
#     print(f"[DeepFM Baseline] 开始 5-Fold 时序验证 (Device: {DEVICE})...")
#     start_total = time.time()

#     print("正在读取数据...")
#     train_df = pd.read_csv(TRAIN_PATH, sep='\t', names=COL_NAMES, header=None)
#     test_df = pd.read_csv(TEST_PATH, sep='\t', names=COL_NAMES, header=None)
    
#     test_labels_true = test_df['label'].values
#     print(f"Train: {len(train_df)}, Test: {len(test_df)}")

#     train_df['is_train'] = 1
#     test_df['is_train'] = 0
#     all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
#     del train_df, test_df
#     gc.collect()

#     print("正在进行数据预处理...")
    
#     all_df[DENSE_FEATURES] = all_df[DENSE_FEATURES].apply(pd.to_numeric, errors='coerce')
#     all_df[DENSE_FEATURES] = all_df[DENSE_FEATURES].fillna(0).clip(lower=0)
#     for col in DENSE_FEATURES:
#         all_df[col] = np.log1p(all_df[col])

#     print("正在对数值特征进行 MinMax 归一化...")
#     mms = MinMaxScaler()
#     all_df[DENSE_FEATURES] = mms.fit_transform(all_df[DENSE_FEATURES]).astype(np.float32)

#     print(f"正在处理类别特征 (保留训练集Top{VOCAB_SIZE}高频值)...")
#     for feat in SPARSE_FEATURES:
#         all_df[feat] = all_df[feat].fillna('unk').astype(str)
#         train_feat_series = all_df[all_df['is_train'] == 1][feat]
#         top_values = train_feat_series.value_counts().index[:VOCAB_SIZE].tolist()
#         vocab_map = {v: i+1 for i, v in enumerate(top_values)}
#         all_df[feat] = all_df[feat].map(vocab_map).fillna(0).astype('int32')

#     print("正在生成DeepFM特征列...")
#     fixlen_feature_columns = [
#         SparseFeat(feat, vocabulary_size=all_df[feat].max() + 1, embedding_dim=EMBEDDING_DIM)
#         for feat in SPARSE_FEATURES
#     ] + [
#         DenseFeat(feat, 1) 
#         for feat in DENSE_FEATURES
#     ]

#     dnn_feature_columns = fixlen_feature_columns
#     linear_feature_columns = fixlen_feature_columns
#     feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

#     train = all_df[all_df['is_train'] == 1].reset_index(drop=True)
#     test = all_df[all_df['is_train'] == 0].reset_index(drop=True)
    
#     train_model_input = {name: train[name].values for name in feature_names}
#     test_model_input = {name: test[name].values for name in feature_names}
    
#     y = train['label'].values
    
#     del all_df, train, test
#     gc.collect()

#     oof_preds = np.zeros(y.shape[0])
#     test_preds = np.zeros(len(test_labels_true))
#     folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,random_state=SEED)
    
#     print(f"\n开始 {N_FOLDS} 折训练 (Batch: {BATCH_SIZE}, Epochs: {EPOCHS})...")
    
#     for fold, (train_idx, val_idx) in enumerate(folds.split(train_model_input[feature_names[0]], y)):
#         fold_start = time.time()
#         print(f"\n--- Fold {fold + 1} / {N_FOLDS} ---")
        
#         X_train_input = {name: train_model_input[name][train_idx] for name in feature_names}
#         y_train_fold = y[train_idx]
#         X_val_input = {name: train_model_input[name][val_idx] for name in feature_names}
#         y_val_fold = y[val_idx]
        
#         model = DeepFM(
#             linear_feature_columns, dnn_feature_columns, 
#             task='binary',
#             l2_reg_embedding=1e-5, 
#             l2_reg_dnn=1e-5,
#             dnn_dropout=DNN_DROPOUT,
#             dnn_hidden_units=(512, 256), 
#             dnn_use_bn=True,
#             device=DEVICE
#         )
        
#         model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
        
#         model.fit(
#             X_train_input, y_train_fold,
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             verbose=1,
#             validation_data=(X_val_input, y_val_fold)
#         )
        
#         val_pred = model.predict(X_val_input, batch_size=BATCH_SIZE*2)
#         oof_preds[val_idx] = val_pred.flatten()
        
#         test_pred_fold = model.predict(test_model_input, batch_size=BATCH_SIZE*2)
#         test_preds += test_pred_fold.flatten() / folds.n_splits
        
#         cur_auc = roc_auc_score(y_val_fold, val_pred)
#         print(f"Fold {fold+1} AUC: {cur_auc:.5f} (耗时: {time.time()-fold_start:.1f}s)")
        
#         del model, X_train_input, y_train_fold, X_val_input, y_val_fold
#         torch.cuda.empty_cache()
#         gc.collect()

#     print("\n" + "="*40)
#     print("DeepFM 基线最终报告")
#     print("="*40)
    
#     oof_auc = roc_auc_score(y, oof_preds)
#     oof_loss = log_loss(y, oof_preds)
#     print(f"Train OOF AUC  : {oof_auc:.5f}")
#     print(f"Train OOF Loss : {oof_loss:.5f}")
    
#     test_auc = roc_auc_score(test_labels_true, test_preds)
#     test_loss = log_loss(test_labels_true, test_preds)
#     print(f"Test Score AUC : {test_auc:.5f}")
#     print(f"Test Score Loss: {test_loss:.5f}")
#     print("="*40)
    
#     # ===================== 画 ROC 图（报告用） =====================
#     fpr, tpr, _ = roc_curve(test_labels_true, test_preds)
#     roc_auc = sk_auc(fpr, tpr)

#     plt.figure(figsize=(7,6))
#     plt.plot(fpr, tpr, lw=2, color='#9933cc', label=f'AUC = {roc_auc:.4f}')
#     plt.plot([0,1],[0,1], '--', color='gray')
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.title('DeepFM Test ROC Curve')
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.savefig('deepfm_single_roc.png', dpi=300)
#     plt.show()
#     # ==============================================================

#     # ===================== 保存预测结果 =====================
#     np.save("y_true.npy", test_labels_true)
#     np.save("pred_deepfm.npy", test_preds)
#     # ========================================================
    
#     print(f"\n流程结束！总耗时: {time.time() - start_total:.2f} s")

# if __name__ == '__main__':
#     main()


























# seed=2024(5/2)

# ========================================
# DeepFM 基线最终报告
# ========================================
# Train OOF AUC  : 0.79640
# Train OOF Loss : 0.45001
# Test Score AUC : 0.79524
# Test Score Loss: 0.44665
# ========================================

# 流程结束！总耗时: 795.19 s


#seed=2025(5/2)
# ========================================
# DeepFM 基线最终报告
# ========================================
# Train OOF AUC  : 0.79559
# Train OOF Loss : 0.45076
# Test Score AUC : 0.79515
# Test Score Loss: 0.44687
# ========================================

# 流程结束！总耗时: 772.59 s


#seed=2026(5/2)
# ========================================
# DeepFM 基线最终报告
# ========================================
# Train OOF AUC  : 0.79606
# Train OOF Loss : 0.45021
# Test Score AUC : 0.79521
# Test Score Loss: 0.44674
# ========================================

# 流程结束！总耗时: 785.63 s