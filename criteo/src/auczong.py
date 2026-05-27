import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

# ===================== 固定路径 =====================
BASE_PATH = r"D:\code\Python\criteo"

# 加载全部模型预测 + 真实标签
y_true    = np.load(f"{BASE_PATH}\\y_true.npy")
pred_lr   = np.load(f"{BASE_PATH}\\pred_lr.npy")
pred_lgb  = np.load(f"{BASE_PATH}\\pred_lgb.npy")
pred_df   = np.load(f"{BASE_PATH}\\pred_deepfm.npy")
pred_base = np.load(f"{BASE_PATH}\\pred_base.npy")
pred_final= np.load(f"{BASE_PATH}\\pred_final.npy")

# ===================== 计算ROC&AUC工具函数 =====================
def get_roc_data(y, pred):
    fpr, tpr, _ = roc_curve(y, pred)
    return fpr, tpr, sk_auc(fpr, tpr)

# 逐个计算
fpr_lr, tpr_lr, auc_lr       = get_roc_data(y_true, pred_lr)
fpr_lgb, tpr_lgb, auc_lgb    = get_roc_data(y_true, pred_lgb)
fpr_df, tpr_df, auc_df       = get_roc_data(y_true, pred_df)
fpr_base, tpr_base, auc_base = get_roc_data(y_true, pred_base)
fpr_fin, tpr_fin, auc_fin    = get_roc_data(y_true, pred_final)

# ===================== 绘制最终对比ROC =====================
plt.figure(figsize=(10, 8))

# 五条曲线 配色区分明显
plt.plot(fpr_lr,    tpr_lr,    lw=2, label=f'LR          (AUC={auc_lr:.4f})',    color='#FF4500')
plt.plot(fpr_lgb,   tpr_lgb,   lw=2, label=f'LightGBM    (AUC={auc_lgb:.4f})',   color='#0066FF')
plt.plot(fpr_df,    tpr_df,    lw=2, label=f'DeepFM      (AUC={auc_df:.4f})',    color='#9933CC')
plt.plot(fpr_base,  tpr_base,  lw=2, label=f'Base Model  (AUC={auc_base:.4f})',  color='#2ECC71')
plt.plot(fpr_fin,   tpr_fin,   lw=2, label=f'Final Optim(AUC={auc_fin:.4f})',    color='#E63946')

# 随机基准线
plt.plot([0, 1], [0, 1], '--', color='gray', lw=1.5, alpha=0.6)

# 图表样式
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('All Models ROC Curve Comparison', fontsize=14, pad=15)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0, 1)
plt.ylim(0, 1.02)

# 保存高清图
plt.savefig(f"{BASE_PATH}\\all_models_final_roc.png", dpi=300, bbox_inches='tight')
plt.show()

# 控制台打印所有AUC，方便写论文表格
print("===== 所有模型 AUC 汇总 =====")
print(f"LR         : {auc_lr:.4f}")
print(f"LightGBM   : {auc_lgb:.4f}")
print(f"DeepFM     : {auc_df:.4f}")
print(f"Base Model : {auc_base:.4f}")
print(f"Final Opt  : {auc_fin:.4f}")