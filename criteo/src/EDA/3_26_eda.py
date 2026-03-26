import pandas as pd
import numpy as np

# =========================
# 1. 文件路径
# =========================
TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"

# =========================
# 2. 列名定义
# =========================
columns = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]


# =========================
# 3. 读取数据
# =========================
def load_criteo_data(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        low_memory=False
    )
    return df


# =========================
# 4. 数值特征统计
# =========================
def analyze_dense_features(df):
    dense_stats = []

    for col in dense_features:
        series = pd.to_numeric(df[col], errors="coerce")
        stat = {
            "feature": col,
            "mean": round(series.mean(), 2),
            "median": round(series.median(), 2),
            "max": round(series.max(), 2),
            "missing_count": int(series.isnull().sum()),
            "missing_rate(%)": round(series.isnull().mean() * 100, 2)
        }
        dense_stats.append(stat)

    return pd.DataFrame(dense_stats)


# =========================
# 5. 类别特征统计
# =========================
def analyze_sparse_features(df):
    sparse_stats = []
    total_samples = len(df)

    for col in sparse_features:
        series = df[col]
        missing_count = series.isnull().sum()
        unique_count = series.nunique(dropna=True)

        stat = {
            "feature": col,
            "unique_count": int(unique_count),
            "missing_count": int(missing_count),
            "missing_rate(%)": round(missing_count / total_samples * 100, 2),
            "sparsity_ratio": round(unique_count / total_samples, 6)
        }
        sparse_stats.append(stat)

    return pd.DataFrame(sparse_stats)


# =========================
# 6. 主程序
# =========================
if __name__ == "__main__":
    print("开始读取训练集...")
    train_df = load_criteo_data(TRAIN_PATH)
    print(f"训练集读取完成: {train_df.shape}")

    print("开始读取测试集...")
    test_df = load_criteo_data(TEST_PATH)
    print(f"测试集读取完成: {test_df.shape}")

    # 训练集统计
    train_dense_stats = analyze_dense_features(train_df)
    train_sparse_stats = analyze_sparse_features(train_df)

    # 测试集统计
    test_dense_stats = analyze_dense_features(test_df)
    test_sparse_stats = analyze_sparse_features(test_df)

    # 保存结果
    train_dense_stats.to_csv("train_dense_stats.csv", index=False, encoding="utf-8-sig")
    train_sparse_stats.to_csv("train_sparse_stats.csv", index=False, encoding="utf-8-sig")
    test_dense_stats.to_csv("test_dense_stats.csv", index=False, encoding="utf-8-sig")
    test_sparse_stats.to_csv("test_sparse_stats.csv", index=False, encoding="utf-8-sig")

    print("\n===== 训练集数值特征统计（前几行）=====")
    print(train_dense_stats.head())

    print("\n===== 训练集类别特征统计（前几行）=====")
    print(train_sparse_stats.head())

    print("\n===== 测试集数值特征统计（前几行）=====")
    print(test_dense_stats.head())

    print("\n===== 测试集类别特征统计（前几行）=====")
    print(test_sparse_stats.head())

    # 推荐论文展示的代表性特征
    key_dense = ["I2", "I5", "I6", "I9"]
    key_sparse = ["C3", "C12", "C16", "C21"]

    print("\n===== 论文推荐展示：训练集数值特征 =====")
    print(train_dense_stats[train_dense_stats["feature"].isin(key_dense)])

    print("\n===== 论文推荐展示：训练集类别特征 =====")
    print(train_sparse_stats[train_sparse_stats["feature"].isin(key_sparse)])

    print("\n统计结果已保存为 CSV 文件。")




# (base) PS D:\code\Python\criteo> & D:/code/AnacondaEnvs/dl_env/python.exe d:/code/Python/criteo/criteo/src/EDA/3_26_eda.py
# 开始读取训练集...
# 训练集读取完成: (5000000, 40)
# 开始读取测试集...
# 测试集读取完成: (1000000, 40)

# ===== 训练集数值特征统计（前几行）=====
#   feature      mean  median        max  missing_count  missing_rate(%)
# 0      I1      3.25     1.0     1209.0        2177685            43.55
# 1      I2    106.52     3.0    22066.0              0             0.00
# 2      I3     20.89     6.0    65535.0        1090400            21.81
# 3      I4      7.10     4.0      561.0        1119846            22.40
# 4      I5  19074.72  2704.0  2623579.0         125205             2.50

# ===== 训练集类别特征统计（前几行）=====
#   feature  unique_count  missing_count  missing_rate(%)  sparsity_ratio
# 0      C1          1396              0             0.00        0.000279
# 1      C2           549              0             0.00        0.000110
# 2      C3       1373638         167237             3.34        0.274728
# 3      C4        406654         167237             3.34        0.081331
# 4      C5           290              0             0.00        0.000058

# ===== 测试集数值特征统计（前几行）=====
#   feature      mean  median        max  missing_count  missing_rate(%)
# 0      I1      3.34     1.0     1013.0         456440            45.64
# 1      I2    106.74     3.0    20881.0              0             0.00
# 2      I3     24.26     7.0    65535.0         194611            19.46
# 3      I4      7.89     5.0      418.0         193692            19.37
# 4      I5  18765.87  2903.0  2636562.0          30934             3.09

# ===== 测试集类别特征统计（前几行）=====
#   feature  unique_count  missing_count  missing_rate(%)  sparsity_ratio
# 0      C1          1247              0             0.00        0.001247
# 1      C2           531              0             0.00        0.000531
# 2      C3        336682          32802             3.28        0.336682
# 3      C4        133369          32802             3.28        0.133369
# 4      C5           264              0             0.00        0.000264

# ===== 论文推荐展示：训练集数值特征 =====
#   feature      mean  median        max  missing_count  missing_rate(%)
# 1      I2    106.52     3.0    22066.0              0             0.00
# 4      I5  19074.72  2704.0  2623579.0         125205             2.50
# 5      I6    115.43    33.0   233523.0        1123057            22.46
# 8      I9    102.75    37.0    19327.0         210795             4.22

# ===== 论文推荐展示：训练集类别特征 =====
#    feature  unique_count  missing_count  missing_rate(%)  sparsity_ratio
# 2       C3       1373638         167237             3.34        0.274728
# 11     C12       1156253         167237             3.34        0.231251
# 15     C16        833956         167237             3.34        0.166791
# 20     C21       1015597         167237             3.34        0.203119

# 统计结果已保存为 CSV 文件。
# (base) PS D:\code\Python\criteo>