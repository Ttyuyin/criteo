import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ================= 配置参数 =================
TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"
COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]  # 所有数值特征（I1-I13）
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]  # 所有类别特征（C1-C26）

# 设置pandas显示参数，确保所有列/行都显示，不折叠
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', 1000)        # 加宽显示宽度
pd.set_option('display.float_format', lambda x: f"{x:.4f}")  # 数值格式化
pd.set_option('display.max_colwidth', 10)   # 限制列宽，避免单行过长

# ================= 读取数据（仅读取必要列，提升速度） =================
def load_data():
    print("正在读取数据...")
    # 读取数据，数值特征转数值类型，类别特征暂存为字符串
    train_df = pd.read_csv(
        TRAIN_PATH, sep='\t', names=COL_NAMES, header=None,
        dtype={col: str for col in SPARSE_FEATURES}  # 类别特征避免解析错误
    )
    test_df = pd.read_csv(
        TEST_PATH, sep='\t', names=COL_NAMES, header=None,
        dtype={col: str for col in SPARSE_FEATURES}
    )
    # 数值特征转数值类型（处理缺失值标记<0）
    for col in DENSE_FEATURES:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    return train_df, test_df

# ================= 核心统计函数 =================
def data_statistics(train_df, test_df):
    print("\n" + "="*80)
    print("Criteo数据集核心统计信息（展示所有特征）")
    print("="*80)
    
    # 1. 基础规模统计
    print("\n【1. 数据规模】")
    print(f"训练集：样本数 = {len(train_df):,} | 特征数 = {train_df.shape[1]}")
    print(f"测试集：样本数 = {len(test_df):,} | 特征数 = {test_df.shape[1]}")
    
    # 新增：数据集前5行示例（含特征标题）
    print("\n【1.5 数据集前5行示例（含特征标题）】")
    print("▌训练集前5行：")
    print(train_df.head().to_string(index=False))  # index=False去掉行号，只显示特征标题+数据
    print("\n▌测试集前5行：")
    print(test_df.head().to_string(index=False))
    
    # 2. 标签分布（CTR核心）
    print("\n【2. 点击标签分布（label=1为点击）】")
    train_ctr = train_df['label'].mean()
    test_ctr = test_df['label'].mean()
    print(f"训练集CTR = {train_ctr:.4f}（点击数：{train_df['label'].sum():,} / 总样本：{len(train_df):,}）")
    print(f"测试集CTR = {test_ctr:.4f}（点击数：{test_df['label'].sum():,} / 总样本：{len(test_df):,}）")
    
    # 3. 数值特征（I1-I13）统计（缺失率+均值+中位数）—— 展示所有
    print("\n【3. 数值特征（I1-I13）完整统计】")
    dense_stats = []
    for col in DENSE_FEATURES:
        # Criteo中<0表示缺失值
        train_miss_rate = (train_df[col] < 0).mean()
        test_miss_rate = (test_df[col] < 0).mean()
        # 有效数值的均值/中位数（过滤<0）
        train_valid = train_df[train_df[col] >= 0][col]
        test_valid = test_df[test_df[col] >= 0][col]
        
        dense_stats.append({
            '特征名': col,
            '训练集缺失率': train_miss_rate,
            '测试集缺失率': test_miss_rate,
            '训练集均值': train_valid.mean(),
            '测试集均值': test_valid.mean(),
            '训练集中位数': train_valid.median(),
            '测试集中位数': test_valid.median()
        })
    # 打印所有数值特征统计（格式优化）
    dense_df = pd.DataFrame(dense_stats)
    print(dense_df.to_string(index=False))
    
    # 4. 类别特征（C1-C26）统计（唯一值数量+缺失率）—— 展示所有
    print("\n【4. 类别特征（C1-C26）完整统计】")
    sparse_stats = []
    for col in SPARSE_FEATURES:
        train_unique = train_df[col].nunique()
        test_unique = test_df[col].nunique()
        # 缺失值（Criteo中常见用'\N'或空表示）
        train_miss = (train_df[col].isin(['', '\\N', '-1'])).mean()
        test_miss = (test_df[col].isin(['', '\\N', '-1'])).mean()
        
        sparse_stats.append({
            '特征名': col,
            '训练集唯一值数': train_unique,
            '测试集唯一值数': test_unique,
            '训练集缺失率': train_miss,
            '测试集缺失率': test_miss
        })
    # 打印所有类别特征统计（格式优化）
    sparse_df = pd.DataFrame(sparse_stats)
    # 唯一值数格式化（带千分位）
    sparse_df['训练集唯一值数'] = sparse_df['训练集唯一值数'].apply(lambda x: f"{x:,}")
    sparse_df['测试集唯一值数'] = sparse_df['测试集唯一值数'].apply(lambda x: f"{x:,}")
    print(sparse_df.to_string(index=False))
    
    print("\n" + "="*80)

# ================= 主函数 =================
if __name__ == '__main__':
    train_df, test_df = load_data()
    data_statistics(train_df, test_df)



# 正在读取数据...

# ================================================================================
# Criteo数据集核心统计信息（展示所有特征）
# ================================================================================

# 【1. 数据规模】
# 训练集：样本数 = 5,000,000 | 特征数 = 40
# 测试集：样本数 = 1,000,000 | 特征数 = 40

# 【1.5 数据集前5行示例（含特征标题）】
# ▌训练集前5行：
#  label     I1  I2      I3      I4        I5      I6      I7     I8       I9    I10    I11    I12     I13       C1       C2       C3       C4       C5       C6       C7       C8       C9      C10      C11      C12      C13      C14      C15      C16      C17      C18      C19      C20      C21      C22      C23      C24      C25      C26
#      0 1.0000   1  5.0000  0.0000 1382.0000  4.0000 15.0000 2.0000 181.0000 1.0000 2.0000    NaN  2.0000 68fd1e64 80e26c9b fb936136 7b4723c4 25c83c98 7e0ccccf de7995b8 1f89b562 a73ee510 a8cd5504 b2cb9c98 37c9c164 2824a5f6 1adce6ef 8ba8b39a 891b62e7 e5ba7672 f54016b9 21ddcdc9 b1252a9d 07b5194c      NaN 3a171ecb c5c50484 e8b83407 9727dd16
#      0 2.0000   0 44.0000  1.0000  102.0000  8.0000  2.0000 2.0000   4.0000 1.0000 1.0000    NaN  4.0000 68fd1e64 f0cf0024 6f67f7e5 41274cd7 25c83c98 fe6b92e5 922afcc0 0b153874 a73ee510 2b53e5fb 4f1b46f3 623049e6 d7020589 b28479f6 e6c5b5cd c92f3b61 07c540c4 b04e4670 21ddcdc9 5840adea 60f6221e      NaN 3a171ecb 43f13e8b e8b83407 731c3655
#      0 2.0000   0  1.0000 14.0000  767.0000 89.0000  4.0000 2.0000 245.0000 1.0000 3.0000 3.0000 45.0000 287e684f 0a519c5c 02cf9876 c18be181 25c83c98 7e0ccccf c78204a1 0b153874 a73ee510 3b08e48b 5f5e6091 8fe001f4 aa655a2f 07d13a8f 6dc710ed 36103458 8efede7f 3412118d      NaN      NaN e587c466 ad3062eb 3a171ecb 3b183c5c      NaN      NaN
#      0    NaN 893     NaN     NaN 4392.0000     NaN  0.0000 0.0000   0.0000    NaN 0.0000    NaN     NaN 68fd1e64 2c16a946 a9a87e68 2e17d6f6 25c83c98 fe6b92e5 2e8a689b 0b153874 a73ee510 efea433b e51ddf94 a30567ca 3516f6e6 07d13a8f 18231224 52b8680f 1e88c74f 74ef3502      NaN      NaN 6b3a5ca6      NaN 3a171ecb 9117a34a      NaN      NaN
#      0 3.0000  -1     NaN  0.0000    2.0000  0.0000  3.0000 0.0000   0.0000 1.0000 1.0000    NaN  0.0000 8cf07265 ae46a29d c81688bb f922efad 25c83c98 13718bbd ad9fa255 0b153874 a73ee510 5282c137 e5d8af57 66a76a26 f06c53ac 1adce6ef 8ff4b403 01adbab4 1e88c74f 26b3c7a7      NaN      NaN 21c9516a      NaN 32c7478e b34f3128      NaN      NaN

# ▌测试集前5行：
#  label       I1  I2     I3      I4         I5       I6       I7      I8       I9    I10    I11  I12     I13       C1       C2       C3       C4       C5       C6       C7       C8       C9      C10      C11      C12      C13      C14      C15      C16      C17      C18      C19      C20      C21 C22      C23      C24      C25      C26        
#      0   0.0000  76 2.0000  2.0000  1548.0000 149.0000   1.0000  2.0000  53.0000 0.0000 1.0000  NaN  2.0000 68fd1e64 38d50e09 92eb3174 88e439d9 25c83c98 7e0ccccf 0ec6f284 0b153874 a73ee510 c1c39cbf a7b606c4 604f499b eae197fd b28479f6 06373944 8e662061 3486227d fffe2a63 21ddcdc9 b1252a9d 872c22d6 NaN 3a171ecb df487a73 001f3601 c27f155b        
#      0   0.0000   1 7.0000 13.0000  3126.0000 168.0000   3.0000 13.0000  46.0000 0.0000 1.0000  NaN 13.0000 f473b8dc 2c16a946 8f6db6a5 65723153 f281d2a7 7e0ccccf 38eb9cf4 0b153874 a73ee510 e9995d97 7f8ffe57 756c3bd1 46f42a63 b28479f6 3628a186 c0c7a39b d4bb7bd8 e4ca448c      NaN      NaN fd47484b NaN 32c7478e 9117a34a      NaN      NaN        
#      0   0.0000   0 1.0000  0.0000 14856.0000 731.0000   1.0000  4.0000  61.0000 0.0000 1.0000  NaN  3.0000 68fd1e64 ae46a29d 7caddb0d f922efad 4cf72387 fe6b92e5 84057fed 0b153874 a73ee510 3b08e48b caecb243 9bf4fa6a 85fd1cb8 b28479f6 bfb03e99 01adbab4 d4bb7bd8 1a47ef6b      NaN      NaN 21c9516a NaN 32c7478e b34f3128      NaN      NaN        
#      1   0.0000   1 3.0000     NaN  4688.0000 144.0000  12.0000 11.0000 117.0000 0.0000 2.0000  NaN     NaN 68fd1e64 09e68b86 aa8c1539 85dd697c 25c83c98 fe6b92e5 9d8d7034 0b153874 a73ee510 b3d657b8 51ef0313 d8c29807 e8f6ccfe b28479f6 2d49999f c64d548f e5ba7672 63cdbb21 cf99e5de 5840adea 5f957280 NaN 32c7478e 1793a828 e8b83407 b7d9c3bc        
#      0 131.0000   0 7.0000  4.0000     5.0000   4.0000 131.0000  4.0000   4.0000 1.0000 1.0000  NaN  4.0000 be589b51 287130e0 0c38a323 7bbfd67a 25c83c98 7e0ccccf ae82fae4 0b153874 a73ee510 d229fbfe 785a24cb 9a103204 0c7dd611 1adce6ef 310d155b 0fda2db5 e5ba7672 891589e7 4764bf77 5840adea 1dc95fcd NaN 85d5a995 e29de47e ea9a246c 164d3259        

# 【2. 点击标签分布（label=1为点击）】
# 训练集CTR = 0.2506（点击数：1,252,825 / 总样本：5,000,000）
# 测试集CTR = 0.2456（点击数：245,631 / 总样本：1,000,000）

# 【3. 数值特征（I1-I13）完整统计】
# 特征名  训练集缺失率  测试集缺失率      训练集均值      测试集均值    训练集中位数    测试集中位数
#  I1  0.0000  0.0000     3.2529     3.3402    1.0000    1.0000
#  I2  0.1095  0.0936   119.7497   117.8662    5.0000    6.0000
#  I3  0.0000  0.0000    20.8883    24.2592    6.0000    7.0000
#  I4  0.0000  0.0000     7.0956     7.8909    4.0000    5.0000
#  I5  0.0000  0.0000 19074.7167 18765.8664 2704.0000 2903.0000
#  I6  0.0000  0.0000   115.4327   122.7866   33.0000   34.0000
#  I7  0.0000  0.0000    14.7579    16.0685    3.0000    3.0000
#  I8  0.0000  0.0000    12.7532    12.1927    7.0000    7.0000
#  I9  0.0000  0.0000   102.7514   106.9973   37.0000   37.0000
# I10  0.0000  0.0000     0.5868     0.5718    0.0000    0.0000
# I11  0.0000  0.0000     2.4993     2.6460    1.0000    1.0000
# I12  0.0000  0.0000     1.0303     0.8491    0.0000    0.0000
# I13  0.0000  0.0000     7.9446     9.1842    4.0000    5.0000

# 【4. 类别特征（C1-C26）完整统计】
# 特征名   训练集唯一值数 测试集唯一值数  训练集缺失率  测试集缺失率
#  C1     1,396   1,247  0.0000  0.0000
#  C2       549     531  0.0000  0.0000
#  C3 1,373,638 336,682  0.0000  0.0000
#  C4   406,654 133,369  0.0000  0.0000
#  C5       290     264  0.0000  0.0000
#  C6        20      15  0.0000  0.0000
#  C7    11,862  10,813  0.0000  0.0000
#  C8       607     553  0.0000  0.0000
#  C9         3       3  0.0000  0.0000
# C10    53,574  31,069  0.0000  0.0000
# C11     5,173   4,708  0.0000  0.0000
# C12 1,156,253 303,424  0.0000  0.0000
# C13     3,119   3,082  0.0000  0.0000
# C14        26      26  0.0000  0.0000
# C15    11,689   8,926  0.0000  0.0000
# C16   833,956 234,188  0.0000  0.0000
# C17        10      10  0.0000  0.0000
# C18     4,710   3,867  0.0000  0.0000
# C19     2,061   1,819  0.0000  0.0000
# C20         3       3  0.0000  0.0000
# C21 1,015,597 274,585  0.0000  0.0000
# C22        16      12  0.0000  0.0000
# C23        15      15  0.0000  0.0000
# C24    95,859  42,309  0.0000  0.0000
# C25        89      71  0.0000  0.0000
# C26    64,258  31,034  0.0000  0.0000