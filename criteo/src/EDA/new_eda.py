import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\train_small.txt"
TEST_PATH = r"D:\code\Python\dachuang\criteo\criteo\data\after_decompression\test_small.txt"
COL_NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f"{x:.4f}")
pd.set_option('display.max_colwidth', 10)

def load_data():
    print("正在读取数据...")
    train_df = pd.read_csv(
        TRAIN_PATH, sep='\t', names=COL_NAMES, header=None,
        dtype={col: str for col in SPARSE_FEATURES}
    )
    test_df = pd.read_csv(
        TEST_PATH, sep='\t', names=COL_NAMES, header=None,
        dtype={col: str for col in SPARSE_FEATURES}
    )
    for col in DENSE_FEATURES:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    return train_df, test_df

def data_statistics(train_df, test_df):
    print("\n" + "="*80)
    print("Criteo数据集核心统计信息（展示所有特征）")
    print("="*80)
    
    print("\n【1. 数据规模】")
    print(f"训练集：样本数 = {len(train_df):,} | 特征数 = {train_df.shape[1]}")
    print(f"测试集：样本数 = {len(test_df):,} | 特征数 = {test_df.shape[1]}")
    
    print("\n1.5 数据集前5行示例（含特征标题）】")
    print("训练集前5行：")
    print(train_df.head().to_string(index=False))
    print("\n测试集前5行：")
    print(test_df.head().to_string(index=False))
    
    print("\n2. 点击标签分布（label=1为点击）")
    train_ctr = train_df['label'].mean()
    test_ctr = test_df['label'].mean()
    print(f"训练集CTR = {train_ctr:.4f}（点击数：{train_df['label'].sum():,} / 总样本：{len(train_df):,}）")
    print(f"测试集CTR = {test_ctr:.4f}（点击数：{test_df['label'].sum():,} / 总样本：{len(test_df):,}）")
    
    print("\n【3. 数值特征（I1-I13）完整统计】")
    dense_stats = []
    for col in DENSE_FEATURES:
        train_miss_rate = (train_df[col] < 0).mean()
        test_miss_rate = (test_df[col] < 0).mean()
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
    dense_df = pd.DataFrame(dense_stats)
    print(dense_df.to_string(index=False))
    
    print("\n【4. 类别特征（C1-C26）完整统计】")
    sparse_stats = []
    for col in SPARSE_FEATURES:
        train_unique = train_df[col].nunique()
        test_unique = test_df[col].nunique()
        train_miss = (train_df[col].isin(['', '\\N', '-1'])).mean()
        test_miss = (test_df[col].isin(['', '\\N', '-1'])).mean()
        
        sparse_stats.append({
            '特征名': col,
            '训练集唯一值数': train_unique,
            '测试集唯一值数': test_unique,
            '训练集缺失率': train_miss,
            '测试集缺失率': test_miss
        })
    sparse_df = pd.DataFrame(sparse_stats)
    sparse_df['训练集唯一值数'] = sparse_df['训练集唯一值数'].apply(lambda x: f"{x:,}")
    sparse_df['测试集唯一值数'] = sparse_df['测试集唯一值数'].apply(lambda x: f"{x:,}")
    print(sparse_df.to_string(index=False))
    
    print("\n" + "="*80)

if __name__ == '__main__':
    train_df, test_df = load_data()
    data_statistics(train_df, test_df)