import pandas as pd
from sklearn.model_selection import train_test_split

# 设置文件路径
train_data_path = "../data/ethics_raw/train.csv"  # 确保这个路径是正确的
df = pd.read_csv(train_data_path)

# 打印列名，查看实际列
print(df.columns)

# 只选择 'label' 和 'input' 列，重命名为 'label' 和 'text'
df = df[['label', 'input']]  # 只保留 'label' 和 'input' 列
df.columns = ['label', 'text']  # 设置列名

# 拆分训练和验证集
train_df, val_df = train_test_split(df, test_size=0.1)

# 保存处理后的数据
train_df.to_csv("../data/ethics_dataset_train.csv", index=False)
val_df.to_csv("../data/ethics_dataset_val.csv", index=False)
