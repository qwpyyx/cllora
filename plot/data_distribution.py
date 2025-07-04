import pandas as pd
import matplotlib.pyplot as plt

# 假设 client_datasets 是 partition_dataset 返回的列表
# alpha 是你用来划分时传进去的参数


def compare(client_datasets, alpha):
    # 1. 统计每个 client 上每个类别的样本数
    client_counts = []
    for ds in client_datasets:
        # 把 Dataset 转成 pandas.DataFrame
        df = ds.to_pandas()
        # 从嵌套的 Instance dict 中拿到真正的 label
        labels = df['Instance'].apply(lambda x: x['label'])
        client_counts.append(labels.value_counts())

    # 2. 转成 DataFrame：行是 client，列是类别，缺失值补 0
    counts_df = pd.DataFrame(client_counts).fillna(0)

    # 如果你想看比例而不是绝对数，可以 normalize：
    counts_prop = counts_df.div(counts_df.sum(axis=1), axis=0)

    # 3. 画堆叠柱状图
    plt.figure(figsize=(12, 6))
    counts_prop.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6)
    )
    plt.xlabel("Client ID")
    plt.ylabel("bili")
    plt.title(f"w")
    plt.legend(title="label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()