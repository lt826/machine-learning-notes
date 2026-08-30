# util/common.py
import numpy as np
import pandas as pd


def data_preprocessing(file_path=None):
    """
    对数据做预处理 -> 时间格式化, 按照时间升序排列, 且对数据去重.

    Args:
        file_path: 数据文件路径，如果为 None 则使用默认路径 '../data/train.csv'

    Returns:
        预处理后的 DataFrame
    """
    # 1. 加载数据集
    if file_path is None:
        file_path = '../data/train.csv'

    data = pd.read_csv(file_path)
    print(f"加载数据文件：{file_path}")
    data.info()

    # 2. 检查列
    if 'time' not in data.columns:
        raise ValueError(f"数据中缺少 'time' 列，当前列名：{data.columns.tolist()}")

    # 3. 时间格式化, 转为: '%Y-%m-%d %H:%M:%S'
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 4. 按照时间升序排列
    data.sort_values('time', ascending=True, inplace=True)

    # 5. 去重
    data.drop_duplicates(inplace=True)

    # 6. 重置索引
    data.reset_index(drop=True, inplace=True)

    print(f"预处理完成，共 {len(data)} 条记录")
    return data


if __name__ == '__main__':
    # 测试
    train_data = data_preprocessing('../data/train.csv')
    print(f"\n训练数据形状：{train_data.shape}")
    print(train_data.head())

    # 测试加载测试数据
    test_data = data_preprocessing('../data/test.csv')
    print(f"\n测试数据形状：{test_data.shape}")
    print(test_data.head())