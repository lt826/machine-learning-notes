import sys
import os

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导包
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from util.log import Logger
from util.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error,mean_absolute_percentage_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


# 1. 定义电力负荷模型类, 配置日志, 获取数据源.
class PowerLoadModel:
    # 1.1 初始化属性信息.
    def __init__(self):
        # 1.2 拼接日志文件名.
        logfile_name = 'train_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # 1.3 创建日志对象.
        self.logfile = Logger('../', logfile_name).get_logger()
        # 测试写一条日志.
        self.logfile.info('开始创建 电力负荷模型类的 对象了')
        # 1.4 获取数据源.
        self.data_source = data_preprocessing()


# 2. 查看数据的整体分布情况.
def ana_data(data):     # analysis: 分析
    """
    1.查看数据整体情况
    2.负荷整体的分布情况
    3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data: 数据源
    :return:
    """
    # 0. 为了防止会修改源数据, 我们做一次拷贝.
    ana_data = data.copy()

    # 1. 查看数据整体情况
    ana_data.info()

    # 2. 负荷整体的分布情况, 直方图.
    # 2.1 创建画布.
    fig = plt.figure(figsize=(20, 40))
    # 2.2 添加子图.
    ax1 = fig.add_subplot(411)
    ax1.hist(ana_data['power_load'], bins=100)      # 负荷, 直方图, 100个区间
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')

    # 3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    # 3.1 新增1列, 充当小时.
    ana_data['hour'] = ana_data['time'].str[11:13]
    # 3.2 根据小时分组, 计算平均值.
    hour_load_mean = ana_data.groupby('hour', as_index=False)['power_load'].mean()
    # print(hour_load_mean)       # [列1 hour, 列2 power_load 当前小时的平均负荷]

    # # 3.3 画出折线图.
    ax2 = fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'], hour_load_mean['power_load'])
    ax2.set_title('各个小时的平均负荷趋势')
    ax2.set_xlabel('小时')

    # 4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    ana_data['mouth'] = ana_data['time'].str[5:7]
    # 3.2 根据月份分组, 计算平均值.
    mouth_load_mean = ana_data.groupby('mouth', as_index=False)['power_load'].mean()
    # print(hour_load_mean)       # [列1 mouth, 列2 power_load 当前月份的平均负荷]

    # # 3.3 画出折线图.
    ax2 = fig.add_subplot(413)
    ax2.plot(mouth_load_mean['mouth'], mouth_load_mean['power_load'])
    ax2.set_title('各个月份的平均负荷趋势')
    ax2.set_xlabel('月份')

    # 5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    ana_data['weekday'] = ana_data['time'].apply(lambda x: pd.to_datetime(x).weekday())
    ana_data['is_weekend'] = ana_data['weekday'].apply(lambda x: 1 if x >= 5 else 0)        # 1: 周末, 0: 周一至周五
    work_load_mean = ana_data[ana_data['is_weekend'] == 0].power_load.mean()
    weekend_load_mean = ana_data[ana_data['is_weekend'] == 1].power_load.mean()

    ax3 = fig.add_subplot(414)
    ax3.bar(['工作日', '周末'], [work_load_mean, weekend_load_mean])
    ax3.set_title('工作日与周末的平均负荷情况')

    plt.savefig('../data/fig/负荷整体的分布情况.png')
    plt.show()


# 3. 特征工程.
def feature_engineering(data,log):
    feature_data = data.copy()
    #1.提取出时间特征：小时、月份
    feature_data['hour'] = feature_data['time'].str[11:13]
    feature_data['month'] = feature_data['time'].str[5:7]
    #热编码one-hot处理hour和month字段
    hour_month_data = pd.get_dummies(feature_data[['hour', 'month']])
    feature_data = pd.concat([feature_data, hour_month_data], axis=1)
    # print(feature_data.head(10))
    # print(feature_data.info())

    #2.提取出相近时间窗口的负荷作为特征，step大小窗口的负荷
    #前1小时、2小时、3小时的负荷作为特征
    load_1h_data = feature_data['power_load'].shift(1)
    load_2h_data = feature_data['power_load'].shift(2)
    load_3h_data = feature_data['power_load'].shift(3)
    #拼接前1小时、2小时、3小时的负荷
    load_shift_data = pd.concat([load_1h_data, load_2h_data, load_3h_data], axis=1)
    #修改列名
    load_shift_data.columns = ['前1小时','前2小时','前3小时']
    feature_data = pd.concat([feature_data, load_shift_data], axis=1)
    # print(feature_data.info())

    #3.提取昨日同一时刻的负荷作为特征
    feature_data['yesterday_time'] = feature_data['time'].apply(lambda x: (pd.to_datetime(x) - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    #把所有日期和负荷拼接成字典，方便查找
    time_load_dict = feature_data.set_index('time')['power_load'].to_dict()
    #新增一列yesterday_load，表示：昨日同一时刻的负荷
    feature_data['yesterday_load'] = feature_data['yesterday_time'].apply(lambda x: time_load_dict.get(x))
    feature_data.info()

    #4.因为第一天的负荷无法提取出yesterday_load，所以删除出现空值的数据
    feature_data = feature_data.dropna()

    #5.整理时间特征，并返回
    feature_columns = list(hour_month_data.columns) + list(load_shift_data.columns) + ['yesterday_load']
    print(feature_columns)

    return feature_data,feature_columns


# 4. 模型训练, 评估.
def model_train(data, feature, log):
    #1.数据集的划分
    x = data[feature]
    y = data['power_load']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

    #2.网格搜索和交叉验证
    # log.info('进行网格搜索和交叉验证寻找最优超参')
    # log.info(f'开始时间：{datetime.datetime.now()}')
    # param_dict = {
    #     'n_estimators': [50, 100, 150],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.1, 0.01, 0.001]
    # }
    #
    # #创建XGBoost模型对象
    # estimator = XGBRegressor()
    # #创建网格搜索对象
    # gs = GridSearchCV(estimator, param_dict, cv=5)
    # gs.fit(x_train, y_train)
    # log.info(f"结束时间：{datetime.datetime.now()}")
    # log.info(f'最佳参数：{gs.best_params_}')
    # log.info(f'最佳分数：{gs.best_score_}')

    #3.模型实例化
    estimator = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1)
    #4.模型训练
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    #5.模型评估
    print(f'均方误差：{mean_squared_error(y_test, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y_test, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y_test, y_pred)}')
    print(f'平均绝对百分比误差：{mean_absolute_percentage_error(y_test, y_pred)}')

    #6.模型保存
    joblib.dump(estimator, '../model/xgb_20260830.pkl')
    log.info(f'模型保存完成,保存路径：../model/xgb_20260830.pkl')

# 5. 测试.
if __name__ == '__main__':
    # 5.1 创建电力负荷模型类的对象.
    pm = PowerLoadModel()
    # 5.2 打印数据源.
    # print(pm.data_source)

    # 5.3 查看数据分布.
    #ana_data(pm.data_source)

    # 5.4 特征工程
    feature_data, feature_columns = feature_engineering(pm.data_source, pm.logfile)

    # 5.5 模型训练
    # 参1：处理后的全部数据集，参2：特征列，参3：日志对象
    model_train(feature_data,feature_columns,pm.logfile)