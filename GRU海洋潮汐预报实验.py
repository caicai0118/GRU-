#天津大学 蔡跃 GRU海洋潮汐预报实验
#加载运行库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.stats import probplot

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False 

import tensorflow as tf
tf.random.set_seed(123)
np.random.seed(123)

#设置种子值
import random as python_random
python_random.seed(123)

# 导入GRU相关库
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.utils import plot_model

# 导入归一化库
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score

#日期数据格式标准化
def clean_datetime(row):
    dt=row["date"]+' '+row["time"]
    return pd.to_datetime(dt)


#读取并处理数据

df_pjxg_2019=pd.concat([pd.read_csv('data\\盘锦新港基面数据20190101-20190630.txt',header=None,sep="\t"),
                        pd.read_csv('data\\盘锦新港基面数据20190701-20191231.txt',header=None,sep="\t")],ignore_index=True)
df_pjxg_2019.columns=["date","time","value"]
df_pjxg_2019=df_pjxg_2019[df_pjxg_2019["time"].str.endswith(":00:00")].copy()

df_pjxg_2020=pd.read_csv('data\\盘锦新港基面数据2020.txt',header=None,sep="\s+")
df_pjxg_2020.columns=["date","time","value"]
df_pjxg_2020=df_pjxg_2020[df_pjxg_2020["time"].str.endswith(":00")].copy()

df_pjxg_2021=pd.read_csv('data\\盘锦新港数据2021.txt',header=None,sep="\s+",)
df_pjxg_2021.columns=["date","time","value"]
df_pjxg_2021=df_pjxg_2021[df_pjxg_2021["time"].str.endswith(":00")].copy()

df_pjxg_2021_pred=pd.read_csv('data\\盘锦新港预报潮位2021_预测.txt',header=None,sep="\s+",)
df_pjxg_2021_pred.columns=["date","time","value"]
df_pjxg_2021_pred=df_pjxg_2021_pred[df_pjxg_2021_pred["time"].str.endswith(":00:00")].copy()

# 提取每小时整点的数据
df_pjxg_2019["dt"]=df_pjxg_2019.apply(clean_datetime, axis=1)
df_pjxg_2019=df_pjxg_2019.drop(["date","time"], axis=1).set_index("dt").resample("1h").max().fillna(method="ffill")

df_pjxg_2020["dt"]=df_pjxg_2020.apply(clean_datetime, axis=1)
df_pjxg_2020=df_pjxg_2020.drop(["date","time"], axis=1).set_index("dt").resample("1h").max().fillna(method="ffill")

df_pjxg_2021["dt"]=df_pjxg_2021.apply(clean_datetime, axis=1)
df_pjxg_2021=df_pjxg_2021.drop(["date","time"], axis=1).set_index("dt").resample("1h").max().fillna(method="ffill")

df_pjxg_2021_pred["dt"]=df_pjxg_2021_pred.apply(clean_datetime, axis=1)
df_pjxg_2021_pred=df_pjxg_2021_pred.drop(["date","time"], axis=1).set_index("dt").resample("1h").max().fillna(method="ffill")

# 2019 年 盘锦新港 数据趋势图
plt.figure(figsize=(9,4))
sns.lineplot(x=df_pjxg_2019.index,y="value",data=df_pjxg_2019, alpha=0.5, color='firebrick')

plt.margins(x=0.01,y=0.01)
plt.xlabel("观测日期",fontsize=12)
plt.ylabel("潮位",fontsize=12)
plt.title("每小时潮位值曲线",fontsize=16)
plt.savefig('images/01-每小时潮位值曲线(2019).png', bbox_inches='tight', pad_inches=0)


# 从上面的图看 潮位数据具有一些周期性
# 日内数据大概是有个12小时的短周期，
# 月度数据看起来是存在个15天的长周期

# 72小时
df_pjxg_2019_tmp=df_pjxg_2019.iloc[:24*7].copy()
plt.figure(figsize=(9,4))
sns.lineplot(x=df_pjxg_2019_tmp.index,y="value",data=df_pjxg_2019_tmp, marker="o", alpha=0.5, color='firebrick')

plt.margins(x=0.01,y=0.01)
plt.xlabel("观测时间",fontsize=12)
plt.ylabel("潮位",fontsize=12)
plt.title("每小时潮位值曲线(72小时)",fontsize=16)
plt.savefig('images/02-每小时潮位值曲线(72小时).png', bbox_inches='tight', pad_inches=0)
plt.show()


# 180 日
df_pjxg_2019_tmp=df_pjxg_2019.iloc[:24*30*3].copy()
plt.figure(figsize=(9,4))
sns.lineplot(x=df_pjxg_2019_tmp.index,y="value",data=df_pjxg_2019_tmp, alpha=0.5, color='firebrick')

plt.margins(x=0.01,y=0.01)
plt.xlabel("观测日期",fontsize=12)
plt.ylabel("潮位",fontsize=12)
plt.title("每小时潮位值曲线(90日)",fontsize=16)
plt.savefig('images/03-每小时潮位值曲线(90日).png', bbox_inches='tight', pad_inches=0)
plt.show()

#构建训练、验证、测试数据集
# 设置数据集
data_train=df_pjxg_2019.copy()
data_val=df_pjxg_2020.copy()
data_test=df_pjxg_2021.copy()

# 归一化到0 到 1 之间
scaler=MinMaxScaler()
scaler.fit(data_train)

normalized_train=scaler.transform(data_train)
normalized_val =scaler.transform(data_val)
normalized_test =scaler.transform(data_test)

# 构造训练batch
length=24*15
batch_size32=32

train_tsGenerator=TimeseriesGenerator(normalized_train,normalized_train,length=length,batch_size=batch_size32)
val_tsGenerator=TimeseriesGenerator(normalized_val,normalized_val,length=length,batch_size=batch_size32)
test_tsGenerator=TimeseriesGenerator(normalized_test,normalized_test,length=length,batch_size=batch_size32)

#模型构建
tf.keras.backend.clear_session()

n_features=1
model=Sequential()

# 神经网络第一层： 包含32个 GRU 单元，模型的输入值的维度是 (32,1)
model.add(GRU(32,return_sequences=True,input_shape=(length,n_features)))

# 添加个 dropout 层， 随机擅长1/4 的神经网络单元，能有效避免过拟合
model.add(Dropout(0.25))

# 神经网络第三层： 同样是包含32个 GRU 单元
model.add(GRU(32))

# 输出层
model.add(Dense(1))

# compile

model.compile(optimizer='adam', loss='mse')


#模型训练
# 设置checkpoint 用于保存训练中的最佳参数

model_checkpoint = tf.keras.callbacks.ModelCheckpoint("ckpt/my_checkpoint.h5", save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

model.fit(train_tsGenerator,validation_data=val_tsGenerator,epochs=500,callbacks=[model_checkpoint, early_stopping])

#训练结果保存
model.save('model/my_model')



#数据预测
model = keras.models.load_model('model/my_model')
df_pjxg_2021_mypred=model.predict(test_tsGenerator)

#对比测试数据集构建
df_pjxg_2021_compare=df_pjxg_2021.iloc[360:,:].copy()
df_pjxg_2021_compare.columns=["实际值"]
df_pjxg_2021_compare["GRU 预测值"]=scaler.inverse_transform(df_pjxg_2021_mypred)
df_pjxg_2021_compare["GRU 预测值"]=df_pjxg_2021_compare["GRU 预测值"].map(lambda x: round(x,1))
df_pjxg_2021_compare=pd.merge(df_pjxg_2021_compare, df_pjxg_2021_pred, left_index=True, right_index=True,how="inner")
df_pjxg_2021_compare.columns=["实际值","GRU 预测值","自有软件预测值"]

df_pjxg_2021_compare["GRU 预测误差"]=df_pjxg_2021_compare["实际值"]-df_pjxg_2021_compare["GRU 预测值"]
df_pjxg_2021_compare["自有软件预测误差"]=df_pjxg_2021_compare["实际值"]-df_pjxg_2021_compare["自有软件预测值"]

#mse 对比
mse_pjxg_2021_software=mean_squared_error(df_pjxg_2021_compare["实际值"], df_pjxg_2021_compare["自有软件预测值"])
mse_pjxg_2021_lstm=mean_squared_error(df_pjxg_2021_compare["实际值"], df_pjxg_2021_compare["GRU 预测值"])
print(mse_pjxg_2021_software,mse_pjxg_2021_lstm)

# r^2 对比
r2_pjxg_2021_software=r2_score(df_pjxg_2021_compare["实际值"], df_pjxg_2021_compare["自有软件预测值"])
r2_pjxg_2021_lstm=r2_score(df_pjxg_2021_compare["实际值"], df_pjxg_2021_compare["GRU 预测值"])
print(r2_pjxg_2021_software,r2_pjxg_2021_lstm)


#对比画图
#全数据图
plt.figure(figsize=(9,4))
plt.plot(df_pjxg_2021_compare.index,df_pjxg_2021_compare["实际值"],color="firebrick",linewidth=1, alpha=0.25, label="实际值")
plt.plot(df_pjxg_2021_compare.index,df_pjxg_2021_compare["GRU 预测值"],color="orange", linewidth=1,alpha=0.25, label="GRU 预测值")
plt.plot(df_pjxg_2021_compare.index,df_pjxg_2021_compare["自有软件预测值"],color="gray", linewidth=1,alpha=0.25, label="自有软件预测值")

plt.legend()
plt.margins(x=0.01,y=0.01)

plt.xlabel("日期",fontsize=12)
plt.ylabel("潮位值",fontsize=12)
plt.title("实际潮位与预测潮位对比(盘锦新港)",fontsize=16)
plt.savefig('images/05_1-实际潮位与预测潮位对比(盘锦新港).png', bbox_inches='tight', pad_inches=0)
plt.show()

# 7日周期

df_pjxg_2021_compare_plot=df_pjxg_2021_compare.iloc[-1*7*24:,:]

plt.figure(figsize=(9,4))
plt.plot(df_pjxg_2021_compare_plot.index,df_pjxg_2021_compare_plot["实际值"],color="firebrick",linewidth=2, alpha=0.75, label="实际值")
plt.plot(df_pjxg_2021_compare_plot.index,df_pjxg_2021_compare_plot["GRU 预测值"],color="orange", linewidth=2,alpha=0.75, label="GRU 预测值")
plt.plot(df_pjxg_2021_compare_plot.index,df_pjxg_2021_compare_plot["自有软件预测值"],color="gray", linewidth=2,alpha=0.75, label="自有软件预测值")

plt.legend()
plt.margins(x=0.01,y=0.01)
plt.xlabel("日期",fontsize=12)
plt.ylabel("潮位值",fontsize=12)
plt.title("实际潮位与预测潮位对比(盘锦新港)",fontsize=16)
plt.savefig('images/05_2-实际潮位与预测潮位对比(盘锦新港).png', bbox_inches='tight', pad_inches=0)
plt.show()

#误差与实际值
plt.figure(figsize=(9,4))
plt.scatter(df_pjxg_2021_compare["实际值"],df_pjxg_2021_compare["GRU 预测误差"], s=1,alpha=0.5,label="GRU 预测误差")
plt.scatter(df_pjxg_2021_compare["实际值"],df_pjxg_2021_compare["自有软件预测误差"], s=1,alpha=0.5, label="自有软件预测误差")
plt.axhline(linestyle='--',color="gray")
plt.legend()
plt.margins(x=0.01,y=0.01)

plt.xlabel("实际值",fontsize=12)
plt.ylabel("预测误差",fontsize=12)
plt.title("预测误差对比(盘锦新港)",fontsize=16)
plt.savefig('images/06-预测误差对比(盘锦新港).png', bbox_inches='tight', pad_inches=0)
plt.show()