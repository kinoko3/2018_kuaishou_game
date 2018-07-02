# 高校大数据比赛 快手+清华 

* data_process.py 创建训练集和验证集
* feature.py 特征提取
* model.py 模型构建

[数据集下载链接，Google Drive](https://drive.google.com/open?id=1nFgrc1YPzXqUszdS15-IjYhv6jOSCGB4)

## 比赛数据分析

---
### user_activity_log.txt 600M 用户活动纪录

* #### user_id 用户id 
* #### day 日期
* #### page 行为发生的日志, 每个数字分别对应“关注页”、”个人主页“、”发现页“、”同城页“或”其他页“中的一个
* #### video_id 视频id
* #### author_id 作者id
* #### action_type 用户行为类型。每个数字分别对应“播放“、”关注“、”点赞“、”转发“、”举报“和”减少此类作品“中的一个

---
### user_register_log.txt 700KB 用户注册纪录

* ####  user_id  用户名
* #### register_day 注册日期
* #### register_type 注册类型
* #### device_type  注册设备类型 

---
###  video_create_log.txt 700KB 用户视频拍摄纪录
* #### user_id 用户id
* #### day 拍摄日期

---
### app_launch_log.txt   APP启动日志
* #### user_id 用户id
* #### day 日期
---
## 大致框架

### 该题属于时间序列的用户画像，通过前30天的特征预测后7天活跃用户。

### 使用滑窗法处理数据。对于一个用户是否活跃的判断条件是是否出现在4个数据文件中。

### 对于数据集的划分为（初步计划）：

#### 因为数据集咩有label，我们使用1~16天的数据作为训练，看17~23天是否还在活跃来进行打label

#### 使用两个函数，一个函数使用训练集和验证集判断对训练集打label，一个函数用来处理最后预测7天的测试集。函数主要使用`np.unique`来保证user_id唯一性。

* ### 1~16 训练集 17 ~ 23 验证集
* ### 8~23 训练集 24 ~ 30 验证集
* ### 15~30 测试集 30 ~ 37 预测用户

## 特征提取

> 以下函数都是基于groupby的自定义处理函数，基于每一个user_id
> 主要特征有：(所有数据通用特征)
> * 均值 Mean
> * 标准差 Std
> * 最小值 Min
> * 最大值 Max
> * 众数 Mode
> * 偏度 Skewness
> * 峰度 Kurtosis
> * 计数 Count
> * 当前序列的之后一个(比如最后一天登录) last
> * 以上为统计多连
> * 在文件中最后一次活跃减去注册日期(register) X_SUB_X
> * 离散差值 diff （其实就是后一个元素减前一个元素）

### 对于特征提取这块，我们要分析这个题目的业务逻辑，什么人怎样用快手之后会在30天后的7天内还在继续用快手。

### get_create_feature
> 先对该用户的所有拍摄天数进行diff处理

* diff 统计多连！！！
* feature['create_mode'] 一天最多创建多少
* feature['create_sub_register']  拍摄天书减去创建天数



### get_activity_feature
> 该比赛中最庞大的特征工程

* feature['activity_count'] 活动计数
* 对天数进行diff，然后统计多连
* 对每一页的访问计数，每一页的访问频次(计数/总数)(包括行为) page0~page4 action0~action5
* feature['video_id_mode']  `important`
* feature['author_id_mode'] `important`
* 总活动数统计多连
* 总活动数统计好之后，diff处理，然后统计多连
---
#### 下面才是最可怕的特征，占了50%的特征量
* 构建一个函数，对每个用户的每天活动进行计数，计数包括page、action，一共9项
* 对这9项分别进行统计多连(action4，action5经过重要性分析后几乎重要性为0)
* 7 * 9=72项特征(去掉mode和count)


### get_launch_feature
> 先对该用户的所有拍摄天数进行diff处理

* diff 统计多连！！！

### get_register_feature
* feature['device_type']   
* feature['register_day']
* feature['register_type'] 


> 还有更多的magic feature等待发现

---

### 算法模型

模型随便选一个都可以做，LSTM是有用的，XGBoost可以， lightgbm可以。传统机器学习也可以。


注意一点: 因为数据量小，注意参数大小，过拟合会弄死人的，我已经被这个弄得欲仙欲死了。


对于优化，有两个标准，一个是看F1  Score，来更改参数，一个是固定提交数，然后看线上F1分数，F1数在线下验证容易被过拟合。


