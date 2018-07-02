import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
app_launch_log_data = pd.read_table(
    'app_launch_log.txt',
    header=None,
    names=['user_id', 'day'])
user_activity_log_data = pd.read_table(
    'user_activity_log.txt',
    header=None,
    names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'])
video_create_log_data = pd.read_table(
    'video_create_log.txt',
    header=None,
    names=['user_id', 'day']
)
user_register_log_data = pd.read_table(
    'user_register_log.txt',
    header=None,
    names=['user_id', 'register_day', 'register_type', 'device type']
)
launch = app_launch_log_data
activity = user_activity_log_data
register = user_register_log_data
video = video_create_log_data

one_dataSet_train_path = 'dealed_data/one_dataSet_train'
one_dataSet_test_path = 'dealed_data/one_dataSet_test'
two_dataSet_train_path = 'dealed_data/two_dataSet_train'
two_dataSet_test_path = 'dealed_data/two_dataSet_test'
three_dataSet_train_path = 'dealed_data/three_dataSet_test'


def cut_data_as_time(new_dataset_path, begin_day, end_day):
    temp_register = register[(register['register_day'] >= begin_day)
                            & (register['register_day'] <= end_day)
                            ]
    temp_create = video[(video['day'] >= begin_day)
                            & (video['day'] <= end_day)
                            ]
    temp_launch = launch[(launch['day'] >= begin_day)
                            & (launch['day'] <= end_day)
                            ]
    temp_activity = activity[(activity ['day'] >= begin_day)
                            & (activity['day'] <= end_day)
                            ]
    temp_register.to_csv(new_dataset_path+'register.csv', index=False)
    temp_create.to_csv(new_dataset_path+'create.csv', index=False)
    temp_launch.to_csv(new_dataset_path+'launch.csv', index=False)
    temp_activity.to_csv(new_dataset_path+'activity.csv', index=False)
    print('数据取出成功，存入中')


begin_day = 1
end_day = 16
cut_data_as_time(one_dataSet_train_path,
                 begin_day=begin_day,
                 end_day=end_day)
begin_day = 17
end_day = 23
cut_data_as_time(one_dataSet_test_path,
                begin_day=begin_day,
                end_day=end_day
                )
print('第一数据集划分完成')
begin_day = 8
end_day = 23
cut_data_as_time(two_dataSet_train_path,
                 begin_day=begin_day,
                 end_day=end_day
                )
begin_day = 24
end_day = 30
cut_data_as_time(two_dataSet_test_path,
                begin_day=begin_day,
                end_day=end_day
                )
print('第二数据集划分完成')
begin_day = 1
end_day = 30
cut_data_as_time(three_dataSet_train_path, begin_day=begin_day, end_day=end_day)
print('第三数据集划分完成')
print('OK')