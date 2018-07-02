from collections import Counter
from scipy.stats import stats
import pandas as pd
import numpy as np

one_dataSet_train_path = 'dealed_data/one_dataSet_train'
one_dataSet_test_path = 'dealed_data/one_dataSet_test'
two_dataSet_train_path = 'dealed_data/two_dataSet_train'
two_dataSet_test_path = 'dealed_data/two_dataSet_test'
three_dataSet_train_path = 'dealed_data/three_dataSet_test'

train_path = 'train-and-test/train.csv'
train_path_one = 'train-and-test/train_one.csv'
train_path_two = 'train-and-test/train_two.csv'
test_path_two = 'train-and-test/test_two.csv'
test_path = 'train-and-test/test.csv'

register = 'register.csv'
create = 'create.csv'
launch = 'launch.csv'
activity = 'activity.csv'


def day_cut_max_day(input):
    data = np.array(input)
    last_day = data[-1]
    data = np.delete(data, -1)
    max = 0
    for i in data:
        acc = last_day - i
        if acc > max:
            max = acc
    return max


def day_count(row_2):
    feature = pd.Series()
    feature['page0_count'] = len(np.where(row_2['page'] == 0)[0])
    feature['page1_count'] = len(np.where(row_2['page'] == 1)[0])
    feature['page2_count'] = len(np.where(row_2['page'] == 2)[0])
    feature['page3_count'] = len(np.where(row_2['page'] == 3)[0])
    feature['page4_count'] = len(np.where(row_2['page'] == 4)[0])
    feature['action0_count'] = len(np.where(row_2['action_type'] == 0)[0])
    feature['action1_count'] = len(np.where(row_2['action_type'] == 1)[0])
    feature['action2_count'] = len(np.where(row_2['action_type'] == 2)[0])
    feature['action3_count'] = len(np.where(row_2['action_type'] == 3)[0])
    feature['action4_count'] = len(np.where(row_2['action_type'] == 4)[0])
    feature['action5_count'] = len(np.where(row_2['action_type'] == 5)[0])
    return feature


def day_count_func(row_1):
    feature = {}
    day_count_b = row_1.groupby('day', sort=True).apply(day_count)
    feature['page0_mean'] = np.mean(day_count_b['page0_count'])
    feature['page0_max'] = np.max(day_count_b['page0_count'])
    feature['page0_min'] = np.min(day_count_b['page0_count'])
    feature['page0_kur'] = stats.kurtosis(day_count_b['page0_count'])
    feature['page0_ske'] = stats.skew(day_count_b['page0_count'])
    feature['page0_last'] = np.array(day_count_b['page0_count'])[-1]
    feature['page0_std'] = np.std(day_count_b['page0_count'])

    feature['page1_mean'] = np.mean(day_count_b['page1_count'])
    feature['page1_max'] = np.max(day_count_b['page1_count'])
    feature['page1_min'] = np.min(day_count_b['page1_count'])
    feature['page1_kur'] = stats.kurtosis(day_count_b['page1_count'])
    feature['page1_ske'] = stats.skew(day_count_b['page1_count'])
    feature['page1_last'] = np.array(day_count_b['page1_count'])[-1]
    feature['page1_std'] = np.std(day_count_b['page1_count'])

    feature['page2_mean'] = np.mean(day_count_b['page2_count'])
    feature['page2_max'] = np.max(day_count_b['page2_count'])
    feature['page2_min'] = np.min(day_count_b['page2_count'])
    feature['page2_kur'] = stats.kurtosis(day_count_b['page2_count'])
    feature['page2_ske'] = stats.skew(day_count_b['page2_count'])
    feature['page2_last'] = np.array(day_count_b['page2_count'])[-1]
    feature['page2_std'] = np.std(day_count_b['page2_count'])

    feature['page3_mean'] = np.mean(day_count_b['page3_count'])
    feature['page3_max'] = np.max(day_count_b['page3_count'])
    feature['page3_min'] = np.min(day_count_b['page3_count'])
    feature['page3_kur'] = stats.kurtosis(day_count_b['page3_count'])
    feature['page3_ske'] = stats.skew(day_count_b['page3_count'])
    feature['page3_last'] = np.array(day_count_b['page3_count'])[-1]
    feature['page3_std'] = np.std(day_count_b['page3_count'])

    feature['page4_mean'] = np.mean(day_count_b['page4_count'])
    feature['page4_max'] = np.max(day_count_b['page4_count'])
    feature['page4_min'] = np.min(day_count_b['page4_count'])
    feature['page4_kur'] = stats.kurtosis(day_count_b['page4_count'])
    feature['page4_ske'] = stats.skew(day_count_b['page4_count'])
    feature['page4_last'] = np.array(day_count_b['page4_count'])[-1]
    feature['page4_std'] = np.std(day_count_b['page4_count'])

    feature['action0_mean'] = np.mean(day_count_b['action0_count'])
    feature['action0_max'] = np.max(day_count_b['action0_count'])
    feature['action0_min'] = np.min(day_count_b['action0_count'])
    feature['action0_kur'] = stats.kurtosis(day_count_b['action0_count'])
    feature['action0_ske'] = stats.skew(day_count_b['action0_count'])
    feature['action0_last'] = np.array(day_count_b['action0_count'])[-1]
    feature['action0_std'] = np.std(day_count_b['action0_count'])

    feature['action1_mean'] = np.mean(day_count_b['action1_count'])
    feature['action1_max'] = np.max(day_count_b['action1_count'])
    feature['action1_min'] = np.min(day_count_b['action1_count'])
    feature['action1_kur'] = stats.kurtosis(day_count_b['action1_count'])
    feature['action1_ske'] = stats.skew(day_count_b['action1_count'])
    feature['action1_last'] = np.array(day_count_b['action1_count'])[-1]
    feature['action1_std'] = np.std(day_count_b['action1_count'])

    feature['action2_mean'] = np.mean(day_count_b['action2_count'])
    feature['action2_max'] = np.max(day_count_b['action2_count'])
    feature['action2_min'] = np.min(day_count_b['action2_count'])
    feature['action2_kur'] = stats.kurtosis(day_count_b['action2_count'])
    feature['action2_ske'] = stats.skew(day_count_b['action2_count'])
    feature['action2_last'] = np.array(day_count_b['action2_count'])[-1]
    feature['action2_std'] = np.std(day_count_b['action2_count'])

    feature['action3_mean'] = np.mean(day_count_b['action3_count'])
    feature['action3_max'] = np.max(day_count_b['action3_count'])
    feature['action3_min'] = np.min(day_count_b['action3_count'])
    feature['action3_kur'] = stats.kurtosis(day_count_b['action3_count'])
    feature['action3_ske'] = stats.skew(day_count_b['action3_count'])
    feature['action3_last'] = np.array(day_count_b['action3_count'])[-1]
    feature['action3_std'] = np.std(day_count_b['action3_count'])

    # feature['action4_mean'] = np.mean(day_count_b['action4_count'])
    # feature['action4_max'] = np.max(day_count_b['action4_count'])
    # feature['action4_min'] = np.min(day_count_b['action4_count'])
    # feature['action4_kur'] = stats.kurtosis(day_count_b['action4_count'])
    # feature['action4_ske'] = stats.skew(day_count_b['action4_count'])
    # feature['action4_last'] = np.array(day_count_b['action4_count'])[-1]
    # feature['action4_std'] = np.std(day_count_b['action4_count'])
    #
    # feature['action5_mean'] = np.mean(day_count_b['action5_count'])
    # feature['action5_max'] = np.max(day_count_b['action5_count'])
    # feature['action5_min'] = np.min(day_count_b['action5_count'])
    # feature['action5_kur'] = stats.kurtosis(day_count_b['action5_count'])
    # feature['action5_ske'] = stats.skew(day_count_b['action5_count'])
    # feature['action5_last'] = np.array(day_count_b['action5_count'])[-1]
    # feature['action5_std'] = np.std(day_count_b['action5_count'])
    return feature


def get_train_label(train_path, test_path):
    train_reg = pd.read_csv(train_path + register, usecols=['user_id'])
    train_cre = pd.read_csv(train_path + create, usecols=['user_id'])
    train_lau = pd.read_csv(train_path + launch, usecols=['user_id'])
    train_act = pd.read_csv(train_path + activity, usecols=['user_id'])
    train_data_id = np.unique(pd.concat([train_reg, train_cre, train_lau, train_act]))

    test_reg = pd.read_csv(test_path + register, usecols=['user_id'])
    test_cre = pd.read_csv(test_path + create, usecols=['user_id'])
    test_lau = pd.read_csv(test_path + launch, usecols=['user_id'])
    test_act = pd.read_csv(test_path + activity, usecols=['user_id'])
    # unique（）保留数组中不同的值，返回两个参数。
    test_data_id = np.unique(pd.concat([test_reg, test_cre, test_lau, test_act]))
    train_label = []
    for i in train_data_id:
        if i in test_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = train_data_id
    train_data['label'] = train_label
    return train_data


def get_test(test_path):
    test_reg = pd.read_csv(test_path + register, usecols=['user_id'])
    test_cre = pd.read_csv(test_path + create, usecols=['user_id'])
    test_lau = pd.read_csv(test_path + launch, usecols=['user_id'])
    test_act = pd.read_csv(test_path + activity, usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_reg, test_cre, test_lau, test_act]))
    test_data = pd.DataFrame()
    test_data['user_id'] = test_data_id
    return test_data


def get_create_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    # feature['create_count'] = len(row)
    diff_day = np.diff(row['day'])
    if len(diff_day) != 0:
        # feature['create_day_diff_mean'] = np.mean(diff_day)
        # feature['create_day_diff_std'] = np.std(diff_day)
        # feature['create_day_diff_min'] = np.min(diff_day)
        # feature['create_day_diff_mode'] = stats.mode(interval_data)[0][0]
        feature['create_day_diff_ske'] = stats.skew(diff_day)
        feature['create_day_diff_kur'] = stats.kurtosis(diff_day)
        # feature['create_day_diff_max'] = np.max(diff_day)
        feature['create_day_last'] = diff_day[-1]
        feature['create_sub_register'] = np.subtract(np.max(row['max_day']), np.max(row['day']))
        feature['create_mode'] = stats.mode(row['day'])[0][0]
        return feature


def get_register_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['device_type'] = list(row['device type'])[0]
    feature['register_day'] = list(row['register_day'])[0]
    feature['register_type'] = list(row['register_type'])[0]
    # feature['register_day_cut_max_day'] = day_cut_max_day(row['register_day'])
    return feature


def get_launch_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    # feature['launch_count'] = len(row)
    diff_day = np.diff(row['day'])
    if len(diff_day) != 0:
        feature['launch_day_diff_mean'] = np.mean(diff_day)
        feature['launch_day_diff_std'] = np.std(diff_day)
        feature['launch_day_diff_max'] = np.max(diff_day)
        feature['launch_day_diff_min'] = np.min(diff_day)
        feature['launch_day_diff_kur'] = stats.kurtosis(diff_day)
        feature['launch_day_diff_ske'] = stats.skew(diff_day)
        feature['launch_day_diff_last'] = diff_day[-1]
        # feature['launch_day_cut_max_day'] = day_cut_max_day(row['day'])
        feature['launch_sub_register'] = np.subtract(np.max(row['max_day']), np.max(row['day']))
    else:
        feature['launch_day_diff_mean'] = 0
        feature['launch_day_diff_std'] = 0
        feature['launch_day_diff_max'] = 0
        feature['launch_day_diff_min'] = 0
        feature['launch_day_diff_kur'] = 0
        feature['launch_day_diff_ske'] = 0
        feature['launch_day_diff_last'] = 0
        # feature['launch_day_cut_max_day'] = day_cut_max_day(row['day'])
        feature['launch_sub_register'] = np.subtract(np.max(row['max_day']), np.max(row['day']))

    launch_day_count = np.bincount(row['day'])[np.nonzero(np.bincount(row['day']))[0]]
    feature['launch_day_count_mean'] = np.mean(launch_day_count)
    feature['launch_day_count_max'] = np.max(launch_day_count)
    feature['launch_day_count_std'] = np.std(launch_day_count)
    return feature


def get_activity_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['activity_count'] = len(row)
    diff_day = np.diff(np.unique(row['day']))
    if len(diff_day) != 0:
        # feature['activity_day_diff_mean'] = np.mean(diff_day)
        # feature['activity_day_diff_std'] = np.std(diff_day)
        # feature['activity_day_diff_max'] = np.max(diff_day)
        # feature['activity_day_diff_min'] = np.min(diff_day)
        feature['activity_day_diff_kur'] = stats.kurtosis(diff_day)
        feature['activity_day_diff_ske'] = stats.skew(diff_day)
        feature['activity_day_diff_last'] = diff_day[-1]
    else:
        # feature['activity_day_diff_mean'] = 0
        # feature['activity_day_diff_std'] = 0
        # feature['activity_day_diff_max'] = 0
        # feature['activity_day_diff_min'] = 0
        feature['activity_day_diff_kur'] = 0
        feature['activity_day_diff_ske'] = 0
        feature['activity_day_diff_last'] = 0
    feature['0_page_count'] = len(np.where(row['page'] == 0)[0])
    feature['1_page_count'] = len(np.where(row['page'] == 1)[0])
    feature['2_page_count'] = len(np.where(row['page'] == 2)[0])
    feature['3_page_count'] = len(np.where(row['page'] == 3)[0])
    # feature['4_page_count'] = len(np.where(row['page'] == 4)[0])
    feature['0_page_count_div_sum'] = len(np.where(row['page'] == 0)[0]) / len(row)
    feature['1_page_count_div_sum'] = len(np.where(row['page'] == 1)[0]) / len(row)
    feature['2_page_count_div_sum'] = len(np.where(row['page'] == 2)[0]) / len(row)
    feature['3_page_count_div_sum'] = len(np.where(row['page'] == 3)[0]) / len(row)
    feature['4_page_count_div_sum'] = len(np.where(row['page'] == 4)[0]) / len(row)
    feature['0_action_count'] = len(np.where(row['action_type'] == 0)[0])
    # feature['1_action_count'] = len(np.where(row['action_type'] == 1)[0])
    # feature['2_action_count'] = len(np.where(row['action_type'] == 2)[0])
    # feature['3_action_count'] = len(np.where(row['action_type'] == 3)[0])
    # feature['4_action_count'] = len(np.where(row['action_type'] == 4)[0])
    # feature['5_action_count'] = len(np.where(row['action_type'] == 5)[0])
    # np.where 返回符合条件的索引
    feature['0_action_count_div_sum'] = len(np.where(row['action_type'] == 0)[0]) / len(row)
    feature['1_action_count_div_sum'] = len(np.where(row['action_type'] == 1)[0]) / len(row)
    feature['2_action_count_div_sum'] = len(np.where(row['action_type'] == 2)[0]) / len(row)
    feature['3_action_count_div_sum'] = len(np.where(row['action_type'] == 3)[0]) / len(row)
    # feature['4_action_count_div_sum'] = len(np.where(row['action_type'] == 4)[0]) / len(row)
    # feature['5_action_count_div_sum'] = len(np.where(row['action_type'] == 5)[0]) / len(row)
    feature['video_id_mode'] = stats.mode(row['video_id'])[0][0]
    feature['author_id_mode'] = stats.mode(row['author_id'])[0][0]
    activity_cout = np.bincount(row['day'])[np.nonzero(np.bincount(row['day']))[0]]
    activity_cout_diff = np.diff(activity_cout)
    feature['activity_count_mean'] = np.mean(activity_cout)
    if len(activity_cout_diff) != 0:
        feature['activity_diff_count_max'] = np.max(activity_cout_diff)
        feature['activity_diff_count_mean'] = np.mean(activity_cout_diff)
        feature['activity_diff_count_last'] = activity_cout_diff[-1]
        feature['activity_diff_count_min'] = np.min(activity_cout_diff)
        # feature['activity_diff_count_std'] = np.std(activity_cout_diff)
        feature['activity_diff_count_ske'] = stats.skew(activity_cout_diff)
        feature['activity_diff_count_kur'] = stats.kurtosis(activity_cout_diff)
    else:
        feature['activity_diff_count_max'] = 0
        feature['activity_diff_count_mean'] = 0
        feature['activity_diff_count_last'] = 0
        feature['activity_diff_count_min'] = 0
        # feature['activity_diff_count_std'] = 0
        feature['activity_diff_count_ske'] = 0
        feature['activity_diff_count_kur'] = 0
    feature['activity_count_min'] = np.min(activity_cout)
    feature['activity_count_max'] = np.max(activity_cout)
    feature['activity_count_kur'] = stats.kurtosis(activity_cout)
    feature['activity_count_std'] = np.std(activity_cout)
    feature['activity_count_last'] = np.array(activity_cout)[-1]

    day_statistic = day_count_func(row)
    feature['activity_type1_mean'] = day_statistic['action1_mean']

    feature['activity_page0_last'] = day_statistic['page0_last']
    feature['activity_page0_mean'] = day_statistic['page0_mean']
    feature['activity_page0_std'] = day_statistic['page0_std']
    # feature['activity_page0_max'] = day_statistic['page0_max']
    feature['activity_page0_min'] = day_statistic['page0_min']
    feature['activity_page0_kur'] = day_statistic['page0_kur']
    feature['activity_page0_ske'] = day_statistic['page0_ske']

    feature['activity_page1_last'] = day_statistic['page1_last']
    feature['activity_page1_mean'] = day_statistic['page1_mean']
    feature['activity_page1_std'] = day_statistic['page1_std']
    feature['activity_page1_max'] = day_statistic['page1_max']
    feature['activity_page1_min'] = day_statistic['page1_min']
    feature['activity_page1_kur'] = day_statistic['page1_kur']
    feature['activity_page1_ske'] = day_statistic['page1_ske']

    feature['activity_page2_last'] = day_statistic['page2_last']
    feature['activity_page2_mean'] = day_statistic['page2_mean']
    feature['activity_page2_std'] = day_statistic['page2_std']
    feature['activity_page2_max'] = day_statistic['page2_max']
    # feature['activity_page2_min'] = day_statistic['page2_min']
    feature['activity_page2_kur'] = day_statistic['page2_kur']
    feature['activity_page2_ske'] = day_statistic['page2_ske']

    feature['activity_page3_last'] = day_statistic['page3_last']
    # feature['activity_page3_mean'] = day_statistic['page3_mean']
    feature['activity_page3_std'] = day_statistic['page3_std']
    feature['activity_page3_max'] = day_statistic['page3_max']
    feature['activity_page3_min'] = day_statistic['page3_min']
    feature['activity_page3_kur'] = day_statistic['page3_kur']
    feature['activity_page3_ske'] = day_statistic['page3_ske']

    feature['activity_page4_last'] = day_statistic['page4_last']
    # feature['activity_page4_mean'] = day_statistic['page4_mean']
    feature['activity_page4_std'] = day_statistic['page4_std']
    # feature['activity_page4_max'] = day_statistic['page4_max']
    # feature['activity_page4_min'] = day_statistic['page4_min']
    # feature['activity_page4_kur'] = day_statistic['page4_kur']
    # feature['activity_page4_ske'] = day_statistic['page4_ske']

    feature['activity_type0_last'] = day_statistic['action0_last']
    feature['activity_type0_mean'] = day_statistic['action0_mean']
    feature['activity_type0_std'] = day_statistic['action0_std']
    # feature['activity_type0_max'] = day_statistic['action0_max']
    # feature['activity_type0_min'] = day_statistic['action0_min']
    # feature['activity_type0_kur'] = day_statistic['action0_kur']
    feature['activity_type0_ske'] = day_statistic['action0_ske']

    feature['activity_type1_last'] = day_statistic['action1_last']
    # feature['activity_type1_mean'] = day_statistic['action1_mean']
    feature['activity_type1_std'] = day_statistic['action1_std']
    # feature['activity_type1_max'] = day_statistic['action1_max']
    # feature['activity_type1_min'] = day_statistic['action1_min']
    feature['activity_type1_kur'] = day_statistic['action1_kur']
    # feature['activity_type1_ske'] = day_statistic['action1_ske']

    # feature['activity_type2_last'] = day_statistic['action2_last']
    feature['activity_type2_mean'] = day_statistic['action2_mean']
    feature['activity_type2_std'] = day_statistic['action2_std']
    # feature['activity_type2_max'] = day_statistic['action2_max']
    # feature['activity_type2_min'] = day_statistic['action2_min']
    # feature['activity_type2_kur'] = day_statistic['action2_kur']
    # feature['activity_type2_ske'] = day_statistic['action2_ske']

    feature['activity_type3_last'] = day_statistic['action3_last']
    # feature['activity_type3_mean'] = day_statistic['action3_mean']
    feature['activity_type3_std'] = day_statistic['action3_std']
    # feature['activity_type3_max'] = day_statistic['action3_max']
    # feature['activity_type3_min'] = day_statistic['action3_min']
    # feature['activity_type3_kur'] = day_statistic['action3_kur']
    # feature['activity_type3_ske'] = day_statistic['action3_ske']

    # feature['activity_type4_last'] = day_statistic['action4_last']
    # feature['activity_type4_mean'] = day_statistic['action4_mean']
    # feature['activity_type4_std'] = day_statistic['action4_std']
    # feature['activity_type4_max'] = day_statistic['action4_max']
    # feature['activity_type4_min'] = day_statistic['action4_min']
    # feature['activity_type4_kur'] = day_statistic['action4_kur']
    # feature['activity_type4_ske'] = day_statistic['action4_ske']

    # feature['activity_type5_last'] = day_statistic['action5_last']
    # feature['activity_type5_mean'] = day_statistic['action5_mean']
    # feature['activity_type5_std'] = day_statistic['action5_std']
    # feature['activity_type5_max'] = day_statistic['action5_max']
    # feature['activity_type5_min'] = day_statistic['action5_min']
    # feature['activity_type5_kur'] = day_statistic['action5_kur']
    # feature['activity_type5_ske'] = day_statistic['action5_ske']

    feature['max_activity_day'] = np.argmax(np.bincount(row['day']))
    feature['activity_sub_register'] = np.subtract(np.max(row['max_day']), np.max(row['day']))
    # feature['activity_day_cut_max_day'] = day_cut_max_day(row['day'])
    return feature


def deal_feature(path, user_id):
    reg = pd.read_csv(path + register)
    cre = pd.read_csv(path + create)
    lau = pd.read_csv(path + launch)
    act = pd.read_csv(path + activity)
    feature = pd.DataFrame()
    feature['user_id'] = user_id

    # create表
    cre['max_day'] = np.max(reg['register_day'])
    cre_feature = cre.groupby('user_id', sort=True).apply(get_create_feature)
    feature = pd.merge(feature, pd.DataFrame(cre_feature), on='user_id', how='left')
    print('create提取成功')

    # launch表
    lau['max_day'] = np.max(reg['register_day'])
    lau_feature = lau.groupby('user_id', sort=True).apply(get_launch_feature)
    feature = pd.merge(feature, pd.DataFrame(lau_feature), on='user_id', how='left')
    print('launch提取成功')

    # register表
    reg_feature = reg.groupby('user_id', sort=True).apply(get_register_feature)
    feature = pd.merge(feature, pd.DataFrame(reg_feature), on='user_id', how='left')
    print('register提取成功')

    # activity表
    act['max_day'] = np.max(reg['register_day'])
    act_feature = act.groupby('user_id', sort=True).apply(get_activity_feature)
    feature = pd.merge(feature, pd.DataFrame(act_feature), on='user_id', how='left')
    print('activity提取成功')

    return feature


def get_data_feature():
    one_train_data = get_train_label(one_dataSet_train_path, one_dataSet_test_path)
    one_feature = deal_feature(one_dataSet_train_path, one_train_data['user_id'])
    one_feature['label'] = one_train_data['label']
    one_feature.to_csv(train_path_one, index=False)
    print('第一组提取完毕')

    two_train_data = get_train_label(two_dataSet_train_path, two_dataSet_test_path)
    two_feature = deal_feature(two_dataSet_train_path, two_train_data['user_id'])
    two_feature.to_csv(test_path_two, index=False)
    two_feature['label'] = two_train_data['label']
    two_feature.to_csv(train_path_two, index=False)
    print('第二组提取完毕')

    train_feature = pd.concat([one_feature, two_feature])
    train_feature.to_csv(train_path, index=False)
    print('训练数据存储完毕')

    test_data = get_test(three_dataSet_train_path)
    test_feature = deal_feature(three_dataSet_train_path, test_data['user_id'])
    test_feature.to_csv(test_path, index=False)
    print('测试数据存储完毕')


if __name__ == '__main__':
    get_data_feature()