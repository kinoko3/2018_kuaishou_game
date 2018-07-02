import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics
import lightgbm as lgb
train_path = 'train.csv'
test_path = 'test.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_feature = train.drop(['user_id','label'], axis=1)
label = train['label']
test_feature = test.drop(['user_id'], axis=1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2,random_state=1017)
print('载入数据......')
lgb_train = lgb.Dataset(train_feature, label)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
print('开始训练......')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc', 'binary_logloss'}
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=65,
                valid_sets=lgb_eval
                )
gbm.save_model('model/lgb_model.txt')
temp = gbm.predict(X_test)
temp[temp >= 0.43] = 1
temp[temp < 0.43] = 0
# print(Y_test)
print('结果：' + str(sklearn.metrics.f1_score(Y_test, temp)))
print('特征重要性：'+ str(list(gbm.feature_importance())))
########################## 保存结果 ############################
pre = gbm.predict(test_feature)
df_result = pd.DataFrame()
pre[pre >= 0.43]=1
pre[pre < 0.43]=0
pre = pre.astype(int)
df_result['user_id'] = test['user_id']
df_result['result'] = pre
df_result.to_csv('result/lgb_result.csv', index=False)
print('为1的个数：' + str(len(np.where(np.array(pre)==1)[0])))
print('为0的个数：' + str(len(np.where(np.array(pre)==0)[0])))
lgb = pd.read_csv('result/lgb_result.csv')
# print(lgb)
res = lgb[lgb['result'] >= 0.43]
print(len(res))

del res['result']
# del finish['result']
res.to_csv('result/dealed_result.txt', index=False, header=False)