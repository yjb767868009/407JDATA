import os
import pandas as pd

from sklearn.model_selection import train_test_split
import xgboost as xgb
from database import Datebase
from matplotlib import pyplot as plt
import operator


def xgboost_make_submission():
    data = Datebase()

    train_start_date = '2018-03-09'
    train_end_date = '2018-04-09'
    test_start_date = '2018-04-09'
    test_end_date = '2018-04-16'

    sub_start_date = '2018-03-15'
    sub_end_date = '2018-04-16'

    user_index, training_data, label = data.make_train_set(train_start_date, train_end_date, test_start_date,
                                                           test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.05,
             'n_estimators': 1000,
             'max_depth': 3,
             'min_child_weight': 5,
             'gamma': 0,
             'subsample': 1.0,
             'colsample_bytree': 0.6,
             'scale_pos_weight': 1,
             'eta': 0.3,
             'silent': 1,
             'objective': 'binary:logistic'
             }

    num_round = 528
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = list(param.items())
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_data = data.make_test_set(sub_start_date, sub_end_date, )
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    # sub_user_index.to_csv('./sub_user_index.csv', index=False, index_label=False)
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    pred = pred[['user_id', 'cate', 'shop_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred['cate'] = pred['cate'].astype(int)
    pred['shop_id'] = pred['shop_id'].astype(int)
    data.save_pred(pred)

    outfile = open('xgb.fmap', 'w')
    features = training_data.columns.values.tolist()
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv('test.csv', index=False, index_label=False)


if __name__ == '__main__':
    xgboost_make_submission()
