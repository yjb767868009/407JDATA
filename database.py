import math
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import scipy.io
import csv
import pandas as pd
from tqdm import tqdm
from glob import glob


class Datebase(object):
    DATA_ROOT = '/home/fish/data/jdata'
    CLEAN_PATH = os.path.join(DATA_ROOT, 'clean')
    CACHE_PATH = os.path.join(DATA_ROOT, 'cache')
    user_path = os.path.join(DATA_ROOT, 'jdata_user.csv')
    product_path = os.path.join(DATA_ROOT, 'jdata_product.csv')
    comment_path = os.path.join(DATA_ROOT, 'jdata_comment.csv')
    shop_path = os.path.join(DATA_ROOT, 'jdata_shop.csv')
    action_path = os.path.join(DATA_ROOT, 'jdata_action.csv')
    comment_date = ["2018-02-01", "2018-02-08", "2018-02-15", "2018-02-22", "2018-03-01", "2018-03-08", "2018-03-15",
                    "2018-03-22", "2018-03-29", "2018-04-05", "2018-04-12", "2018-04-15"]

    # 生成小样本数据，进行测试
    def make_sample(self):
        sample_path = os.path.join(self.DATA_ROOT, 'sample')
        action_sample_path = os.path.join(sample_path, 'jdata_action.csv')
        shop_sample_path = os.path.join(sample_path, 'jdata_shop.csv')
        product_sample_path = os.path.join(sample_path, 'jdata_product.csv')
        user_sample_path = os.path.join(sample_path, 'jdata_user.csv')
        comment_sample_path = os.path.join(sample_path, 'jdata_comment.csv')

        def create_action_sample(user_list):
            action = pd.read_csv(self.action_path)
            action = action[(action['user_id'].isin(user_list))]
            action.to_csv(action_sample_path, index=False)

        def create_product_sample():
            action = pd.read_csv(action_sample_path)
            sku_list = action.sku_id.values
            product = pd.read_csv(self.product_path)
            product = product[product['sku_id'].isin(sku_list)]
            product.to_csv(product_sample_path, index=False)

        def create_shop_sample():
            product = pd.read_csv(product_sample_path)
            shop_list = product.shop_id.values
            shop = pd.read_csv(self.shop_path)
            shop = shop[shop['shop_id'].isin(shop_list)]
            shop.to_csv(shop_sample_path, index=False)

        def create_user_sample(user_list):
            user = pd.read_csv(self.user_path)
            user = user[(user['user_id'].isin(user_list))]
            user.to_csv(user_sample_path, index=False)

        def create_comment_sample():
            action = pd.read_csv(action_sample_path)
            sku_list = action.sku_id.values
            comment = pd.read_csv(self.comment_path)
            comment = comment[comment['sku_id'].isin(sku_list)]
            comment.to_csv(comment_sample_path, index=False)

        user_list = range(1100000, 1200000)
        create_action_sample(user_list)
        create_user_sample(user_list)
        create_product_sample()
        create_shop_sample()
        create_comment_sample()

    # 将评论数量转换成数字
    def convert_comment(self, number):
        if number <= 1:
            return 1
        elif number <= 10:
            return 2
        elif number <= 50:
            return 3
        else:
            return 4

    # 有无差评，1表示有
    def has_bad_comment(self, number):
        if (number > 0):
            return 1
        else:
            return 0

    # 清洗用户
    def clean_user(self):
        clean_action_path = os.path.join(self.CLEAN_PATH, 'jdata_action.csv')
        actions = pd.read_csv(self.action_path)
        # todo 清洗很久不上线的用户
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        actions = actions[actions['action_2'] > 0]

        actions.to_csv(clean_action_path, index=False, index_label=False)

    def get_basic_user_feat(self):
        dump_path = os.path.join(self.CACHE_PATH, 'basic_user.pkl')
        if os.path.exists(dump_path):
            user = pickle.load(open(dump_path, 'rb'))
        else:
            # user = pd.read_csv(self.user_path, encoding='gbk')
            # age_df = pd.get_dummies(user["age"], prefix="age")
            # sex_df = pd.get_dummies(user["sex"], prefix="sex")
            # user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
            # province_df = pd.get_dummies(user["province"], prefix="province")
            # city_df = pd.get_dummies(user["city"], prefix="city")
            # county_df = pd.get_dummies(user["county"], prefix="county")
            # user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df, province_df, city_df, county_df], axis=1)
            # pickle.dump(user, open(dump_path, 'wb'), protocol=4)
            user = pd.read_csv(self.user_path, encoding='gbk')
            del user['user_reg_tm']
            pickle.dump(user, open(dump_path, 'wb'), protocol=4)
        # print(user.head(10))
        return user

    def get_basic_product_feat(self):
        dump_path = os.path.join(self.CACHE_PATH, 'basic_product.pkl')
        if os.path.exists(dump_path):
            product = pickle.load(open(dump_path, 'rb'))
        else:
            product = pd.read_csv(self.product_path)
            # shop_id_df = pd.get_dummies(product["shop_id"], prefix="shop_id")
            # product = pd.concat([product[['sku_id', 'cate', 'brand']], shop_id_df], axis=1)
            del product['market_time']
            pickle.dump(product, open(dump_path, 'wb'), protocol=4)
        # print(product.head(10))
        return product

    def get_actions(self, start_date, end_date):
        """
        :param start_date:
        :param end_date:
        :return: actions: pd.Dataframe
        """
        action_path = 'all_action_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.CACHE_PATH, action_path)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = pd.read_csv(self.action_path)
            actions = actions[(actions.action_time >= start_date) & (actions.action_time < end_date)]
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        return actions

    def get_action_feat(self, start_date, end_date):
        action_path = 'action_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.CACHE_PATH, action_path)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions[['user_id', 'sku_id', 'type']]
            df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
            actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
            actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
            del actions['type']
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        # print(action_path)
        # print(actions.head(10))
        return actions

    def get_accumulate_action_feat(self, start_date, end_date):
        action_path = 'action_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.CACHE_PATH, action_path)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = self.get_actions(start_date, end_date)
            df = pd.get_dummies(actions['type'], prefix='action')
            actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
            # 近期行为按时间衰减
            actions['weights'] = actions['time'].map(
                lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            # actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
            actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
            print(actions.head(10))
            actions['action_1'] = actions['action_1'] * actions['weights']
            actions['action_2'] = actions['action_2'] * actions['weights']
            actions['action_3'] = actions['action_3'] * actions['weights']
            actions['action_4'] = actions['action_4'] * actions['weights']
            actions['action_5'] = actions['action_5'] * actions['weights']
            del actions['model_id']
            del actions['type']
            del actions['time']
            del actions['datetime']
            del actions['weights']
            actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        return actions

    def get_comments_product_feat(self, comment_date_begin, comment_date_end):
        comments = pd.read_csv(self.comment_path)
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        comments_dic = {}
        for index, row in comments.iterrows():
            comment_list = np.array([row['comments'], row['good_comments'], row['bad_comments']])
            if not row['sku_id'] in comments_dic.keys():
                comments_dic[row['sku_id']] = comment_list
            else:
                comments_dic[row['sku_id']] += comment_list
        dict = pd.DataFrame.from_dict(comments_dic, orient='index',
                                      columns=['comments', 'good_comments', 'bad_comments'])
        dict = dict.reset_index().rename(columns={'index': 'sku_id'})
        dict['bad_comments_ratio'] = dict['bad_comments'] / dict['comments']
        dict['comments'] = dict['comments'].map(self.convert_comment)
        dict['has_bad_comments'] = dict['bad_comments'].map(self.has_bad_comment)
        comments_df = pd.get_dummies(dict['comments'], prefix="comments")
        dict = pd.concat([dict, comments_df], axis=1)
        dict = dict[['sku_id', 'has_bad_comments', 'bad_comments_ratio', 'comments_1', 'comments_2', 'comments_3',
                     'comments_4']]
        # print(dict.head(10))
        return dict

    def get_shop_product_feat(self):
        dump_path = os.path.join(self.CACHE_PATH, 'basic_shop.pkl')
        if os.path.exists(dump_path):
            shop = pickle.load(open(dump_path, 'rb'))
        else:
            shop = pd.read_csv(self.shop_path)
            # shop = shop['shop_id', 'fans_num', 'vip_num', 'shop_reg_tm', 'shop_socre']
            del shop['shop_reg_tm']
            del shop['cate']
            pickle.dump(shop, open(dump_path, 'wb'), protocol=4)
        return shop

    def get_accumulate_user_feat(self, start_date, end_date):
        feature = ['user_id', 'user_action_1_ratio', 'user_action_3_ratio', 'user_action_4_ratio',
                   'user_action_5_ratio']
        file_name = 'user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.CACHE_PATH, file_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = self.get_actions(start_date, end_date)
            df = pd.get_dummies(actions['type'], prefix='action')
            actions = pd.concat([actions['user_id'], df], axis=1)
            actions = actions.groupby(['user_id'], as_index=False).sum()
            actions['user_action_1_ratio'] = actions['action_2'] / actions['action_1']
            actions['user_action_3_ratio'] = actions['action_2'] / actions['action_3']
            actions['user_action_4_ratio'] = actions['action_2'] / actions['action_4']
            actions['user_action_5_ratio'] = actions['action_2'] / actions['action_5']
            actions = actions[feature]
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        # print(actions.head(10))
        return actions

    def get_accumulate_product_feat(self, start_date, end_date):
        feature = ['sku_id', 'product_action_1_ratio', 'product_action_3_ratio', 'product_action_4_ratio',
                   'product_action_5_ratio']
        file_name = 'product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.CACHE_PATH, file_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = self.get_actions(start_date, end_date)
            df = pd.get_dummies(actions['type'], prefix='action')
            actions = pd.concat([actions['sku_id'], df], axis=1)
            actions = actions.groupby(['sku_id'], as_index=False).sum()
            actions['product_action_1_ratio'] = actions['action_2'] / actions['action_1']
            actions['product_action_3_ratio'] = actions['action_2'] / actions['action_3']
            actions['product_action_4_ratio'] = actions['action_2'] / actions['action_4']
            actions['product_action_5_ratio'] = actions['action_2'] / actions['action_5']
            actions = actions[feature]
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        # print(actions.head(10))
        return actions

    def get_labels(self, start_date, end_date):
        label_name = 'labels_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.CACHE_PATH, label_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions[actions['type'] == 2]
            actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
            actions['label'] = 1
            actions = actions[['user_id', 'sku_id', 'label']]
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        # actions.to_csv('./labels.csv', index=False, index_label=False)
        return actions

    def make_test_set(self, train_start_date, train_end_date):
        print('start to create test set')
        test_name = 'test_set_%s_%s.pkl' % (train_start_date, train_end_date)
        dump_path = os.path.join(self.CACHE_PATH, test_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            start_days = "2018-02-01"
            user = self.get_basic_user_feat()
            product = self.get_basic_product_feat()
            shop = self.get_shop_product_feat()
            user_acc = self.get_accumulate_user_feat(start_days, train_end_date)
            product_acc = self.get_accumulate_product_feat(start_days, train_end_date)
            comment_acc = self.get_comments_product_feat(train_start_date, train_end_date)
            # labels = self.get_labels(train_start_date, train_end_date)

            # generate 时间窗口
            # actions = get_accumulate_action_feat(train_start_date, train_end_date)
            actions = None
            for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
                start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
                start_days = start_days.strftime('%Y-%m-%d')
                if actions is None:
                    actions = self.get_action_feat(start_days, train_end_date)
                else:
                    actions = pd.merge(actions, self.get_action_feat(start_days, train_end_date), how='left',
                                       on=['user_id', 'sku_id'])

            actions = pd.merge(actions, user, how='left', on='user_id')
            actions = pd.merge(actions, user_acc, how='left', on='user_id')
            product = pd.merge(product, shop, how='left', on='shop_id')
            actions = pd.merge(actions, product, how='left', on='sku_id')
            actions = pd.merge(actions, product_acc, how='left', on='sku_id')
            actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
            # actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
            actions = actions.fillna(0)
        print('end to create test set')
        # actions.to_csv('./test_actions.csv', index=False, index_label=False)

        users = actions[['user_id', 'cate', 'shop_id']].copy()
        del actions['user_id']
        del actions['cate']
        del actions['sku_id']
        del actions['shop_id']
        return users, actions

    def make_train_set(self, train_start_date, train_end_date, test_start_date, test_end_date, days=30):
        print('start to create train set')
        train_name = 'train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
        dump_path = os.path.join(self.CACHE_PATH, train_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            start_days = "2018-02-01"
            user = self.get_basic_user_feat()
            product = self.get_basic_product_feat()
            shop = self.get_shop_product_feat()
            user_acc = self.get_accumulate_user_feat(start_days, train_end_date)
            product_acc = self.get_accumulate_product_feat(start_days, train_end_date)
            comment_acc = self.get_comments_product_feat(train_start_date, train_end_date)
            labels = self.get_labels(test_start_date, test_end_date)

            # generate 时间窗口
            # actions = get_accumulate_action_feat(train_start_date, train_end_date)
            actions = None
            for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
                start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
                start_days = start_days.strftime('%Y-%m-%d')
                if actions is None:
                    actions = self.get_action_feat(start_days, train_end_date)
                else:
                    actions = pd.merge(actions, self.get_action_feat(start_days, train_end_date), how='left',
                                       on=['user_id', 'sku_id'])

            actions = pd.merge(actions, user, how='left', on='user_id')
            actions = pd.merge(actions, user_acc, how='left', on='user_id')
            product = pd.merge(product, shop, how='left', on='shop_id')
            actions = pd.merge(actions, product, how='left', on='sku_id')
            actions = pd.merge(actions, product_acc, how='left', on='sku_id')
            actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
            actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
            actions = actions.fillna(0)

        print('end to create train set')
        # actions.to_csv('./train_actions.csv', index=False, index_label=False)
        users = actions[['user_id', 'cate', 'shop_id']].copy()
        labels = actions['label'].copy()
        del actions['user_id']
        del actions['cate']
        del actions['shop_id']
        del actions['sku_id']
        del actions['label']

        return users, actions, labels

    def report(self, pred, label):
        actions = label
        result = pred

        # 所有用户商品对
        all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
        all_user_item_pair = np.array(all_user_item_pair)
        # 所有购买用户
        all_user_set = actions['user_id'].unique()

        # 所有品类中预测购买的用户
        all_user_test_set = result['user_id'].unique()
        all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
        all_user_test_item_pair = np.array(all_user_test_item_pair)

        # 计算所有用户购买评价指标
        pos, neg = 0, 0
        for user_id in all_user_test_set:
            if user_id in all_user_set:
                pos += 1
            else:
                neg += 1
        all_user_acc = 1.0 * pos / (pos + neg)
        all_user_recall = 1.0 * pos / len(all_user_set)
        print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
        print('所有用户中预测购买用户的召回率 ' + str(all_user_recall))

        pos, neg = 0, 0
        for user_item_pair in all_user_test_item_pair:
            if user_item_pair in all_user_item_pair:
                pos += 1
            else:
                neg += 1
        all_item_acc = 1.0 * pos / (pos + neg)
        all_item_recall = 1.0 * pos / len(all_user_item_pair)
        print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
        print('所有用户中预测购买商品的召回率' + str(all_item_recall))
        F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
        F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
        score = 0.4 * F11 + 0.6 * F12
        print('F11=' + str(F11))
        print('F12=' + str(F12))
        print('score=' + str(score))

    def save_pred(self, pred):
        pred_path = os.path.join(self.DATA_ROOT, 'sub', 'submission.csv')
        pred.to_csv(pred_path, index=False, index_label=False)


if __name__ == '__main__':
    d = Datebase()
    d.get_labels('2018-04-08', '2018-04-15')
