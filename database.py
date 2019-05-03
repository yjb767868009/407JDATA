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


def convert_num(num_str):
    num = int(num_str)


class Datebase(object):
    DATA_ROOT = '/home/fish/data/jdata/sample'
    cache_path = os.path.join(DATA_ROOT, 'cache')
    user_path = os.path.join(DATA_ROOT, 'jdata_user.csv')
    product_path = os.path.join(DATA_ROOT, 'jdata_product.csv')
    comment_path = os.path.join(DATA_ROOT, 'jdata_comment.csv')
    shop_path = os.path.join(DATA_ROOT, 'jdata_shop.csv')
    action_file = os.path.join(DATA_ROOT, 'jdata_action.csv')
    month_file = os.path.join(DATA_ROOT, 'month', 'jdata_action_2.csv')
    USER_PATH = os.path.join(DATA_ROOT, 'user')
    CLEAN_USER_PATH = os.path.join(DATA_ROOT, 'clean_usr')
    DAY_PATH = os.path.join(DATA_ROOT, 'day')
    action_1_path = os.path.join(DATA_ROOT, 'month', 'jdata_action_2.csv')
    action_2_path = os.path.join(DATA_ROOT, 'month', 'jdata_action_3.csv')
    action_3_path = os.path.join(DATA_ROOT, 'month', 'jdata_action_4.csv')
    action_path = os.path.join(DATA_ROOT, 'jdata_action.csv')
    comment_date = ["2018-02-01", "2018-02-08", "2018-02-15", "2018-02-22", "2018-02-29", "2018-03-07", "2018-03-14",
                    "2018-03-21", "2018-03-28", "2018-04-04", "2018-04-11", "2018-04-15"]

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

        user_list = range(1184330, 1184340)
        create_action_sample(user_list)
        create_user_sample(user_list)
        create_product_sample()
        create_shop_sample()
        create_comment_sample()

    # 将行为按月份划分
    def month_div(self):
        month_list = []
        with open(self.action_file, 'r', encoding='UTF-8') as input:
            readCSV = csv.reader(input)
            f_flag = True
            for row in readCSV:
                if f_flag:
                    f_flag = False
                    continue
                date = row[2].split('-')
                if date[1] == '04':
                    month_list.append(row)
                    print('read month 4')

        out_file = open(self.month_file, 'w', newline='')
        csv_writer = csv.writer(out_file)
        for row in month_list:
            csv_writer.writerow(row)

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

    # 将行为按用户id划分第二版
    def user_div2(self):
        user_name = 600000
        while True:
            user_list = []
            with open(self.action_file, 'r', encoding='UTF-8') as input:
                readCSV = csv.reader(input)
                f_flag = True
                for row in readCSV:
                    if f_flag:
                        f_flag = False
                        continue
                    name = row[0]
                    if int(name) == user_name:
                        user_list.append(row)

            user_file = os.path.join(self.USER_PATH, str(user_name) + '.csv')
            out_file = open(user_file, 'w', newline='')
            csv_writer = csv.writer(out_file)
            for row in user_list:
                csv_writer.writerow(row)
            out_file.close()
            user_name += 1
            print("found " + str(user_name))

    # 将行为按用户id划分
    def user_div(self):
        USER_PATH = os.path.join(self.DATA_ROOT, 'user')
        user_map = {}
        with open(self.action_file, 'r', encoding='UTF-8') as input:
            readCSV = csv.reader(input)
            f_flag = True
            for row in readCSV:
                if f_flag:
                    f_flag = False
                    continue
                name = row[0]
                if 1499999 < int(name) < 2000000:
                    if name not in user_map:
                        user_map[name] = [row]
                    else:
                        user_map[name].append(row)

        for user_name in user_map:
            user_list = user_map[user_name]
            user_file = os.path.join(USER_PATH, user_name + '.csv')
            out_file = open(user_file, 'w', newline='')
            csv_writer = csv.writer(out_file)
            for row in user_list:
                csv_writer.writerow(row)

    # 清洗用户
    def clean_user(self):
        # ser_files = sorted(glob(os.path.join(USER_PATH, '*.csv')))
        user_files = ["D://data//jdata//user//1370817.csv"]
        for i, user_file in enumerate(tqdm(user_files)):
            with open(user_file, 'r') as fp:
                readCSV = csv.reader(fp)
                only_see_not_buy = True
                for row in readCSV:
                    print(row)
                    action = row[4]
                    if only_see_not_buy and int(action) == 2:
                        only_see_not_buy = False

    # 按天数划分
    def day_div(self):
        day_map = {}

        with open(self.month_file, 'r')as fp:
            readCSV = csv.reader(fp)
            f_flag = True
            for row in readCSV:
                if f_flag:
                    f_flag = False
                    continue
                date = row[2].split('-')
                day_name = str(date[2]).split(' ')[0]
                if day_name not in day_map:
                    day_map[day_name] = [row]
                else:
                    day_map[day_name].append(row)

        for day in day_map:
            user_list = day_map[day]
            user_file = os.path.join(self.USER_PATH, '2_' + str(day) + '.csv')
            out_file = open(user_file, 'w', newline='')
            csv_writer = csv.writer(out_file)
            for row in user_list:
                csv_writer.writerow(row)

    # 天数购买量统计
    def day_buy_sum(self):
        pass

    def get_basic_user_feat(self):
        dump_path = os.path.join(self.cache_path, 'basic_user.pkl')
        if os.path.exists(dump_path):
            user = pickle.load(open(dump_path, 'rb'))
        else:
            user = pd.read_csv(self.user_path, encoding='gbk')
            age_df = pd.get_dummies(user["age"], prefix="age")
            sex_df = pd.get_dummies(user["sex"], prefix="sex")
            user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
            province_df = pd.get_dummies(user["province"], prefix="province")
            city_df = pd.get_dummies(user["city"], prefix="city")
            county_df = pd.get_dummies(user["county"], prefix="county")
            user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df, province_df, city_df, county_df], axis=1)
            pickle.dump(user, open(dump_path, 'wb'), protocol=4)
        return user

    def get_basic_product_feat(self):
        dump_path = os.path.join(self.cache_path, 'basic_product.pkl')
        if os.path.exists(dump_path):
            product = pickle.load(open(dump_path, 'rb'))
        else:
            product = pd.read_csv(self.product_path)
            shop_id_df = pd.get_dummies(product["shop_id"], prefix="shop_id")
            product = pd.concat([product[['sku_id', 'cate', 'brand']], shop_id_df], axis=1)
            pickle.dump(product, open(dump_path, 'wb'), protocol=4)
        return product

    def get_actions_1(self):
        action = pd.read_csv(self.action_1_path)
        return action

    def get_actions_2(self):
        action2 = pd.read_csv(self.action_2_path)
        return action2

    def get_actions_3(self):
        action3 = pd.read_csv(self.action_3_path)
        return action3

    def get_actions(self, start_date, end_date):
        """
        :param start_date:
        :param end_date:
        :return: actions: pd.Dataframe
        """
        action_path = 'all_action_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.cache_path, action_path)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            # action_1 = self.get_actions_1()
            # action_2 = self.get_actions_2()
            # action_3 = self.get_actions_3()
            # actions = pd.concat([action_1, action_2, action_3])  # type: pd.DataFrame
            actions = pd.read_csv(self.action_path)
            actions = actions[(actions.action_time >= start_date) & (actions.action_time < end_date)]
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        return actions

    def get_action_feat(self, start_date, end_date):
        action_path = 'action_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.cache_path, action_path)
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
        return actions

    def get_accumulate_action_feat(self, start_date, end_date):
        action_path = 'action_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.cache_path, action_path)
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

    # def get_comments_product_feat(self, start_date, end_date):
    #     action_path = 'comments_accumulate_%s_%s.pkl' % (start_date, end_date)
    #     dump_path = os.path.join(self.cache_path, action_path)
    #     if os.path.exists(dump_path):
    #         comments = pickle.load(open(dump_path,'rb'))
    #     else:
    #         comments = pd.read_csv(self.comment_path)
    #         comment_date_end = end_date
    #         comment_date_begin = self.comment_date[0]
    #         for date in reversed(self.comment_date):
    #             if date < comment_date_end:
    #                 comment_date_begin = date
    #                 break
    #         comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
    #         comments['comments'] = comments['comments'].map()
    #         df = pd.get_dummies(comments['comments'], prefix='comments')
    #         comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
    #         # del comments['dt']
    #         # del comments['comment_num']
    #         comments = comments[
    #             ['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3',
    #              'comment_num_4']]
    #         pickle.dump(comments, open(dump_path, 'wb'),protocol=4)
    #     return comments

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
        return dict

    def get_shop_product_feat(self):
        dump_path = os.path.join(self.cache_path, 'shop.pkl')
        if os.path.exists(dump_path):
            shop = pickle.load(open(dump_path, 'rb'))
        else:
            shop = pd.read_csv(self.shop_path)
            shop = shop['shop_id', 'fans_num', 'vip_num', 'shop_reg_tm', 'shop_socre']
            pickle.dump(shop, open(dump_path, 'wb'), protocol=4)
        return shop

    def get_accumulate_user_feat(self, start_date, end_date):
        feature = ['user_id', 'user_action_1_ratio', 'user_action_3_ratio', 'user_action_4_ratio',
                   'user_action_5_ratio']
        file_name = 'user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.cache_path, file_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = self.get_actions(start_date, end_date)
            df = pd.get_dummies(actions['type'], prefix='action')
            actions = pd.concat([actions['user_id'], df], axis=1)
            actions = actions.groupby(['user_id'], as_index=False).sum()
            print(actions)
            actions['user_action_1_ratio'] = actions['action_2'] / actions['action_1']
            actions['user_action_3_ratio'] = actions['action_2'] / actions['action_3']
            actions['user_action_4_ratio'] = actions['action_2'] / actions['action_4']
            actions['user_action_5_ratio'] = actions['action_2'] / actions['action_5']
            actions = actions[feature]
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        return actions

    def get_accumulate_product_feat(self, start_date, end_date):
        feature = ['sku_id', 'product_action_1_ratio', 'product_action_3_ratio', 'product_action_4_ratio',
                   'product_action_5_ratio']
        file_name = 'product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.cache_path, file_name)
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
        return actions

    def get_labels(self, start_date, end_date):
        label_name = 'labels_%s_%s.pkl' % (start_date, end_date)
        dump_path = os.path.join(self.cache_path, label_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            actions = self.get_actions(start_date, end_date)
            actions = actions[actions['type'] == 4]
            actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
            actions['label'] = 1
            actions = actions[['user_id', 'sku_id', 'label']]
            pickle.dump(actions, open(dump_path, 'wb'), protocol=4)
        return actions

    def make_test_set(self, train_start_date, train_end_date):
        test_name = 'test_set_%s_%s.pkl' % (train_start_date, train_end_date)
        dump_path = os.path.join(self.cache_path, test_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            start_days = "2018-02-01"
            user = self.get_basic_user_feat()
            product = self.get_basic_product_feat()
            # shop = self.get_shop_product_feat()
            user_acc = self.get_accumulate_user_feat(start_days, train_end_date)
            product_acc = self.get_accumulate_product_feat(start_days, train_end_date)
            comment_acc = self.get_comments_product_feat(train_start_date, train_end_date)
            # labels = get_labels(test_start_date, test_end_date)

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
            # product = pd.merge(product, shop, how='left', on='shop_id')
            actions = pd.merge(actions, product, how='left', on='sku_id')
            actions = pd.merge(actions, product_acc, how='left', on='sku_id')
            # actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
            # actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
            actions = actions.fillna(0)
            actions = actions[actions['cate'] == 8]

        users = actions[['user_id', 'sku_id']].copy()
        del actions['user_id']
        del actions['sku_id']
        return users, actions

    def make_train_set(self, train_start_date, train_end_date, test_start_date, test_end_date, days=30):
        train_name = 'train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
        dump_path = os.path.join(self.cache_path, train_name)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, 'rb'))
        else:
            start_days = "2018-02-01"
            user = self.get_basic_user_feat()
            product = self.get_basic_product_feat()
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
            actions = pd.merge(actions, product, how='left', on='sku_id')
            actions = pd.merge(actions, product_acc, how='left', on='sku_id')
            # actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
            actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
            actions = actions.fillna(0)

        users = actions[['user_id', 'sku_id']].copy()
        labels = actions['label'].copy()
        del actions['user_id']
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


if __name__ == '__main__':
    d = Datebase()
    d.make_sample()
