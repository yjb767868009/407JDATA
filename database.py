import os
import numpy as np
import scipy.io
import csv
import pandas as pd
from tqdm import tqdm
from glob import glob

DATA_ROOT = 'D:\data\jdata'
action_file = os.path.join(DATA_ROOT, 'jdata_action.csv')
month_file = os.path.join(DATA_ROOT, 'month', 'jdata_action_2.csv')
USER_PATH = os.path.join(DATA_ROOT, 'user')
CLEAN_USER_PATH = os.path.join(DATA_ROOT, 'clean_usr')
DAY_PATH = os.path.join(DATA_ROOT, 'day')


# 将行为按月份划分
def month_div():
    # month2_file = os.path.join(DATA_ROOT, 'jdata_action_2.csv')
    # month3_file = os.path.join(DATA_ROOT, 'jdata_action_3.csv')
    month4_file = os.path.join(DATA_ROOT, 'jdata_action_4.csv')

    month4_list = []
    with open(action_file, 'r', encoding='UTF-8') as input:
        readCSV = csv.reader(input)
        f_flag = True
        for row in readCSV:
            if f_flag:
                f_flag = False
                continue
            date = row[2].split('-')
            if date[1] == '04':
                month4_list.append(row)
                print('read month 4')

    out_file = open(month4_file, 'w', newline='')
    csv_writer = csv.writer(out_file)
    for row in month4_list:
        csv_writer.writerow(row)


# 将行为按用户id划分第二版
def user_div2():
    user_name = 600000
    while True:
        user_list = []
        with open(action_file, 'r', encoding='UTF-8') as input:
            readCSV = csv.reader(input)
            f_flag = True
            for row in readCSV:
                if f_flag:
                    f_flag = False
                    continue
                name = row[0]
                if int(name) == user_name:
                    user_list.append(row)

        user_file = os.path.join(USER_PATH, str(user_name) + '.csv')
        out_file = open(user_file, 'w', newline='')
        csv_writer = csv.writer(out_file)
        for row in user_list:
            csv_writer.writerow(row)
        out_file.close()
        user_name += 1
        print("found " + str(user_name))


# 将行为按用户id划分
def user_div():
    USER_PATH = os.path.join(DATA_ROOT, 'user')
    user_map = {}
    with open(action_file, 'r', encoding='UTF-8') as input:
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
def clean_user():
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
def day_div():
    day_map = {}

    with open(month_file, 'r')as fp:
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
        user_file = os.path.join(USER_PATH, '2_' + str(day) + '.csv')
        out_file = open(user_file, 'w', newline='')
        csv_writer = csv.writer(out_file)
        for row in user_list:
            csv_writer.writerow(row)


# 天数购买量统计
def day_buy_sum():
    pass


if __name__ == '__main__':
    day_div()
