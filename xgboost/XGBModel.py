import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from pyspark import SparkContext


def load_extra_info(dir_path):
    def map_user_count(line):
        tokens = line.split(',')
        if tokens[0] == 'user_id':
            return []
        return [(tokens[0], 1)]

    def map_user_avg(line):
        tokens = line.split(',')
        if tokens[0] == 'user_id':
            return []
        return [(tokens[0], float(tokens[2]))]

    def map_photo(line):
        data = json.loads(line)
        return data['business_id'], 1

    def business_info(line):
        data = json.loads(line)
        d = {'review_count': data['review_count'],
             'avg_review': data['stars']}
        return data['business_id'], d

    sc = SparkContext('local[*]', 'p3task2_1')
    sc.setLogLevel('ERROR')

    f_yelp_train = os.path.join(dir_path, 'yelp_train.csv')
    f_photo = os.path.join(dir_path, 'photo.json')
    f_business = os.path.join(dir_path, 'business.json')

    yelp_train_rdd = sc.textFile(f_yelp_train).persist()
    user_count_map = yelp_train_rdd.flatMap(map_user_count).reduceByKey(lambda v1, v2: v1 + v2).collectAsMap()
    user_avg_review_map = yelp_train_rdd.flatMap(map_user_avg).reduceByKey(lambda v1, v2: v1 + v2).collectAsMap()
    user_info = {}
    for uid in user_count_map:
        user_info[uid] = {}
        user_info[uid]['review_count'] = user_count_map[uid]
        user_info[uid]['avg_review'] = user_avg_review_map[uid] / user_count_map[uid]

    photo_map = sc.textFile(f_photo).map(map_photo).reduceByKey(lambda v1, v2: v1 + v2).collectAsMap()
    business_info = sc.textFile(f_business).map(business_info).collectAsMap()
    for b_id in business_info:
        business_info[b_id]['photo_count'] = photo_map[b_id] if b_id in photo_map else 0

    sc.stop()
    return user_info, business_info


def add_user_info_cols(df, user_info):
    user_id_list = df['user_id'].values
    count = [user_info[id]['review_count'] for id in user_id_list]
    avg = [user_info[id]['avg_review'] for id in user_id_list]
    df['user_rating_count'] = count
    df['user_avg_rating'] = avg


def add_business_info_cols(df, business_info):
    business_id_list = df['business_id'].values
    review_count = [business_info[b_id]['review_count'] for b_id in business_id_list]
    star = [business_info[b_id]['avg_review'] for b_id in business_id_list]
    photo_count = [business_info[b_id]['photo_count'] for b_id in business_id_list]
    df['business_review_count'] = review_count
    df['business_avg_rating'] = star
    df['business_photo_count'] = photo_count


def load_data(csv_file):
    rating_df = pd.read_csv(csv_file)

    column_names = list(rating_df.columns)
    users = rating_df[column_names[0]].unique()
    businesses = rating_df[column_names[1]].unique()

    return rating_df, users, businesses


def replace_id_with_index(df, id_map, col_name):
    l = df[col_name]
    index = np.array([id_map[idd] for idd in l])
    df[col_name] = index


def train(train_x, train_y, test_x, test_y, eta, depth, n_iter=10000):
    param = {'eta': eta,
             'max_depth': depth,
             'n_estimators': n_iter}
    print(param)

    model = xgb.XGBRegressor(**param)
    model.fit(train_x, train_y,
              eval_set=[(test_x, test_y)],
              eval_metric='rmse',
              early_stopping_rounds=50,
              verbose=True)
    # print(model)

    y_hat = model.predict(train_x)
    test_y_hat = model.predict(test_x)
    test_mse = np.sqrt(np.mean((test_y_hat - test_y) ** 2))
    print(param)
    print('train rmse=', np.sqrt(np.mean((y_hat - train_y) ** 2)))
    print('test rmse=', test_mse)


def para_opt(train_x, train_y):
    para_dict = {'eta': [x / 100 for x in range(2, 31, 2)],
                 'max_depth': [x for x in range(3, 12)],
                 'early_stopping_rounds': [10],
                 # 'n_estimators': [10000]
                 }

    xgb_model = xgb.XGBRegressor()
    best_model = GridSearchCV(xgb_model, param_grid=para_dict, scoring='neg_root_mean_squared_error', verbose=1)
    best_model.fit(train_x, train_y)
    print(best_model.best_estimator_)
    print(best_model.best_params_)
    predit_y_train = best_model.predict(train_x)

    print('train rmse=', np.sqrt(np.mean((predit_y_train - train_y) ** 2)))


def process_df(df, user_map, business_map, user_info, business_info):
    add_user_info_cols(df, user_info)
    add_business_info_cols(df, business_info)
    replace_id_with_index(df, user_map, 'user_id')
    replace_id_with_index(df, business_map, 'business_id')
    return df


def get_xy(df):
    y = df.stars.values
    x = df.drop('stars', axis=1).values
    return x, y


def read_csv(f_in, header=True):
    data = []
    with open(f_in, 'r')as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    if header:
        return data[1:]
    return data


def write_csv(f_out, data, header=None):
    """

    :param f_out:
    :param data: list of list: [[1,b,c],[d,e,f]]
    :param header:
    :return:
    """
    if header is None:
        header = []
    with open(f_out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)


def get_rmse(f_out, f_validate):
    validate_data = read_csv(f_validate)
    validate_dict = {}
    for item in validate_data:
        validate_dict[tuple(item[0:2])] = float(item[2])

    out_data = read_csv(f_out)
    s = 0
    for item in out_data:
        e = float(item[2]) - validate_dict[tuple(item[0:2])]
        s += e * e
    print('RMSE', np.sqrt(s / len(out_data)))


def main():
    # folder_path = str(sys.argv[1])
    # f_test = str(sys.argv[2])
    # f_out = str(sys.argv[3])

    folder_path = 'data'
    f_test = os.path.join('data', 'yelp_val.csv')
    f_out = 'out.csv'

    f_train = os.path.join(folder_path, 'yelp_train.csv')

    # read data from file
    train_df, train_users, train_businesses = load_data(f_train)

    train_users_set = set(train_users)
    train_businesses_set = set(train_businesses)
    user_map = {user_id: i for i, user_id in enumerate(train_users_set)}
    business_map = {bss_id: i for i, bss_id in enumerate(train_businesses_set)}
    user_info, business_info = load_extra_info(folder_path)

    # add extra info to df
    train_df = process_df(train_df, user_map, business_map, user_info, business_info)
    train_x, train_y = get_xy(train_df)
    #
    # t = time.time()
    # para_opt(train_x, train_y)
    # print('time', time.time() - t)

    train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y, test_size=0.2)

    # TRAIN
    param = {'eta': 0.16,
             'max_depth': 5,
             'n_estimators': 10000}

    model = xgb.XGBRegressor(**param)
    model.fit(train_x, train_y,
              eval_set=[(validate_x, validate_y)],
              eval_metric='rmse',
              early_stopping_rounds=20,
              verbose=False)

    # y_hat = model.predict(train_x)
    # validate_y_hat = model.predict(validate_x)
    # print(param)
    # print('train rmse=', np.sqrt(np.mean((y_hat - train_y) ** 2)))
    # print('validate rmse=', np.sqrt(np.mean((validate_y_hat - validate_y) ** 2)))

    # TEST
    print('predict out file')
    test_data = read_csv(f_test)
    predict_list = []
    for i in test_data:
        u_id = i[0]
        b_id = i[1]
        if u_id not in user_map and b_id not in business_map:
            y_hat = 3.71
        elif u_id not in user_map:
            y_hat = business_info[b_id]['avg_review']
        elif b_id not in business_map:
            y_hat = user_info[u_id]['avg_review']
        else:
            a_test = np.array([
                user_map[u_id], business_map[b_id],
                user_info[u_id]['review_count'], user_info[u_id]['avg_review'],
                business_info[b_id]['review_count'],
                business_info[b_id]['avg_review'],
                business_info[b_id]['photo_count']]).reshape((1, -1))
            y_hat = model.predict(a_test)[0]
        predict_list.append([u_id, b_id, y_hat])
    write_csv(f_out, predict_list, ['user_id', 'business_id', 'prediction'])

    print('Cal rmse')
    get_rmse(f_out, f_test)

    print('aloha')


if __name__ == '__main__':
    main()
