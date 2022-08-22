# -*- coding: utf-8 -*-
# 8/2/2021 Add the game component, find the optimal policy (same policy for each day)
# 4/30/2021 Add mutual information
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mutual_info_score
import random
import datetime
import os.path

# Configuration
TIME_STEP = 1  # frequency of the data publication in terms of the number of days
#FIRST_DATE = "3/10/2020"  # first day in the patient dataset
BENEFIT = 10  # benefit for sharing each record # $2,000,000,000/333,000,000(US population)/7.8%(positivity rate from tests)
LOSS = 300  # loss for an attack for each record (326)
COST = 1  # cost to attack each record
MAX_LEVELS = np.array([5, 1, 2])  # maximal generalization level for each QID
N_BUFFER_DAYS = 21
Result_folder = "Results20220817_game_9632_mi"
Previous_Result_folder = "Results20220419_optimized/"
ATTRIBUTE_NAMES = ['year_of_birth', 'cde_gender', 'text_res_zip5']
RAND_NUM = 0
n_completed_days = 0
CONTINUE = False
MI_ONLY = False
if MI_ONLY:
    Result_folder += "_only"
Result_folder += "/"
print(Result_folder)
if not os.path.exists(Result_folder):
    os.mkdir(Result_folder)

# input patient dataset
#original_filename_path = 'data/covid_case_0630_nashville_real_adult_recoded_5541.csv'
original_filename_path = 'data/covid_case_20211220_nashville_adult_recoded_9632.csv'
patient_df = pd.read_csv(original_filename_path)
#patient_df = patient_df.drop(columns=['ETHNICITY_SOURCE_VALUE', 'RACE_VALUE'])
#patient_df = patient_df.loc[patient_df['BIRTH_YEAR'] < 2005]
patient_df['TEST_RESULT_TIME'] = pd.to_datetime(patient_df['TEST_RESULT_TIME'])
least_recent_date = patient_df['TEST_RESULT_TIME'].min()  # "3/10/2020"
most_recent_date = patient_df['TEST_RESULT_TIME'].max()  # "6/30/2021"
n_days = (most_recent_date - least_recent_date).days + 1  # 470 (not excluding the last day)

# input reference dataset
voters_df = pd.read_csv('data/voter_list_0714_nashville_real_recoded_337681.csv')
voters_df = voters_df.drop(columns=['cde_race', 'cde_ethnicity'])
voters_df = voters_df.dropna()
voters_df['cde_gender'] = voters_df['cde_gender'].astype(int)

# copy imputed data
imputed_patient_df = patient_df.copy()

if CONTINUE:
    previous_filename_path = Previous_Result_folder + 'imputed_patient_df.csv'
    previous_patient_df = pd.read_csv(previous_filename_path)
    previous_patient_df['TEST_RESULT_TIME'] = pd.to_datetime(previous_patient_df['TEST_RESULT_TIME'])
    imputed_patient_df.loc[previous_patient_df.index, :] = previous_patient_df[:]

# input generalization hierarchy
hierarchy1 = np.genfromtxt('data/hierarchy1.csv', delimiter=',').astype(int)
hierarchy2 = np.genfromtxt('data/hierarchy2.csv', delimiter=',').astype(int)
hierarchy3 = np.genfromtxt('data/hierarchy3.csv', delimiter=',').astype(int)

# build frequency dictionary
dic_freq = {}
for i in range(3):
    attribute_name = ATTRIBUTE_NAMES[i]
    low_val = voters_df[attribute_name].min()
    high_val = voters_df[attribute_name].max()
    for j in range(low_val, high_val+1):
        freq = voters_df[voters_df[attribute_name] == j].shape[0]
        dic_freq[(i, j)] = freq


def marketer_risk_withziptest_adversary(df1, df2):
    # df1: patient dataset under a generalization policy (can be a single record)
    # df2: reference dataset under a generalization policy

    pop = df1.shape[0]
    count = 0
    grouped1_df = df1.groupby(['BIRTH_YEAR', 'GENDER_SOURCE_VALUE', 'ZIP']).size().reset_index(
        name='Equivalence Class Size')
    grouped2_df = df2.groupby(['year_of_birth', 'cde_gender', 'text_res_zip5']).size().reset_index(
        name='Equivalence Class Size')
    grouped2_df.columns = ['BIRTH_YEAR', 'GENDER_SOURCE_VALUE', 'ZIP', 'Equivalence Class Size']

    df = pd.merge(grouped1_df, grouped2_df, how='left', on=['BIRTH_YEAR', 'GENDER_SOURCE_VALUE', 'ZIP'])
    for i in range(0, df.shape[0]):
        if pd.isnull(df.iloc[i, 4]):
            p = 0
        else:
            p = 1 / df.iloc[i, 4]
        if LOSS * p > COST:
            count = count + df.iloc[i, 3] * p

    return count / pop


def PolicyHie_patient(patient_df, x):
    df_copy = patient_df.copy()
    year_level = x[0]
    sex_level = x[1]
    zip_level = x[2]
    if year_level >= 1:
        for i in range(0, 17):
            df_copy.loc[df_copy['BIRTH_YEAR'].between(int(1920 + 5 * i), int(1924 + 5 * i)), 'BIRTH_YEAR'] = i
    if year_level >= 2:
        for i in range(0, 5):
            df_copy.loc[df_copy['BIRTH_YEAR'].between(3 * i, 3 * i + 2), 'BIRTH_YEAR'] = i
        df_copy.loc[df_copy['BIRTH_YEAR'].between(15, 16), 'BIRTH_YEAR'] = 5
    if year_level >= 3:
        for i in range(0, 3):
            df_copy.loc[df_copy['BIRTH_YEAR'].between(2 * i, 2 * i + 1), 'BIRTH_YEAR'] = i
    if year_level >= 4:
        df_copy.loc[df_copy['BIRTH_YEAR'].between(0, 1), 'BIRTH_YEAR'] = 0
        df_copy.loc[df_copy['BIRTH_YEAR'] == 2, 'BIRTH_YEAR'] = 1
    if year_level == 5:
        df_copy['BIRTH_YEAR'] = 0

    if sex_level == 1:
        df_copy['GENDER_SOURCE_VALUE'] = 0

    if zip_level >= 1:
        df_copy['ZIP'] = (df_copy['ZIP'] / 10).astype(int)
    if zip_level >= 2:
        df_copy['ZIP'] = (df_copy['ZIP'] / 10).astype(int)

    return df_copy


def PolicyHie_voter(voters_df, x):
    df_copy = voters_df.copy()
    year_level = x[0]
    sex_level = x[1]
    zip_level = x[2]
    if year_level >= 1:
        for i in range(0, 17):
            df_copy.loc[df_copy['year_of_birth'].between(int(1920 + 5 * i), int(1924 + 5 * i)), 'year_of_birth'] = i
    if year_level >= 2:
        for i in range(0, 5):
            df_copy.loc[df_copy['year_of_birth'].between(3 * i, 3 * i + 2), 'year_of_birth'] = i
        df_copy.loc[df_copy['year_of_birth'].between(15, 16), 'year_of_birth'] = 5
    if year_level >= 3:
        for i in range(0, 3):
            df_copy.loc[df_copy['year_of_birth'].between(2 * i, 2 * i + 1), 'year_of_birth'] = i
    if year_level >= 4:
        df_copy.loc[df_copy['year_of_birth'].between(0, 1), 'year_of_birth'] = 0
        df_copy.loc[df_copy['year_of_birth'] == 2, 'year_of_birth'] = 1
    if year_level == 5:
        df_copy['year_of_birth'] = 0

    if sex_level == 1:
        df_copy['cde_gender'] = 0

    if zip_level >= 1:
        df_copy['text_res_zip5'] = (df_copy['text_res_zip5'] / 10).astype(int)
    if zip_level >= 2:
        df_copy['text_res_zip5'] = (df_copy['text_res_zip5'] / 10).astype(int)

    return df_copy


def information_loss(patient_matrix, policy, max_level, weight):
    IL = 0
    for i in range(len(policy)):
        if policy[i] == 0:
            pass
        elif policy[i] == max_level[i]:
            IL = IL + weight[i]
        else:
            column = patient_matrix[:, i]
            tuple_entry = (tuple(column), i, policy[i])
            if tuple_entry in dic_mi:
                mi = dic_mi[tuple_entry]
            else:
                mi = get_mi(column, i, policy[i])
                dic_mi[tuple_entry] = mi
            IL = IL + weight[i] * (1 - mi)
    return IL


def split_by_date(df, n):
    df_copy = df.copy()
    dates = (pd.date_range(least_recent_date, freq="D", periods=n))
    df_copy = df_copy.loc[df['TEST_RESULT_TIME'].isin(dates)]
    return df_copy

def count_by_date(df, n):
    dates = (pd.date_range(least_recent_date, freq="D", periods=n))
    df_group = df.groupby(['TEST_RESULT_TIME']).size().reset_index(name='Case Class Size')
    df_new = pd.DataFrame()
    df_new['TEST_RESULT_TIME'] = dates
    dfinal = df_new.merge(df_group, on="TEST_RESULT_TIME", how='outer')
    dfinal = dfinal.fillna(0)
    for i in range(0, len(dfinal)):
        dfinal.loc[i, 'TEST_RESULT_TIME'] = i + 1
    y_data = np.array(dfinal.iloc[:, 1])
    return y_data


def predict_time_series(y_data):
    X_data = (np.array(range(len(y_data))) + 1).reshape(-1, 1)
    reg = RandomForestRegressor(random_state=0).fit(X_data, y_data)  # random forest
    X_pred = np.array([[len(X_m)]]).reshape(-1, 1)
    predicted_value = reg.predict(X_pred)
    predicted_value = round(predicted_value[0])
    if predicted_value < 0:
        predicted_value = 0
    return predicted_value


def truncated_rel_error(predicted_value, true_value):
    abs_error = abs(predicted_value - true_value)
    if true_value == 0:
        if abs_error > 0:
            rel_error = 1
        else:
            rel_error = 0
    else:
        rel_error = (1.0 * abs_error) / (1.0 * true_value)
        if rel_error > 1:
            rel_error = 1
    return rel_error


def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy


def get_mi(column, k, level):
    """
    :param column: original data
    :param k: attribute index, [0, 1, 2]
    :param level: generalization level,
    :return: mutual information
    """
    dic_Y = {}
    dic_XY = {}
    for i in range(len(column)):
        value = column[i]
        if k == 0:  # YOB
            hierarchy = hierarchy1
        elif k == 1:  # gender
            hierarchy = hierarchy2
        elif k == 2:  # ZIP
            hierarchy = hierarchy3
        index = value - hierarchy[0, 0]
        if level == 0:
            left_bound = value
        else:
            left_bound = hierarchy[index, level * 2 - 1]
        if left_bound in dic_Y:
            dic_Y[left_bound] += 1
        else:
            dic_Y[left_bound] = 1
        tuple_XY = (value, left_bound)
        if tuple_XY in dic_XY:
            dic_XY[tuple_XY] += 1
        else:
            dic_XY[tuple_XY] = 1
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk_X = hist1 / len(column)
    H_X = -np.sum(pk_X * np.log(pk_X))

    hist2 = np.asarray(list(dic_Y.values()))
    pk_Y = hist2 / sum(hist2)
    H_Y = -np.sum(pk_Y * np.log(pk_Y))

    hist3 = np.asarray(list(dic_XY.values()))
    pk_XY = hist3 / sum(hist3)
    H_XY = -np.sum(pk_XY * np.log(pk_XY))
    mutual_info = H_X + H_Y - H_XY
    """
    lower_bound = min(dic_Y.keys())
    upper_bound = max(dic_Y.keys())
    n_vals = upper_bound - lower_bound + 1
    CT = np.zeros([n_vals, n_vals])
    min_n = n_vals ** 2
    for key in dic_XY:
        x, y = key
        x -= lower_bound
        y -= lower_bound
        CT[x, y] = dic_XY[key]
        if dic_XY[key] < min_n:
            min_n = dic_XY[key]
    CT = CT / min_n
    mi_score = mutual_info_score(column, column, contingency=CT)
    """
    if H_X + H_Y == 0:
        normalized_mutual_info = 1
    else:
        normalized_mutual_info = np.clip(2 * mutual_info / (H_X + H_Y), 0, 1)
    return normalized_mutual_info
"""
a_col = np.asarray([1920, 1920, 1921, 1926, 1930, 1933, 1946, 1965, 1972, 1972, 1973, 1973, 1974, 2004, 1990])
#a_col = np.asarray([1920, 1990])
mi0_0 = get_mi(a_col, 0, 0)
mi0_1 = get_mi(a_col, 0, 1)
mi0_2 = get_mi(a_col, 0, 2)
mi0_3 = get_mi(a_col, 0, 3)
mi0_4 = get_mi(a_col, 0, 4)
mi0_5 = get_mi(a_col, 0, 5)
b_col = np.asarray([0] * 10 + [1] * 5)
mi1_0 = get_mi(b_col, 1, 0)
mi1_1 = get_mi(b_col, 1, 1)
c_col = np.asarray([37201] * 2 + [37214] * 5 + [37221] * 3 + [37238] * 2 + [37241, 37242, 37243])
mi2_0 = get_mi(c_col, 2, 0)
mi2_1 = get_mi(c_col, 2, 1)
mi2_2 = get_mi(c_col, 2, 2)
"""


if __name__ == '__main__':
    start = time.time()
    pid = os.getpid()
    n_t = n_days  # number of days
    list_n_records = np.zeros(n_t)
    list_payoff = np.zeros(n_t)
    list_privacy = np.zeros(n_t)
    list_utility = np.zeros(n_t)
    n_qid = MAX_LEVELS.size
    array_policy = np.zeros([n_t, n_qid])
    counter = 0
    if CONTINUE:
        list_n_records_completed = np.loadtxt(Previous_Result_folder + "list_n_records.csv", dtype=int)
        n_completed_days = len(list_n_records_completed)
        counter = sum(list_n_records_completed)
        print("Continue from a previous run: " + str(n_completed_days) + " days completed.")
        list_n_records[0:n_completed_days] = list_n_records_completed
        list_payoff_completed = np.loadtxt(Previous_Result_folder + "list_payoff_game.csv")
        list_payoff[0:n_completed_days] = list_payoff_completed
        list_privacy_completed = np.loadtxt(Previous_Result_folder + "list_privacy_game.csv")
        list_privacy[0:n_completed_days] = list_privacy_completed
        list_utility_completed = np.loadtxt(Previous_Result_folder + "list_utility_game.csv")
        list_utility[0:n_completed_days] = list_utility_completed
        array_policy_completed = np.genfromtxt(Previous_Result_folder + "array_policy_game.csv", delimiter=",", skip_header=1)
        array_policy[0:n_completed_days, :] = array_policy_completed

    # compute weights
    entropy = []
    for j in range(n_qid):
        entropy.append(get_entropy(np.array(patient_df.iloc[:, j])))
    qid_weight = np.asarray(entropy) / sum(entropy)

    for i in range(n_completed_days, n_t):
        print("[PID:" + str(pid) + " (" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ")] Day: " + str(i))
        dic_predict = {}
        dic_mi = {}
        target_date = pd.to_datetime(least_recent_date) + pd.to_timedelta(np.ceil(i), unit="D")
        dates = (pd.date_range(target_date, freq="D", periods=TIME_STEP))
        patient_df_step = patient_df.loc[patient_df['TEST_RESULT_TIME'].isin(dates)]
        n_records = patient_df_step.shape[0]
        print('Number of records: ' + str(n_records))
        list_n_records[i] = n_records
        if n_records > 0:
            max_payoff = -1
            best_privacy = -1
            best_utility = -1
            best_policy = np.zeros(3).astype(int) - 1
            # predictive utility calculations
            if N_BUFFER_DAYS <= i < n_t - 1 and not MI_ONLY:
                # predict total counts
                list_m_records = []
                list_rel_errors = []
                # get true value for the total count
                next_date = pd.to_datetime(least_recent_date) + pd.to_timedelta(np.ceil(i+1), unit="D")
                dates = (pd.date_range(next_date, freq="D", periods=TIME_STEP))
                patient_df_step_next = patient_df.loc[patient_df['TEST_RESULT_TIME'].isin(dates)]
                true_value = patient_df_step_next.shape[0]
                # predict total count
                imputed_patient_df_existed = split_by_date(imputed_patient_df, i+1)  # time series before tomorrow
                X_m = count_by_date(imputed_patient_df_existed, i + 1)
                tuple_X_m = tuple(X_m)
                if tuple_X_m in dic_predict:
                    predicted_value = dic_predict[tuple_X_m]
                else:
                    predicted_value = predict_time_series(tuple_X_m)
                    dic_predict[tuple_X_m] = predicted_value
                rel_error = truncated_rel_error(predicted_value, true_value)
                list_m_records.append(imputed_patient_df_existed.shape[0])
                list_rel_errors.append(rel_error)

                # year
                list_m_records_year = [[] for i in range(MAX_LEVELS[0] + 1)]
                list_rel_errors_year = [[] for i in range(MAX_LEVELS[0] + 1)]
                for ii in range(0, MAX_LEVELS[0] + 1):
                    policy = np.array([ii, 0, 0])
                    for j in range(n_records):
                        if ii > 0:
                            random.seed(j + RAND_NUM)
                            value = patient_df_step.iloc[j, 0]
                            level = ii
                            hierarchy = hierarchy1
                            index = value - hierarchy[0, 0]
                            left_bound = hierarchy[index, level * 2 - 1]
                            right_bound = hierarchy[index, level * 2]
                            count = []
                            for t in range(left_bound, right_bound + 1):
                                count.append(dic_freq[(0, t)])
                            list_inputed_value = random.choices(list(range(left_bound, right_bound + 1)),
                                                                weights=count, k=1)
                            imputed_value = list_inputed_value[0]
                            imputed_patient_df_existed.iloc[patient_df_step.index[j], 0] = imputed_value

                    for k in range(0, 5):
                        imputed_patient_df_general = PolicyHie_patient(imputed_patient_df_existed, np.array([k, 0, 0]))
                        least_recent_year = imputed_patient_df_general['BIRTH_YEAR'].min()
                        most_recent_year = imputed_patient_df_general['BIRTH_YEAR'].max()

                        for j in range(least_recent_year, most_recent_year + 1):
                            imputed_patient_df_general_one = imputed_patient_df_general.loc[imputed_patient_df_general['BIRTH_YEAR'] == j]
                            list_m_records_year[ii].append(imputed_patient_df_general_one.shape[0])
                            if imputed_patient_df_general_one.shape[0] == 0:  # weight = 0
                                list_rel_errors_year[ii].append(0)
                            else:
                                X_m = count_by_date(imputed_patient_df_general_one, i + 1)
                                tuple_X_m = tuple(X_m)
                                if tuple_X_m in dic_predict:
                                    predicted_value = dic_predict[tuple_X_m]
                                else:
                                    predicted_value = predict_time_series(tuple_X_m)
                                    dic_predict[tuple_X_m] = predicted_value
                                patient_df_step_next_general = PolicyHie_patient(patient_df_step_next, np.array([k, 0, 0]))
                                patient_df_step_next_general_one = patient_df_step_next.loc[patient_df_step_next_general['BIRTH_YEAR'] == j]
                                true_value = patient_df_step_next_general_one.shape[0]
                                rel_error = truncated_rel_error(predicted_value, true_value)
                                list_rel_errors_year[ii].append(rel_error)

                # gender
                list_m_records_gender = [[] for i in range(MAX_LEVELS[1] + 1)]
                list_rel_errors_gender = [[] for i in range(MAX_LEVELS[1] + 1)]
                for jj in range(0, MAX_LEVELS[1] + 1):
                    policy = np.array([0, jj, 0])
                    for j in range(n_records):
                        if jj > 0:
                            random.seed(j + RAND_NUM)
                            value = patient_df_step.iloc[j, 1]
                            level = jj
                            hierarchy = hierarchy2
                            index = value - hierarchy[0, 0]
                            left_bound = hierarchy[index, level * 2 - 1]
                            right_bound = hierarchy[index, level * 2]
                            count = []
                            for t in range(left_bound, right_bound + 1):
                                count.append(dic_freq[(1, t)])
                            list_inputed_value = random.choices(list(range(left_bound, right_bound + 1)),
                                                                weights=count, k=1)
                            imputed_value = list_inputed_value[0]
                            imputed_patient_df_existed.iloc[patient_df_step.index[j], 1] = imputed_value

                    for k in range(0, 1):
                        imputed_patient_df_general = PolicyHie_patient(imputed_patient_df_existed, np.array([0, k, 0]))
                        low_val = imputed_patient_df_general['GENDER_SOURCE_VALUE'].min()
                        high_val = imputed_patient_df_general['GENDER_SOURCE_VALUE'].max()

                        for j in range(low_val, high_val + 1):
                            imputed_patient_df_general_one = imputed_patient_df_general.loc[imputed_patient_df_general['GENDER_SOURCE_VALUE'] == j]
                            list_m_records_gender[jj].append(imputed_patient_df_general_one.shape[0])
                            if imputed_patient_df_general_one.shape[0] == 0:  # weight = 0
                                list_rel_errors_gender[jj].append(0)
                            else:
                                X_m = count_by_date(imputed_patient_df_general_one, i + 1)
                                tuple_X_m = tuple(X_m)
                                if tuple_X_m in dic_predict:
                                    predicted_value = dic_predict[tuple_X_m]
                                else:
                                    predicted_value = predict_time_series(tuple_X_m)
                                    dic_predict[tuple_X_m] = predicted_value
                                patient_df_step_next_general = PolicyHie_patient(patient_df_step_next, np.array([0, k, 0]))
                                patient_df_step_next_general_one = patient_df_step_next.loc[patient_df_step_next_general['GENDER_SOURCE_VALUE'] == j]
                                true_value = patient_df_step_next_general_one.shape[0]
                                rel_error = truncated_rel_error(predicted_value, true_value)
                                list_rel_errors_gender[jj].append(rel_error)

                # zip
                list_m_records_zip = [[] for i in range(MAX_LEVELS[2] + 1)]
                list_rel_errors_zip = [[] for i in range(MAX_LEVELS[2] + 1)]
                for kk in range(0, MAX_LEVELS[2] + 1):
                    policy = np.array([0, 0, kk])
                    for j in range(n_records):
                        if kk > 0:
                            random.seed(j + RAND_NUM)
                            value = patient_df_step.iloc[j, 2]
                            level = kk
                            hierarchy = hierarchy3
                            index = value - hierarchy[0, 0]
                            left_bound = hierarchy[index, level * 2 - 1]
                            right_bound = hierarchy[index, level * 2]
                            count = []
                            for t in range(left_bound, right_bound + 1):
                                count.append(dic_freq[(2, t)])
                            list_inputed_value = random.choices(list(range(left_bound, right_bound + 1)),
                                                                weights=count, k=1)
                            imputed_value = list_inputed_value[0]
                            imputed_patient_df_existed.iloc[patient_df_step.index[j], 2] = imputed_value

                    for k in range(0, 2):
                        imputed_patient_df_general = PolicyHie_patient(imputed_patient_df_existed, np.array([0, 0, k]))
                        low_val = imputed_patient_df_general['ZIP'].min()
                        high_val = imputed_patient_df_general['ZIP'].max()

                        for j in range(low_val, high_val + 1):
                            imputed_patient_df_general_one = imputed_patient_df_general.loc[imputed_patient_df_general['ZIP'] == j]
                            list_m_records_zip[kk].append(imputed_patient_df_general_one.shape[0])
                            if imputed_patient_df_general_one.shape[0] == 0:  # weight = 0
                                list_rel_errors_zip[kk].append(0)
                            else:
                                X_m = count_by_date(imputed_patient_df_general_one, i + 1)
                                tuple_X_m = tuple(X_m)
                                if tuple_X_m in dic_predict:
                                    predicted_value = dic_predict[tuple_X_m]
                                else:
                                    predicted_value = predict_time_series(tuple_X_m)
                                    dic_predict[tuple_X_m] = predicted_value
                                patient_df_step_next_general = PolicyHie_patient(patient_df_step_next, np.array([0, 0, k]))
                                patient_df_step_next_general_one = patient_df_step_next.loc[patient_df_step_next_general['ZIP'] == j]
                                true_value = patient_df_step_next_general_one.shape[0]
                                rel_error = truncated_rel_error(predicted_value, true_value)
                                list_rel_errors_zip[kk].append(rel_error)

            for ii in range(0, MAX_LEVELS[0] + 1):
                for jj in range(0, MAX_LEVELS[1] + 1):
                    for kk in range(0, MAX_LEVELS[2] + 1):
                        policy = np.array([ii, jj, kk])
                        print("Child policy for patient #" + str(counter) + ": " + str(policy))
                        sum_data_quality = n_records * (1 - information_loss(patient_df_step.to_numpy(), policy, MAX_LEVELS, qid_weight))  # data quality for all records on one day
                        sum_privacy_risk = 0
                        for j in range(n_records):
                            patient_df_individual = patient_df_step.iloc[j].to_frame().T  # j-th individual's data record
                            privacy_risk = marketer_risk_withziptest_adversary(PolicyHie_patient(patient_df_individual, policy), PolicyHie_voter(voters_df, policy))
                            sum_privacy_risk = sum_privacy_risk + privacy_risk

                        if N_BUFFER_DAYS <= i < n_t - 1 and not MI_ONLY:
                            list_m_records_total = list_m_records + list_m_records + list_m_records + list_m_records_year[ii] + list_m_records_gender[jj] + list_m_records_zip[kk]
                            list_rel_errors_total = list_rel_errors + list_rel_errors + list_rel_errors + list_rel_errors_year[ii] + list_rel_errors_gender[jj] + list_rel_errors_zip[kk]
                            print(list_m_records_total)
                            print(list_rel_errors_total)

                            array_m_records = np.array(list_m_records_total)
                            array_rel_errors = np.array(list_rel_errors_total)
                            avg_prediction_utility = np.sum(array_m_records * array_rel_errors) / np.sum(array_m_records)
                            sum_prediction_utility = avg_prediction_utility * n_records
                        else:
                            sum_prediction_utility = 0
                        if MI_ONLY:
                            sum_utility = sum_data_quality
                        else:
                            sum_utility = (sum_data_quality + sum_prediction_utility) / 2
                        sum_payoff = BENEFIT * sum_utility - LOSS * sum_privacy_risk
                        child_payoff = sum_payoff / n_records
                        child_utility = sum_utility / n_records
                        child_privacy = 1 - sum_privacy_risk / n_records
                        print("Avg payoff: " + str(child_payoff) + ". Avg utility: " + str(child_utility) + ". Avg privacy: " + str(child_privacy))
                        print("sum_data_quality: " + str(sum_data_quality) + ". sum_prediction_utility: " + str(sum_prediction_utility))
                        if child_payoff > max_payoff:
                            max_payoff = child_payoff
                            best_utility = child_utility
                            best_privacy = child_privacy
                            best_policy = policy
            list_payoff[i] = max_payoff
            list_utility[i] = best_utility
            list_privacy[i] = best_privacy
            array_policy[i, :] = best_policy
            print("Optimal Payoff: " + str(max_payoff)
                  + ". Utility: " + str(best_utility)
                  + ". Privacy: " + str(best_privacy))
            print("Optimal Policy: " + str(best_policy))
            print("---------------------\n")
            # impute the records on this day
            for j in range(n_records):
                counter += 1
                for k in range(3):
                    if best_policy[k] > 0:
                        random.seed(j + RAND_NUM)
                        value = patient_df_step.iloc[j, k]
                        level = best_policy[k]
                        if k == 0:  # YOB
                            hierarchy = hierarchy1
                        elif k == 1:  # gender
                            hierarchy = hierarchy2
                        elif k == 2:  # ZIP
                            hierarchy = hierarchy3
                        index = value - hierarchy[0, 0]
                        left_bound = hierarchy[index, level * 2 - 1]
                        right_bound = hierarchy[index, level * 2]
                        count = []
                        for t in range(left_bound, right_bound + 1):
                            count.append(dic_freq[(k, t)])
                        list_inputed_value = random.choices(list(range(left_bound, right_bound + 1)),
                                                            weights=count, k=1)
                        imputed_value = list_inputed_value[0]
                        imputed_patient_df.iloc[patient_df_step.index[j], k] = imputed_value
    list_payoff_prod_cum = np.cumsum(list_payoff * list_n_records)
    list_privacy_prod_cum = np.cumsum(list_privacy * list_n_records)
    list_utility_prod_cum = np.cumsum(list_utility * list_n_records)
    list_n_records_cum = np.cumsum(list_n_records)
    list_payoff_result = list_payoff_prod_cum/list_n_records_cum
    list_privacy_result = list_privacy_prod_cum/list_n_records_cum
    list_utility_result = list_utility_prod_cum/list_n_records_cum
    end = time.time()
    runtime = end - start
    print("Time used: " + str(runtime) + " seconds.")

    # save results
    np.savetxt(Result_folder + "list_payoff_result_" + str(runtime) + "s.csv", list_payoff_result, delimiter=",")
    np.savetxt(Result_folder + "list_privacy_result.csv", list_privacy_result, delimiter=",")
    np.savetxt(Result_folder + "list_utility_result.csv", list_utility_result, delimiter=",")
    np.savetxt(Result_folder + "list_payoff.csv", list_payoff, delimiter=",")
    np.savetxt(Result_folder + "list_privacy.csv", list_privacy, delimiter=",")
    np.savetxt(Result_folder + "list_utility.csv", list_utility, delimiter=",")
    np.savetxt(Result_folder + "list_n_records.csv", list_n_records, delimiter=",")
    np.savetxt(Result_folder + "array_policy.csv", array_policy, delimiter=",",
               header="BIRTH_YEAR,GENDER_SOURCE_VALUE,ZIP")
    compression_opts = dict(method='zip', archive_name='imputed_patient_df.csv')
    imputed_patient_df.to_csv(Result_folder + "imputed_patient_df.zip", index=False, compression=compression_opts)
    imputed_patient_df.to_pickle(Result_folder + "imputed_patient_df.pkl")
    # plof figures
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('Game Policy')
    ax.set_ylabel('Payoff')
    ax.set_xlabel(r'Time (day)')
    t = np.arange(0, n_t)
    ax.scatter(t, list_payoff_result)
    fig.show()
    fig.savefig(Result_folder + 'Payoff.png')

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('Game Policy')
    ax.set_ylabel('Privacy')
    ax.set_xlabel(r'Time (day)')
    t = np.arange(0, n_t)
    ax.scatter(t, list_privacy_result)
    fig.show()
    fig.savefig(Result_folder + 'Privacy.png')

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('Game Policy')
    ax.set_ylabel('Utility')
    ax.set_xlabel(r'Time (day)')
    t = np.arange(0, n_t)
    ax.scatter(t, list_utility_result)
    fig.show()
    fig.savefig(Result_folder + 'Utility.png')
