# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

sns.set_style("darkgrid")
fig_id = 100
n_policy = 4
MI_ONLY = True
Exp_ID = 0
fig_width = 6.5 # 7.5
fig_height = 3.5 # 6
n_records = 9632
Exp_date = ''
if MI_ONLY:
    suffix = '_MI_only'
    Result_folders = ["Results" + Exp_date + "_noprotection_" + str(n_records) + "_mi_only/",
                      "Results" + Exp_date + "_cdc_" + str(n_records) + "_mi_only/",
                      "Results" + Exp_date + "_dynamic_" + str(n_records) + "_mi_only/",
                      "Results" + Exp_date + "_game_" + str(n_records) + "_mi_only/"]
else:
    suffix = ''
    Result_folders = ["Results" + Exp_date + "_noprotection_" + str(n_records) + "_mi/",
                      "Results" + Exp_date + "_cdc_" + str(n_records) + "_mi/",
                      "Results" + Exp_date + "_dynamic_" + str(n_records) + "_mi/",
                      "Results" + Exp_date + "_game_" + str(n_records) + "_mi/"]
if Exp_ID == 0:
    Figure_folder = "Results" + Exp_date + "_Figures/"
else:
    Figure_folder = "Results" + Exp_date + "_Figures/Exp_" + str(Exp_ID) + "/"
print(Figure_folder)
if not os.path.exists(Figure_folder):
    os.mkdir(Figure_folder)
payoff = []
privacy = []
utility = []
privacy_risk = []
total_payoff = []
total_privacy = []
avg_payoff = []
avg_utility = []
avg_privacy_risk = []
avg_privacy = []
least_recent_date = pd.to_datetime("3/11/2020")
time_stamp = []
for i in range(n_policy):

    list_n_records = np.loadtxt(Result_folders[i] + "list_n_records.csv", dtype=int)
    n_days = len(list_n_records)
    dates = (pd.date_range(least_recent_date, freq="D", periods=n_days))
    list_payoff = np.loadtxt(Result_folders[i] + "list_payoff.csv")
    list_utility = np.loadtxt(Result_folders[i] + "list_utility.csv")
    list_privacy = np.loadtxt(Result_folders[i] + "list_privacy.csv")
    list_date = np.datetime_as_string(dates, unit='D')

    list_avg_payoff = np.loadtxt(Result_folders[i] + "list_payoff_result.csv")
    list_avg_utility = np.loadtxt(Result_folders[i] + "list_utility_result.csv")
    list_avg_privacy = np.loadtxt(Result_folders[i] + "list_privacy_result.csv")

    avg_payoff.extend(list_avg_payoff)
    avg_utility.extend(list_avg_utility)
    avg_privacy.extend(list_avg_privacy)
    time_stamp.extend(list_date)

day = []
for i in range(n_policy):
    day.extend(list(range(n_days)))
time_stamp = day

policy = []
policy_name = np.array(['No-protection', 'CDC-based (5-Anonymity)', 'Dynamic (risk<0.01)', 'Game-theoretic'])
for i in range(n_policy):
    label = [policy_name[i] for j in range(n_days)]
    policy.extend(label)

# for each record
payoff = []
privacy = []
utility = []
for ii in range(n_policy):
    list_payoff = np.loadtxt(Result_folders[ii] + "list_payoff.csv")
    list_privacy = np.loadtxt(Result_folders[ii] + "list_privacy.csv")
    list_utility = np.loadtxt(Result_folders[ii] + "list_utility.csv")
    for i in range(n_days):
        for j in range(list_n_records[i]):
            payoff.append(list_payoff[i])
            privacy.append(list_privacy[i])
            utility.append(list_utility[i])


policy_array1 = np.genfromtxt('data/policy/cdc_policy_pk5_real_' + str(n_records) + '_monthly.csv', delimiter=',', skip_header=1).astype(int)  # CDC-based

array_policy = np.genfromtxt(Result_folders[2] + "array_policy.csv", delimiter=",", skip_header=1).astype(int)
policy_array2 = []
counter = 0
for i in range(n_days):
    for j in range(list_n_records[i]):
        counter += 1
        policy_array2.extend(array_policy[i, :])
policy_array2 = np.array([policy_array2]).reshape(-1, 3)
policy_array0 = np.zeros((counter, 3))
matrix_policy = np.vstack((policy_array0, policy_array1, policy_array2))

policy2 = []
for i in range(0, n_policy):
    label = [policy_name[i] for j in range(counter)]
    policy2.extend(label)
colors = {'No-protection': 'tab:red', 'CDC-based (5-Anonymity)': 'tab:orange', 'Dynamic (risk<0.01)': 'tab:blue', 'Game-theoretic': 'tab:purple'}

# plof figures
if fig_id == 0:  # avg payoff plot
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    dataset = pd.DataFrame({'Time (date)': time_stamp, 'Average payoff per record ($)': avg_payoff, 'Data publishing approach': policy})
    sns.lineplot(data=dataset, x='Time (date)', y='Average payoff per record ($)', hue='Data publishing approach', ax=ax, palette=colors)
    plt.xticks(list(range(0, 700, 100)))
    #plt.xticks([0, 100, 200, 300, 400, 500, 600],
    #           ['2020-03-11', '2020-06-19', '2020-09-27', '2021-01-05', '2021-04-15', '2021-07-24', '2021-11-01'])
    fig.show()
    fig.savefig(Figure_folder + 'Average_payoff' + suffix + '.png')

if fig_id == 100:  # avg payoff utility privacy plot
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height * 3))

    plt.subplot(3, 1, 1)
    dataset = pd.DataFrame({'Time (date)': time_stamp, 'Average payoff per record ($)': avg_payoff, 'Data publishing approach': policy})
    sns.lineplot(data=dataset, x='Time (date)', y='Average payoff per record ($)', hue='Data publishing approach', palette=colors)
    #plt.xticks(list(range(0, 700, 100)))
    plt.xticks([0, 91, 183, 274, 364, 456, 548, 639],
               ['3/11/2020', '6/11', '9/11', '12/11', '3/11/2021', '6/11', '9/11', '12/11'])
    if MI_ONLY:
        plt.text(-110, 10.2, str(chr(ord('@') + 1)), size=11, fontfamily='sans-serif', weight='bold')
    else:
        plt.text(-110, 6.5, str(chr(ord('@') + 1)), size=11, fontfamily='sans-serif', weight='bold')
    plt.subplot(3, 1, 3)
    dataset = pd.DataFrame({'Time (date)': time_stamp, 'Average privacy': avg_privacy, 'Data publishing approach': policy})
    sns.lineplot(data=dataset, x='Time (date)', y='Average privacy', hue='Data publishing approach', palette=colors)
    plt.xticks([0, 91, 183, 274, 364, 456, 548, 639],
               ['3/11/2020', '6/11', '9/11', '12/11', '3/11/2021', '6/11', '9/11', '12/11'])
    if MI_ONLY:
        plt.text(-110, 1, str(chr(ord('@') + 3)), size=11, fontfamily='sans-serif', weight='bold')
    else:
        plt.text(-110, 1, str(chr(ord('@') + 3)), size=11, fontfamily='sans-serif', weight='bold')
    plt.subplot(3, 1, 2)
    dataset = pd.DataFrame({'Time (date)': time_stamp, 'Average data utility': avg_utility, 'Data publishing approach': policy})
    sns.lineplot(data=dataset, x='Time (date)', y='Average data utility', hue='Data publishing approach', palette=colors)
    plt.xticks([0, 91, 183, 274, 364, 456, 548, 639],
               ['3/11/2020', '6/11', '9/11', '12/11', '3/11/2021', '6/11', '9/11', '12/11'])
    if MI_ONLY:
        plt.text(-110, 1, str(chr(ord('@') + 2)), size=11, fontfamily='sans-serif', weight='bold')
    else:
        plt.text(-110, 0.75, str(chr(ord('@') + 2)), size=11, fontfamily='sans-serif', weight='bold')
    fig.show()
    fig.savefig(Figure_folder + 'Results' + suffix + '.png')

if fig_id == 1:  # avg privacy plot
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    dataset = pd.DataFrame({'Time (date)': time_stamp, 'Average privacy': avg_privacy, 'Data publishing approach': policy})
    sns.lineplot(data=dataset, x='Time (date)', y='Average privacy', hue='Data publishing approach', ax=ax, palette=colors)
    fig.show()
    fig.savefig(Figure_folder + 'Average_privacy' + suffix + '.png')

if fig_id == 2:  # avg data utility plot
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    dataset = pd.DataFrame({'Time (date)': time_stamp, 'Average data utility': avg_utility, 'Data publishing approach': policy})
    sns.lineplot(data=dataset, x='Time (date)', y='Average data utility', hue='Data publishing approach', ax=ax, palette=colors)
    fig.show()
    fig.savefig(Figure_folder + 'Average_utility' + suffix + '.png')
