import os
import json
import numpy as np
from tabulate import tabulate
from collections import defaultdict


# Datasets and number of folds
DATASETS = [
            'Beijing-Opera',
            'CREMA-D',
            'ESC50-Actions',
            'ESC50',
            'GT-Music-Genre',
            'NS-Instruments',
            'RAVDESS',
            'SESA',
            'TUT2017',
            'UrbanSound8K',
            'VocalSound',
        ]


# methods = ['coop']
methods = ['zeroshot', 'coop', 'cocoop', 'palm']

accuracy_dict_all = defaultdict(list)
f1_score_dict_all = defaultdict(list)

accuracy_all = []
f1_score_all = []

for method in methods:    
    # Folder containing the JSON files
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), method)


    accuracy_dict = json.load(open(os.path.join(results_folder, 'accuracy.json')))
    f1_score_dict = json.load(open(os.path.join(results_folder, 'f1_score.json')))


    for dataset in DATASETS:
        accuracy_dict_all[dataset].extend(accuracy_dict[dataset])
        f1_score_dict_all[dataset].extend(f1_score_dict[dataset])




# average accuracy and F1-score across all datasets
for dataset in DATASETS:
    accuracy_all.append([accuracy for accuracy in accuracy_dict_all[dataset]])
    f1_score_all.append([f1_score for f1_score in f1_score_dict_all[dataset]])

accuracy_all = np.array(accuracy_all)
f1_score_all = np.array(f1_score_all)

avg_accuracy_all = accuracy_all.mean(axis=0)
avg_f1_score_all = f1_score_all.mean(axis=0)



# print latex table
string_acc = ''
string_f1 = ''
for dataset in DATASETS:
    string_acc = string_acc +  f'{dataset} & ' + ' & '.join([f'{accuracy:0.4f}' for accuracy in accuracy_dict_all[dataset]]) + ' \\\\\n'
    string_f1 = string_f1 + f'{dataset} & ' + ' & '.join([f'{f1_score:0.4f}' for f1_score in f1_score_dict_all[dataset]]) + ' \\\\\n'


string_acc = string_acc +  f'\midrule\nAVERAGE & ' + ' & '.join([f'{accuracy:0.4f}' for accuracy in avg_accuracy_all]) + ' \\\\\n'
string_f1 = string_f1 + f'\midrule\nAVERAGE & ' + ' & '.join([f'{f1_score:0.4f}' for f1_score in avg_f1_score_all]) + ' \\\\\n'


top_row = f"DATASETS ↓ & ZERO SHOT & "
for method in methods[1:]:
    for seed in range(3): top_row = top_row + f"{method.upper()}-SEED{seed} & "
    top_row = top_row + f"{method.upper()}-AVG & "
top_row = top_row[:-2] + ' \\\\'

print("\n\n########## ACCURACY (LaTeX Table) ##########")
results_acc = top_row+"\n"+string_acc
print(results_acc)

print('\n\n')
print("\n\n########## F1-SCORE (LaTeX Table) ##########")
results_f1 = top_row+"\n"+string_f1
print(results_f1)



table_acc = []
for i, row in enumerate(results_acc.split("\n")):
    row_list = row.split("&")
    col_list = []
    for j, col in enumerate(row_list):
        if col.endswith("\\\\"): col = col[:-3]
        col = col.strip()
        col_list.append(col)
    if '\\midrule' in col_list or '' in col_list: continue
    table_acc.append(col_list)
print("\n\nAccuracy")
print(tabulate(table_acc, tablefmt="simple"))

print("\n\n")

table_f1 = []
for i, row in enumerate(results_f1.split("\n")):
    row_list = row.split("&")
    col_list = []
    for j, col in enumerate(row_list):
        if col.endswith("\\\\"): col = col[:-3]
        col = col.strip()
        col_list.append(col)
    if '\\midrule' in col_list or '' in col_list: continue
    table_f1.append(col_list)
print("\n\nF1-Score")
print(tabulate(table_f1, tablefmt="simple"))



