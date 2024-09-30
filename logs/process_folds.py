import os
import json
import numpy as np
from collections import defaultdict



# Function to load JSON data from a file
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def check_seed_existence(results):
    seeds_exist = []
    for seed in SEEDS:
        if f'seed_{seed}' in results.keys(): seeds_exist.append(seed)
    return seeds_exist

# Function to check if all seeds exist in results       
def check_seed_results(dataset):
    json_path = f"{os.path.join(results_folder, dataset)}.json"
    if os.path.exists(json_path):
        results = load_json(json_path)
        seeds_exist = check_seed_existence(results)
        if len(seeds_exist) != len(SEEDS):
            raise ValueError(f"Seeds {set(SEEDS)-set(seeds_exist)} not found in {json_path} file. Get results of '{dataset}' on all seeds.")
    else:
        raise ValueError(f"File {json_path} does not exist. Get results of Dataset='{dataset}'.")
    return results


# Function to check if all folds exist for a dataset       
def get_folds_results(dataset, folds):
    results = {}
    for fold in range(1,folds+1):
        json_path = f"{os.path.join(results_folder, dataset+'-FOLD'+str(fold))}.json"
        if os.path.exists(json_path):
            results_fold = load_json(json_path)
            results['FOLD'+str(fold)] = results_fold
            seeds_exist = check_seed_existence(results_fold)
            if len(seeds_exist) != len(SEEDS):
                raise ValueError(f"Seeds {set(SEEDS)-set(seeds_exist)} not found in {json_path} file. Get results '{dataset}-FOLD{fold}' on all seeds.")
        else:
            raise ValueError(f"File {json_path} does not exist. Get results for all folds of Dataset='{dataset}'.")
    return results

# Function to calculate average metrics for all folds of dataset
def average_folds(dataset, folds):
    results = get_folds_results(dataset, folds)
    
    results_avg = {}

    for seed in SEEDS:  
        for fold in range(1,folds+1):
            results_fold = results['FOLD'+str(fold)]
            results_seed = results_fold[f'seed_{seed}']
            if fold == 1:
                results_avg[f'seed_{seed}'] = results_seed
            else:
                for metric in results_seed.keys():
                    if metric != 'epoch':
                        results_avg[f'seed_{seed}'][metric] += results_seed[metric]

    for seed in SEEDS:
        for metric in results_avg[f'seed_{seed}'].keys():
            if metric != 'epoch':
                results_avg[f'seed_{seed}'][metric] /= folds
                results_avg[f'seed_{seed}'][metric] = float(f"{results_avg[f'seed_{seed}'][metric]:0.4f}")
    

    json_path = f'{os.path.join(results_folder,dataset)}.json'
    with open(json_path, 'w') as f:
        json.dump(results_avg, f, indent=2) 


    return results_avg


def get_results():    
    for dataset, folds in DATASETS.items():
        print(f"Calculating average metrics for {dataset} dataset...")
        if folds>1:
            results_avg = average_folds(dataset, folds)
            print(f"Average metrics for {dataset} dataset calculated successfully.")
            print(f"Results saved in {os.path.join(results_folder,dataset)}.json")
            print(results_avg)
            print()
        else:
            print(f"Only one fold exists for {dataset} dataset. No need to calculate average metrics.")
            check_seed_results(dataset)
            print()

    
if __name__ == "__main__":

    # Datasets and number of folds
    DATASETS = {
                'Beijing-Opera':5,
                'CREMA-D':1,
                'ESC50-Actions':5,
                'ESC50':5,
                'GT-Music-Genre':1,
                'NS-Instruments':1,
                'RAVDESS':1,
                'SESA':1,
                'TUT2017':4,
                'UrbanSound8K':10,
                'VocalSound':1
            }
    

    methods = ['zeroshot', 'coop', 'cocoop', 'palm']

    for method in methods:
        # Folder containing the JSON files
        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), method)

        if method == 'zeroshot':
            SEEDS = [0]
        else:
            SEEDS = [0,1,2]
        
        get_results()

        print("All average metrics calculated successfully.")



