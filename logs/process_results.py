import os
import json
import numpy as np



# Function to load JSON data from a file
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def check_seed_existence(results):
    seeds_exist = []
    for seed in SEEDS:
        if f'seed_{seed}' in results.keys(): seeds_exist.append(seed)
    return seeds_exist


# Function to get results for all seeds of a dataset and method   
def get_dataset_results(dataset):
    json_path = f"{os.path.join(results_folder, dataset)}.json"
    
    if os.path.exists(json_path):
        results = load_json(json_path)
        seeds_exist = check_seed_existence(results)
        if len(seeds_exist) != len(SEEDS): raise ValueError(f"Seeds {set(SEEDS)-set(seeds_exist)} not found in {json_path} file. Get results for all seeds first in '{json_path}'.")
    else:
        raise ValueError(f"File {json_path} does not exist. Get results for Dataset='{dataset}'.") 
    
    return results



def get_results():
    results = {}
    for dataset in DATASETS:
        results[dataset] = get_dataset_results(dataset)
    return results
  
if __name__ == "__main__":
    
    # Datasets 
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



    methods = ['zeroshot', 'coop', 'cocoop', 'palm']

    for method in methods:
        # Folder containing the JSON files
        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), method)

        if method == 'zeroshot':
            SEEDS = [0]
        else:
            SEEDS = [0,1,2]

        results = get_results()
        

        accuracy_dict = {}
        f1_score_dict = {}

        for dataset in DATASETS:

            accuracy_sub_list = []
            f1_score_sub_list = []

            for seed in SEEDS:
                accuracy_sub_list.append(results[dataset][f'seed_{seed}']['accuracy'])
                f1_score_sub_list.append(results[dataset][f'seed_{seed}']['f1_score'])
            
            if len(accuracy_sub_list) > 1:
                accuracy_sub_list.append(np.mean(accuracy_sub_list))
                f1_score_sub_list.append(np.mean(f1_score_sub_list))

            accuracy_dict[dataset] = accuracy_sub_list
            f1_score_dict[dataset] = f1_score_sub_list

        
        with open(os.path.join(results_folder,'accuracy.json'), 'w') as f:
            json.dump(accuracy_dict, f, indent=2)
        print(f"Accuracy results saved in {os.path.join(results_folder,'accuracy.json')} file.")


        with open(os.path.join(results_folder,'f1_score.json'), 'w') as f:
            json.dump(f1_score_dict, f, indent=2)
        print(f"F1-score results saved in {os.path.join(results_folder,'f1_score.json')}.")


    print("\n\nResults saved successfully.\n\n")