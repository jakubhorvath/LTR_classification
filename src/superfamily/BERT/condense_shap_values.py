import pickle
import itertools
import numpy as np
import tqdm
import json 
shap_values = pickle.load(open("shap_values_all.b", "rb"))

shap_dict = {}
kmer_count = {}

# Generate all permutations of a 6-mer consisting of A, C, T, and G
permutations = [''.join(p) for p in itertools.product('ACTG', repeat=6)]

# Initialize the dictionaries with keys and values set to 0
for kmer in permutations:
    shap_dict[kmer] = 0
    kmer_count[kmer] = 0



for i in tqdm.tqdm(range(shap_values.shape[0])):
    for val, kmer in zip(shap_values[i].values, shap_values[i].feature_names):
        if kmer == '':
            continue
        shap_dict[kmer] += val
        kmer_count[kmer] += 1



# Convert the shap_dict to a JSON string
shap_dict_json = json.dumps(shap_dict)

# Save the JSON string to a file
with open("shap_dict.json", "w") as file:
    file.write(shap_dict_json)


# Convert the shap_dict to a JSON string
kmer_count_json = json.dumps(kmer_count)

# Save the JSON string to a file
with open("kmer_count.json", "w") as file:
    file.write(kmer_count_json)