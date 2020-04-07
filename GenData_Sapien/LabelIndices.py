import os
from collections import defaultdict
import json
import pickle as pk

def yup():
    return []

dicc = defaultdict(yup)
os.chdir('dataset')
directories = os.listdir()
for directory in directories:
    os.chdir(directory)
    with open('result.json', 'r') as f:
        info = f.read()
    details = json.loads(info)
    categorie = details[0]['name']
    dicc[categorie].append(directory)
    print(f"Model: {directory} , Cat: {categorie}")
    os.chdir('..')

with open('Category_indices.pkl', 'wb') as f:
    pk.dump(dicc, f)