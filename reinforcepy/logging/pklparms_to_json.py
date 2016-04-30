import pickle
from reinforcepy.handlers.events import toJSON
import json
import os

jsons = list()
for file in os.listdir(os.getcwd()):
    if os.path.isfile(file) and file[-3:] == 'pkl' and file[0:5] == 'async':
        with open(file, 'rb') as in_file:
            parms = pickle.load(in_file)
            jsons.append({'key': file, 'values': toJSON(parms)})

with open('parms.json', 'w') as out_file:
    json.dump(jsons[0:2], out_file)