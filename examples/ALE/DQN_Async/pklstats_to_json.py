import pickle
from reinforcepy.handlers.events import toJSON
import json
import os

jsons = list()
for file in os.listdir(os.getcwd()):
    if os.path.isfile(file) and file[-9:] == 'stats.pkl':
        with open(file, 'rb') as in_file:
            count = 0
            events = list()
            while 1:
                try:
                    a = pickle.load(in_file)
                    events += toJSON(a)
                    count += 1
                except EOFError:
                    jsons.append({"key": file, "values": events})
                    events = list()
                    print('done', file, count)
                    break

with open('json.json', 'w') as out_file:
    json.dump(jsons, out_file)