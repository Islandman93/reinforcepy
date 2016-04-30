import pickle
from reinforcepy.handlers.events import toJSON
import json
import os


def load_stat_file(file):
    events = []
    count = 0
    with open(file, 'rb') as in_file:
        while 1:
            try:
                a = pickle.load(in_file)
                for ind, event in enumerate(a):
                    if 'grads' in event:
                        del a[ind]['grads']
                    if 'input' in event:
                        del a[ind]['input']
                    if event['type'] == 'Epoch':
                        del a[ind]
                events += toJSON(a)
                count += 1
            except EOFError:
                print('done', file, count)
                break
    return events
# jsons = list()
# for file in os.listdir(os.getcwd()):
#     if os.path.isfile(file) and file[-9:] == 'stats.pkl':
#         with open(file, 'rb') as in_file:
#             count = 0
#             events = list()
#             while 1:
#                 try:
#                     a = pickle.load(in_file)
#                     tmplist = list()
#                     for ind, event in enumerate(a):
#                         if 'grads' in event:
#                             del a[ind]['grads']
#                         if 'input' in event:
#                             del a[ind]['input']
#                         if event['type'] == 'Epoch':
#                             del a[ind]
#                     events += toJSON(a)
#                     count += 1
#                 except EOFError:
#                     jsons.append({"key": file, "values": events})
#                     events = list()
#                     print('done', file, count)
#                     break
#
# with open('json.json', 'w') as out_file:
#     json.dump(jsons[0:5], out_file)