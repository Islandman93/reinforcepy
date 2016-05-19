import pickle
from reinforcepy.logging.toJSON import toJSON
import os


def load_stats():
    stat_files = list()
    for file in os.listdir(os.getcwd()):
        if os.path.isfile(file) and file[-9:] == 'stats.pkl':
            stat_files.append(file)
    return stat_files


def load_stat_file(file):
    events = []
    count = 0
    with open(file, 'rb') as in_file:
        while 1:
            try:
                a = pickle.load(in_file)
                events += toJSON(a)
                count += 1
            except EOFError:
                print('done', file, count)
                break
    return events