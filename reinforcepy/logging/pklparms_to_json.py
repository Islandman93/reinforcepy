import pickle
from reinforcepy.logging.toJSON import toJSON
import os


def load_parms():
    parm_files = list()
    for file in os.listdir(os.getcwd()):
        if os.path.isfile(file) and file[-3:] == 'pkl' and file[0:5] == 'parms':
            parm_files.append(file)

    def sorter(value):
        number_pkl = value.split('_')[1]
        number = number_pkl.split('.')[0]
        return int(number)
    return sorted(parm_files, key=sorter)


def load_parm_file(file):
    with open(file, 'rb') as in_file:
        parms = pickle.load(in_file)
    return toJSON(parms)
