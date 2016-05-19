import os
import pickle
import numpy as np

# os.chdir('..\\novelty\\saves')
os.chdir('saves')

# rename to parms
filename = 'async_network_parameters'
lenfile = len(filename)
for file in os.listdir(os.getcwd()):
    if os.path.isfile(file) and file[0] == 'a':
        newname = file[lenfile:]
        os.rename(file, 'parms_'+newname)

# create checkpoints
checkpoints = []
for file in os.listdir(os.getcwd()):
    if os.path.isfile(file) and file[0:3] == 'par':
        checkpoint = {}
        with open(file, 'rb') as in_file:
            parms = pickle.load(in_file)

        checkpoint['epoch'] = round(int(file.split('_')[1].split('.')[0])/4000000, 2)
        networkshape = [{'type': 'w', 'index': 1, 'description': 'Conv Relu', 'shape': parms[0].shape},
                        {'type': 'b', 'index': 1, 'description': 'Untied', 'shape': parms[1].shape},
                        {'type': 'w', 'index': 2, 'description': 'Conv Relu', 'shape': parms[2].shape},
                        {'type': 'b', 'index': 2, 'description': 'Untied', 'shape': parms[3].shape},
                        {'type': 'w', 'index': 3, 'description': 'Dense Relu', 'shape': parms[4].shape},
                        {'type': 'b', 'index': 3, 'description': '', 'shape': parms[5].shape},
                        {'type': 'w', 'index': 4, 'description': 'Dense Linear', 'shape': parms[6].shape},
                        {'type': 'b', 'index': 4, 'description': '', 'shape': parms[7].shape}]
        checkpoint['networkshape'] = networkshape
        stats = [{'name': 'l2norm', 'type': 'w', 'index': 1, 'value': np.sqrt(np.mean(parms[0] ** 2))},
                 {'name': 'l2norm', 'type': 'b', 'index': 1, 'value': np.sqrt(np.mean(parms[1] ** 2))},
                 {'name': 'l2norm', 'type': 'w', 'index': 2, 'value': np.sqrt(np.mean(parms[2] ** 2))},
                 {'name': 'l2norm', 'type': 'b', 'index': 2, 'value': np.sqrt(np.mean(parms[3] ** 2))},
                 {'name': 'l2norm', 'type': 'w', 'index': 3, 'value': np.sqrt(np.mean(parms[4] ** 2))},
                 {'name': 'l2norm', 'type': 'b', 'index': 3, 'value': np.sqrt(np.mean(parms[5] ** 2))},
                 {'name': 'l2norm', 'type': 'w', 'index': 4, 'value': np.sqrt(np.mean(parms[6] ** 2))},
                 {'name': 'l2norm', 'type': 'b', 'index': 4, 'value': np.sqrt(np.mean(parms[7] ** 2))}]

        checkpoint['stats'] = stats
        checkpoint['weightFile'] = file

        checkpoints.append(checkpoint)

with open('checkpoints.pkl', 'wb') as out_file:
    pickle.dump(checkpoints, out_file)



