import pickle
from reinforcepy.logging.toJSON import toJSON


def load_checkpoint_file():
    with open('checkpoints.pkl', 'rb') as in_file:
        parms = pickle.load(in_file)
    return toJSON(parms)
