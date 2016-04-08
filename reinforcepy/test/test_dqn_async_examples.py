# Code taken from https://github.com/Lasagne/Lasagne/blob/master/lasagne/tests/test_examples.py
from glob import glob
from importlib import import_module
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext
import os
import sys
import pytest
from reinforcepy.handlers import Parameters


EXAMPLES_DIR = join(dirname(dirname(dirname(__file__))), 'examples/ALE/DQN_Async')


def _example_modules():
    paths = glob(join(EXAMPLES_DIR, "*py"))
    return [splitext(basename(path))[0] for path in paths]


@pytest.fixture
def example(request):
    os.chdir(EXAMPLES_DIR)
    sys.path.insert(0, EXAMPLES_DIR)
    request.addfinalizer(lambda: sys.path.remove(EXAMPLES_DIR))

@pytest.mark.slow
@pytest.mark.parametrize("module_name", _example_modules())
@pytest.mark.usefixtures("example")
def test_example(module_name):
    main = getattr(import_module(module_name), 'main')

    exp_parms = {
        "epochs": 0.005,
        "rom": b"D:\\_code\\breakout.bin",
        "learner_count": 4,
        "save_interval": None,
    }
    parameters = Parameters('experiment_parameters', exp_parms)
    main(parameters)
