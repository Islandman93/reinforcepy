import pytest
from reinforcepy.handlers import Parameters


@pytest.fixture(scope='module')
def parameters():
    p = Parameters("Name", {'a': 1, 'b': 2})
    return p


def test_required(parameters: Parameters):
    # test to make sure raises attribute error
    with pytest.raises(AttributeError):
        parameters.required(['c'])

    # make sure doesn't raise error
    parameters.required(['a'])


def test_get(parameters: Parameters):
    assert parameters.get('a') == 1

    # test to make sure raises attribute error
    with pytest.raises(AttributeError):
        parameters.get('c')


def test_set(parameters: Parameters):
    parameters.set('d', 5)
    assert parameters.get('d') == 5


def test_has(parameters: Parameters):
    assert parameters.has('a') is True
    assert parameters.has('c') is False


def test_fromJSON():
    test_dic = {'a': {'a1': 2}, 'network_parameters': {'input_shape': ["None", 1]}}
