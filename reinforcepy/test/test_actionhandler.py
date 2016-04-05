import pytest
from reinforcepy.handlers import ActionHandler, ActionPolicy
import numpy as np


@pytest.fixture(scope='module')
def action_handler():
    act = ActionHandler((1, 0, 3))
    return act


def test_set_legal_actions(action_handler: ActionHandler):
    # test to make sure action raises error on matrix input
    with pytest.raises(AssertionError):
        action_handler.set_legal_actions([[0, 2, 4, 6]])

    action_handler.set_legal_actions([0, 2, 4, 6])
    assert action_handler.numActions == 4


def test_get_action(action_handler: ActionHandler):
    action_ind = action_handler.get_action([1, 0, 0, 0], random=False)
    assert isinstance(action_ind, np.integer), "expected int got {}".format(type(action_ind))
    assert action_ind == 0

    action_handler.get_action([1, 0, 0, 0])  # just make sure random doesn't fail


def test_game_action_to_action_ind(action_handler: ActionHandler):
    action_ind = action_handler.game_action_to_action_ind(2)
    assert isinstance(action_ind, np.integer), "expected int got {}".format(type(action_ind))
    assert action_ind == 1


def test_action_vect_to_game_action(action_handler: ActionHandler):
    game_action = action_handler.action_vect_to_game_action([0, 0, 1, 0], random=False)
    assert isinstance(game_action, np.integer), "expected int got {}".format(type(game_action))
    assert game_action == 4


def test_anneal(action_handler: ActionHandler):
    action_handler.anneal()
    action_handler.anneal()
    action_handler.anneal()
    assert action_handler.curr_rand_val == 0
    assert action_handler.curr_rand_val == action_handler.lowest_rand_val


def test_anneal_to(action_handler: ActionHandler):
    # zero should be highest rand val
    action_handler.anneal_to(0)
    assert action_handler.curr_rand_val == 1

    # one should be in the middle
    action_handler.anneal_to(1)
    assert action_handler.curr_rand_val == 0.5

    # 2 and greater should be lowest
    action_handler.anneal_to(2)
    assert action_handler.curr_rand_val == 0
    assert action_handler.curr_rand_val == action_handler.lowest_rand_val

    action_handler.anneal_to(999)
    assert action_handler.curr_rand_val == 0
    assert action_handler.curr_rand_val == action_handler.lowest_rand_val


def test_get_random(action_handler: ActionHandler):
    # reset curr rand val
    action_handler.curr_rand_val = 1

    # should be random action
    random, action = action_handler.get_random()
    assert random is True
    assert action in [0, 2, 4, 6]

    # test shouldn't be random
    action_handler.curr_rand_val = 0
    random, action = action_handler.get_random()
    assert random is False
    assert action is None


def test_rand_vals():
    # just test to make sure rand vals doesn't fail
    action_handler = ActionHandler((1, 0.1, 2), ActionPolicy.randVals, [0, 2, 4, 6])
    action_handler.get_action([0, 0, 0, 0])
