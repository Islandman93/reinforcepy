import pytest
from reinforcepy.handlers import FrameBuffer
import numpy as np


@pytest.fixture(scope='module')
def framebuffer():
    fb = FrameBuffer((1, 3, 4, 4))
    return fb


def test_get_buffer_with(framebuffer: FrameBuffer):
    # create test state of ones
    state = np.ones((4, 4))
    buffer = framebuffer.get_buffer_with(state)

    assert np.all(buffer[0, 2] == state)
    assert np.all(buffer[0, 0:2] == np.zeros((2, 4, 4)))


def test_add_state_to_buffer(framebuffer: FrameBuffer):
    # add a test state of ones
    state1 = np.ones((4, 4))
    framebuffer.add_state_to_buffer(state1)

    assert np.all(framebuffer.frame_buffer[0, 2] == state1)
    assert np.all(framebuffer.frame_buffer[0, 0:2] == np.zeros((2, 4, 4)))

    # add again
    state2 = np.ones((4, 4))
    framebuffer.add_state_to_buffer(state2)

    assert np.all(framebuffer.frame_buffer[0, 1] == state1)
    assert np.all(framebuffer.frame_buffer[0, 2] == state2)
    assert np.all(framebuffer.frame_buffer[0, 0] == np.zeros((4, 4)))


def test_get_buffer(framebuffer: FrameBuffer):
    assert np.all(framebuffer.frame_buffer == framebuffer.get_buffer())


def test_set_buffer(framebuffer: FrameBuffer):
    buffer = np.ones((1, 3, 4, 4)) * 100
    framebuffer.set_buffer(buffer)
    assert np.all(buffer == framebuffer.frame_buffer)