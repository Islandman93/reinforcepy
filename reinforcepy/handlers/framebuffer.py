import numpy as np


class FrameBuffer:
    """
    The :class:`FrameBuffer` class takes care of storing a buffer of previous frames for the learner. It allows the
    learner to receive information about past states by using the channels of an image. So it can only be used with
    convolutional neural networks. Used by Deepmind methods.

    Parameters
    ----------
    shape : tuple(1, number of frames to store, environment width, environment height)
       Specifies the shape of the buffer. Most commonly the same as the input_shape to the neural network

    dtype : data type
       Default :class:`np.uint8`. The data type used for the buffer
    """
    def __init__(self, shape, dtype=np.uint8):
        self.shape = shape
        self.length = shape[1]
        self.frame_buffer = np.zeros(shape, dtype=dtype)
        self.dtype = dtype

    def add_state_to_buffer(self, state):
        """
        Adds a state to the current buffer. Works like a stack with a limited length, new states will push out old
        states

        Parameters
        ----------
        state : np.array(shape == (environment width, environment height))
            State from environment
        """
        # states 0-length-1 = 1:length (like a stack but pushing on top removes bottom)
        self.frame_buffer[0, 0:self.length - 1] = self.frame_buffer[0, 1:self.length]
        self.frame_buffer[0, self.length - 1] = state

    def get_buffer_with(self, state):
        """
        Gets the current buffer with the inputed state on the top. Works just like add_state_to_buffer but doesn't
        change the internal frame buffer

        Parameters
        ----------
        state : np.array(shape == (environment width, environment height))
            State from environment
        """
        # create a copy and fill it with current buffer plus state
        buffer_with = np.zeros(self.shape, dtype=self.dtype)
        buffer_with[0, 0:self.length - 1] = self.frame_buffer[0, 1:self.length]
        buffer_with[0, self.length - 1] = state
        return buffer_with

    def reset(self):
        """
        Resets the current buffer with zeros
        """
        self.frame_buffer = np.zeros(self.shape, dtype=self.dtype)

    def get_buffer(self):
        """
        Returns a copy of the current buffer
        """
        return np.copy(self.frame_buffer)

    def set_buffer(self, frame_buffer):
        """
        Sets the buffer
        """
        self.frame_buffer = frame_buffer
