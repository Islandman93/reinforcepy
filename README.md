# ReinforcePy

ReinforcePy is meant to be an easy to use collection of helpers, networks, and learners for reinforcement learning. Right now the project is mainly focused on implementing papers from DeepMind and neural network based methods. There have been a ton of new papers published about deep reinforcement learning but no combined place or package of implementations, that is the main purpose of this library.

### Current Status & Usage
Currently supported methods are:
- [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) kinda old code, replaced by async paper.
- Async 1 step DQN & SARSA, N-step, A3C, Recurrent support is a WIP. [Paper](https://arxiv.org/abs/1602.01783)
- Async [Double Q-Learning](https://arxiv.org/abs/1509.06461), Double N-step [no paper]
- Async with Experience Replay, supports all Async methods but A3C

Upcoming features can be found in the wiki roadmap.

Example usage, trained models, and results can be found under
[examples/ALE/](examples/ALE).
A more in depth look at implementation details can be found in the wiki.

## Installation
If you don't already I recommend using [Anaconda](https://www.continuum.io/downloads) to manage python environments, it also makes installation of Numpy & Scipy  a breeze. Required packages:

- [NumPy](http://www.scipy.org/scipylib/download.html) (conda install numpy)
- [SciPy](https://www.scipy.org/install.html) (conda install scipy)
- [Pillow](https://python-pillow.org/) (conda install pillow)
- [TensorFlow](https://www.tensorflow.org/) >= 1.0
- [TFLearn](https://github.com/tflearn/tflearn) >= 0.3 (git clone then python setup.py install)
- [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment#quick-start)
- OpenAI Gym support is coming soon

Then simply:
```sh
    git clone https://github.com/Islandman93/reinforcepy
    cd reinforcepy/
    python setup.py install
```
### Windows support
NEW: TensorFlow supports windows, the ALE uses cmake but I was unable to get it working with windows. A Visual Studio port can be found [here](https://github.com/Islandman93/Arcade-Learning-Environment)

## Documentation
Documentation is a work in progress available at: [http://reinforcepy.readthedocs.org/](http://reinforcepy.readthedocs.org/).

For support, please submit an issue.

## Development
All pull requests are welcome, this project is not tied to any specific reinforcement learning method so feel free to submit any published method or environment.

To hack on the code simply use:
```sh
    python setup.py develop
```
