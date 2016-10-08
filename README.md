# reinforcePy

reinforcePy is meant to be an easy to use collection of helpers, networks, and learners for reinforcement learning. Right now the project is mainly focused on implementing papers from DeepMind and neural network based methods. There have been a ton of new papers published about deep reinforcement learning but no combined place or package of implementations, that is the main purpose of this library.

### Current Status & Usage
Currently supported methods are DQN, Async 1 step DQN & SARSA, and N-step DQN. A3C support is coming soon. 

Example usage, trained models, and results can be found under /examples/ALE(https://github.com/Islandman93/reinforcepy/tree/master/examples/ALE).
A more in depth look at imlementation details can be found in the wiki.

## Installation
Required packages: 

- NumPy (pip install or conda install numpy)
- SciPy (pip install or conda install scipy)
- Pillow (pip install or conda install pillow)
- Tensorflow[link]
- TFLearn[link]
- Arcade Learning Environment[link]
- OpenAI Gym support is coming soon

Then simply:
```sh
    python setup.py install
```
### Windows support
This project relies on TensorFlow which does not currently support Windows. When it does there should be no issue using this project in a Windows environment. 

## Documentation
Documentation is a work in progress available at: http://reinforcepy.readthedocs.org/
For support, please submit an issue. 

## Development
All pull requests are welcome, this project is not tied to any specific reinforcement learning method so feel free to submit any published method.

To hack on the code simply use: 
```sh
    python setup.py develop
```
