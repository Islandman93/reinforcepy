reinforcePy
===========

reinforcePy is meant to be an easy to use collection of helpers, networks, and learners for reinforcement learning.
Right now the project is mainly focused on implementing papers from DeepMind and neural network based methods. There
have been a ton of new papers published by DeepMind but no combined place or package of implementations that is the main
purpose of this library.

That being said, I welcome pull requests for any reinforcement learning method if it follows the tenants of this library

Requirements & Installation
---------------------------

Standard scientific packages needed: numpy.

Neural net specific packages: `Theano <https://github.com/Theano/Theano>`_ and
`Lasagne <https://github.com/Lasagne/Lasagne>`_.

I recommend using `Anaconda <https://www.continuum.io/downloads>`_. Linux setup is easy (though untested):
.. code-block:: bash

    conda create -n name_your_environment python=3.4 numpy theano lasagne
    
    source activate name_your_environment
    
    
Windows Setup is a little harder. If you don't have theano installed check out my blog post at
http://www.islandman93.com/2016/04/tutorial-theano-install-on-windows-7-8.html.

After installing dependencies just use python setup.py install.

Documentation
-------------

Documentation is available online: http://reinforcepy.readthedocs.org/

For support, please email islandman93 at gmail or submit an issue/pull request.

Example DQN Usage
-----------------

.. code-block:: python

    from reinforcepy.environments.ALE import ALEEnvironment
    from reinforcepy.handlers import Parameters
    from reinforcepy.learners.Deepmind import DQNLearner
    from reinforcepy.networks.Deepmind import DQN_NIPS
    import time

    # setup vars
    rom = b'D:\\_code\\breakout.bin'
    epochs = 1
    ep_mod = 0.5
    epoch_def = 50000

    network_parameters = {
        'input_shape': (None, 4, 84, 84),
        'output_num': 4,
        'stride': (4, 2),
        'untie_biases': True
    }
    training_parameters = {
        'learning_rate': 0.0001,
        'minibatch_size': 32,
        'discount': 0.95
    }
    learner_parameters = {
        'skip_frame': 4,
        'anneal_egreedy_steps': 1000000,
        'dataset_shape': {'width': 84, 'height': 84},
        'max_dataset_size': 1000000,
        'phi_length': 4,
        'minimum_replay_size': 100,
        'minibatch_size': 32
    }
    network_parms = Parameters('Network Parameters', network_parameters)
    training_parms = Parameters('Training Parameters', training_parameters)
    learner_parms = Parameters('Learner Parameters', learner_parameters)

    # initialize environment and network/learner
    environment = ALEEnvironment(rom)
    network = DQN_NIPS(network_parms, training_parms)
    learner = DQNLearner(learner_parms, network)
    learner.set_legal_actions(environment.get_legal_actions())

    # main loop
    ep_count = 0
    while ep_count < epochs:
        reward = learner.run_episode(environment)
        if learner.step_count > epoch_def * ep_count:
            ep_count += ep_mod


For other examples see the `examples <examples/>`_ folder.

Development & Code Contribution
-------------------------------

This project is by no means finished and will constantly improve as I have time to work on it. I readily accept pull
requests, and will try to fix issues when they come up. All python code should use the PEP standard, and anything that
isn't PEP should be refactored (including my own code).

There are three things I believe are fundamental to this project:

1.  Premature optimization is the root of all evil. That being said this code needs to be fast so we don't have to wait
weeks for it to train. Try to be smart about where you put optimizations so that they don't obfuscate your code.

2.  Some of these algorithms can be very complex. Code must be commented/documented and be easily readable.

3.  On the same note as 2. Try to prevent 'spaghetti' code as much as possible. If a learner is composed of
10 different files it becomes impossible to read or to change just one thing as we so often do in research. Because of
this I try to keep almost all of the code for a learner in its own file in the run_episode function. This may cause some
code duplication but makes it easy to read and to change.


I'm still pretty new to github, docs, and python tests. I welcome refactoring, advice on folder structure and file
formats.

README lovingly edited from https://github.com/Lasagne/Lasagne without that project this one wouldn't be possible.
