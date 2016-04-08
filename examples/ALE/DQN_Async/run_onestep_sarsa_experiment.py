from reinforcepy.learners.base_async import AsyncLearnerHost, AsyncClientProcess
from reinforcepy.networks.Deepmind import AsyncTargetCNNSarsa
from reinforcepy.learners.Deepmind import Async1StepSarsaLearner
from reinforcepy.environments.ALE import ALEEnvironment
from functools import partial


def load_config():
    # load config from json
    from reinforcepy.handlers import Parameters

    # function to fix rom string to byte string
    def str2byte(string):
        return string.encode()

    rom_fix = {'experiment_parameters': {'rom': str2byte}}

    parameters = Parameters.fromJSON('onestep_sarsa_cfg.json', rom_fix)
    return [parameters['network_parameters'], parameters['training_parameters'], parameters['learner_parameters'],
            parameters['experiment_parameters']]


def main(experiment_parameters):
    experiment_parameters.required(['epochs', 'save_interval', 'learner_count', 'rom'])
    # load parameters
    network_parameters, training_parameters, learner_parameters, _ = load_config()

    # create initial cnn that will be used as the host
    cnn = AsyncTargetCNNSarsa(network_parameters, training_parameters)

    # create partials for client processes
    network_parameters.set('initial_values', cnn.get_parameters())
    network_parameters.required(['initial_values'])
    cnn_partial = partial(AsyncTargetCNNSarsa, network_parameters, training_parameters)

    # create list of learners and the environments to run them on
    learners = list()
    environments = list()
    for learner in range(experiment_parameters.get('learner_count')):
        learner_process = (partial(Async1StepSarsaLearner, learner_parameters, cnn_partial), AsyncClientProcess)
        learners.append(learner_process)
        environments.append(partial(ALEEnvironment, experiment_parameters.get('rom')))

    # create host
    host = AsyncLearnerHost(cnn, learners, environments)

    # run the thing and time it
    import time
    st = time.time()
    host.run(experiment_parameters.get('epochs'), save_interval=experiment_parameters.get('save_interval'),
             show_status=True)
    host.block_until_done()
    et = time.time()
    print('total time', et-st)
    return host

# python needs this to run processes
if __name__ == '__main__':
    _, _, _, experiment_parameters = load_config()
    main(experiment_parameters)
