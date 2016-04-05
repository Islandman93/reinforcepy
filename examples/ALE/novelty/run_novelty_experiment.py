from reinforcepy.learners.base_async import AsyncClientProcess
from reinforcepy.networks.Deepmind import AsyncTargetCNNNstep
from .NoveltyClient import AsyncNStepNoveltyLearner
from .NoveltyHost import NoveltyHost



def main():
    # setup vars
    rom = b'D:\\_code\\montezuma_revenge.bin'
    gamename = 'montezuma'
    num_actions = 8
    discount = 0.95
    learner_count = 8
    epochs = 15
    status_interval = 0.01

    from functools import partial

    cnn = AsyncTargetCNNNstep((None, 4, 84, 84), num_actions, 5)

    learners = list()
    for learner in range(learner_count):
        learner_process = (partial(AsyncNStepNoveltyLearner, num_actions, cnn.get_parameters(), discount=discount), AsyncClientProcess)
        learners.append(learner_process)

    host = NoveltyHost(cnn, learners, rom, show_rom=True)

    import time
    st = time.time()
    host.run(epochs, save_interval=status_interval, show_status=True)
    host.block_until_done()
    et = time.time()
    print('total time', et-st)
    return host

# python needs this to run processes
if __name__ == '__main__':
    try:
        host = main()
    except Exception as e:
        print(e)