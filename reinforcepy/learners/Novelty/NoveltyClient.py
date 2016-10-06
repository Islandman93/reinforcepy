from reinforcepy.learners.Deepmind import BaseAsyncProcessTargetLearner
from reinforcepy.learners.Deepmind import AsyncNStepDQNLearner


class AsyncNStepNoveltyLearner(BaseAsyncProcessTargetLearner, AsyncNStepDQNLearner):
    def __init__(self, learner_parms, network_partial, pipe):
        super().__init__(learner_parms, network_partial, pipe)

        learner_parms.required(['novelty_dictionary'])
        self.novel_frames = learner_parms.get('novelty_dictionary')

    def update(self, state, action, reward, state_tp1, terminal):
        """
        Here we just inject the novel reward, then run the super NStepLearner update
        """
        reward += self.calc_novel_reward(state, state_tp1)
        super().update(state, action, reward, state_tp1, terminal)

    def calc_novel_reward(self, old_frame, new_frame):
        # novelty reward
        frame_hash = hash(new_frame.data.tobytes())

        # if already in table
        if frame_hash in self.novel_frames:
            novelty_reward = 0
            self.novel_frames[frame_hash] += 1
        # new state
        else:
            novelty_reward = 1
            self.novel_frames[frame_hash] = 1

        return novelty_reward
