import pickle
import os


class EnvironmentRecorder:
    def __init__(self, wrapped_environment, save_dir):
        self.wrapped_environment = wrapped_environment
        self.clear_storage()
        self.save_dir = save_dir
        self.episode = 0
        os.makedirs(save_dir)

    def clear_storage(self):
        self.states = []
        self.extra_state_data = []
        self.actions = []
        self.rewards = []

    def __getattr__(self, attr):
        # if step function wrap it
        if attr == 'step':
            return self.record_step
        # else pass to environment
        else:
            return self.wrapped_environment.__getattribute__(attr)

    def record_step(self, action, *args, **kwargs):
        if hasattr(self.wrapped_environment, 'get_full_state'):
            state, extra_state_data = self.wrapped_environment.get_full_state()
            self.extra_state_data.append(extra_state_data)
        else:
            state = self.wrapped_environment.get_state()
        reward = self.wrapped_environment.step(action, *args, **kwargs)
        # we don't need to record state_tp1 as it will be recorded next time if not terminal
        terminal = self.wrapped_environment.get_terminal()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        # if terminal save current episode
        if terminal:
            save_obj = {
                "states": self.states,
                "extra_state_data": self.extra_state_data,
                "actions": self.actions,
                "rewards": self.rewards
            }
            with open(os.path.join(self.save_dir, 'episode{}.pkl'.format(self.episode)), 'wb') as out_file:
                pickle.dump(save_obj, out_file)

            self.clear_storage()
            self.episode += 1

        # need to return the reward because that's what the step function does
        return reward
