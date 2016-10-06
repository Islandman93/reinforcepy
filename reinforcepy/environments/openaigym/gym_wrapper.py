from ..base_environment import BaseEnvironment


class GymWrapper(BaseEnvironment):
    def __init__(self, env):
        self.env = env
        self.current_state = env.reset()
        self.terminal = False

    def get_state(self):
        return self.current_state

    def reset(self):
        self.current_state = self.env.reset()
        self.terminal = False

    def get_terminal(self):
        return self.terminal

    def step(self, action):
        observation, reward, terminal, _ = self.env.step(action)
        self.current_state = observation
        self.terminal = terminal
        return reward

    def get_input_space(self):
        return self.env.observation_space.n

    def get_legal_actions(self):
        return [i for i in range(self.env.action_space.n)]

    def render(self):
        return self.env.render()
