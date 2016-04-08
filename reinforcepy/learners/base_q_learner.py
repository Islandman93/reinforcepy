from reinforcepy.learners import BaseLearner


class BaseQLearner(BaseLearner):
    def reset(self):
        raise NotImplementedError("Implementations of base q learner must implement reset")

    def get_action(self, state):
        raise NotImplementedError("Implementations of base q learner must implement get_action(state)")

    def step(self, environment_step_fn, action):
        raise NotImplementedError(
            "Implementations of base q learner must implement step(environment_step_fn, action)")

    def update(self, state, action, reward, state_tp1, terminal):
        raise NotImplementedError("Implementations of base q learner must implement update")

    def episode_end(self):
        raise NotImplementedError("Implementations of base q learner must implement episode_end")
    
    def run_episode(self, environment):
        # reset environment and self
        environment.reset()
        self.reset()

        # get initial state pair
        state = environment.get_state()

        # run until environment terminal
        terminal = False
        while not terminal:
            # get action for state
            action = self.get_action(state)

            # step through environment
            reward = self.step(environment.step, action)

            # get new state
            state_tp1 = environment.get_state()

            # check if I'm now terminal
            terminal = environment.get_terminal()

            # update learner
            self.update(state, action, reward, state_tp1, terminal)

            state = state_tp1

        self.episode_end()