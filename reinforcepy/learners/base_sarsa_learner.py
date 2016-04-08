from reinforcepy.learners import BaseLearner


class BaseSarsaLearner(BaseLearner):
    def reset(self):
        raise NotImplementedError("Implementations of base sarsa learner must implement reset")

    def get_action(self, state):
        raise NotImplementedError("Implementations of base sarsa learner must implement get_action(state)")

    def step(self, environment_step_fn, action):
        raise NotImplementedError("Implementations of base sarsa learner must implement step(environment_step_fn, action)")

    def update(self, state, action, reward, state_tp1, action_tp1, terminal):
        raise NotImplementedError("Implementations of base sarsa learner must implement update")

    def episode_end(self):
        raise NotImplementedError("Implementations of base sarsa learner must implement episode_end")

    def run_episode(self, environment):
        # reset environment and self
        environment.reset()
        self.reset()

        # get initial state action pair
        state = environment.get_state()
        action = self.get_action(state)

        # run until environment terminal
        terminal = False
        while not terminal:
            # step through environment
            reward = self.step(environment.step, action)

            # get new state
            state_tp1 = environment.get_state()
            # get new action
            action_tp1 = self.get_action(state_tp1)

            # check if I'm now terminal
            terminal = environment.get_terminal()

            # update learner
            self.update(state, action, reward, state_tp1, action_tp1, terminal)

            state = state_tp1
            action = action_tp1

        self.episode_end()