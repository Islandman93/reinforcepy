import numpy as np
from reinforcepy.learners.base_async.PipeCmds import PipeCmds
from .BaseAsyncTargetLearner import BaseAsyncTargetLearner
from reinforcepy.environments.base_environment import BaseEnvironment


class AsyncNStepDQNLearner(BaseAsyncTargetLearner):
    def __init__(self, learner_parms, network_partial, pipe):
        super().__init__(learner_parms, network_partial, pipe)

        learner_parms.required(['discount'])
        self.discount = learner_parms.get('discount')

    def run_episode(self, environment: BaseEnvironment):
        # reset game
        environment.reset()

        # loop vars
        self.loss_list = list()
        total_score = 0
        terminal = False

        # nstep vars, rewards and states are stored in a list of length self.async_update_step
        t_start = self.thread_steps
        train_states = np.zeros((self.async_update_step, self.phi_length, 84, 84), dtype=np.float32)
        train_actions = list()
        rewards = list()

        # get initial state
        state = np.asarray(environment.get_state() / 255.0, dtype=np.float32)

        while not terminal and not self.done:
            # get action
            train_states[self.thread_steps-t_start] = self.frame_buffer.get_buffer_with(state)
            self.frame_buffer.add_state_to_buffer(state)
            action = self.get_game_action(self.frame_buffer.get_buffer())

            # step and get new state
            reward = 0
            for frame in range(self.skip_frame):
                env_reward = environment.step(action)
                reward += np.clip(env_reward, 0, 1)

            state_tp1 = np.asarray(environment.get_state() / 255.0, dtype=np.float32)

            # check for terminal
            terminal = environment.get_episode_over()

            # update nstep vars
            rewards.append(reward)
            train_actions.append(self.action_handler.game_action_to_action_ind(action))

            # update loop vars
            state = state_tp1
            self.thread_steps += 1
            total_score += reward

            # if update or terminal do a gradient update
            if (self.thread_steps - t_start) % self.async_update_step == 0 or terminal:
                # calculate training rewards
                train_rewards = np.zeros(self.thread_steps - t_start, dtype=np.float32)
                train_reward = 0 if terminal else np.max(self.cnn.get_target_output(self.frame_buffer.get_buffer_with(state_tp1)))
                for i in range((self.thread_steps - t_start)-1, -1, -1):  # for a reversed range we have to subtract 1
                    train_reward = rewards[i] + self.discount*train_reward
                    train_rewards[i] = train_reward

                # if terminal cuts off the async step then just take the states that correspond to how many steps we did
                if terminal:
                    train_states = train_states[0:self.thread_steps-t_start]

                # calculate gradients
                loss = self.cnn.accumulate_gradients(train_states, train_actions, train_rewards)
                self.loss_list.append(float(loss))

                # process cmds from host, this will flush pipe recv commands
                self.process_host_cmds()

                # synchronous update parameters, this sends then waits for host to send back
                new_params, global_vars = self.synchronous_update(self.cnn.get_gradients(), self.thread_steps*self.skip_frame)
                self.cnn.clear_gradients()
                self.cnn.set_parameters(new_params)

                self.action_handler.anneal_to(global_vars['counter'])
                if self.check_update_target(global_vars['counter']):
                    print(self, 'setting target')
                    self.cnn.set_target_parameters(new_params)

                # re-initialize nstep vars
                t_start = self.thread_steps
                train_states = np.zeros((self.async_update_step, self.phi_length, 84, 84), dtype=np.float32)
                train_actions = list()
                rewards = list()

                # if terminal send stats
                if terminal:
                    stats = {'score': total_score, 'frames': self.thread_steps*self.skip_frame, 'loss': self.loss_list}
                    self.pipe.send((PipeCmds.ClientSendingStats, stats))
        print(self, 'ending episode. Step counter:', self.thread_steps,
              'Score:', total_score, 'Current Rand Val:', self.action_handler.curr_rand_val)