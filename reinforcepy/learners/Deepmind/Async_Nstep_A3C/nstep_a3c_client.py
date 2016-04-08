import numpy as np
from AsyncA3CLearner import AsyncA3CLearner
from learningALE.handlers.async.PipeCmds import PipeCmds
from learningALE.learners.async_a3c_cnn import AsyncA3CCNN
from functools import partial


class AsyncNStepA3CLearner(AsyncA3CLearner):
    def __init__(self, num_actions, initial_cnn_values, pipe,
                 skip_frame=4, phi_length=4, async_update_step=5, discount=0.95):
        cnn_partial = partial(AsyncA3CCNN, (None, phi_length, 84, 84), num_actions, async_update_step)

        super().__init__(num_actions, initial_cnn_values, cnn_partial, pipe, skip_frame=skip_frame,
                         phi_length=phi_length, async_update_step=async_update_step)

        self.discount = discount

    def run_episode(self, emulator):
        # reset game
        emulator.reset()

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
        state = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

        while not terminal and not self.done:
            # get action
            train_states[self.thread_steps-t_start] = self.frame_buffer_with(state)
            self.add_state_to_buffer(state)
            action = self.get_game_action(self.frame_buffer)

            # step and get new state
            reward = emulator.step(action, clip=1)
            state_tp1 = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

            # check for terminal
            terminal = emulator.get_terminal()

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
                train_reward = 0 if terminal else self.cnn.get_value_output(self.frame_buffer_with(state_tp1))[0]
                for i in range((self.thread_steps - t_start)-1, -1, -1):  # for a reversed range we have to subtract 1
                    train_reward = rewards[i] + self.discount*train_reward
                    train_rewards[i] = train_reward

                # if terminal cuts off the async step then just take the states that correspond to how many steps we did
                if terminal:
                    train_states = train_states[0:self.thread_steps-t_start]

                # calculate gradients
                loss = self.cnn.accumulate_gradients(train_states, train_actions, train_rewards)
                self.loss_list.append(loss)

                # process cmds from host, this will flush pipe recv commands
                self.process_host_cmds()

                # synchronous update parameters, this sends then waits for host to send back
                new_params, global_vars = self.synchronous_update(self.cnn.get_gradients(), self.thread_steps*self.skip_frame)
                self.cnn.clear_gradients()
                self.cnn.set_parameters(new_params)

                self.update_global_vars(global_vars)

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

    def run_no_pipe(self, emulator):
        # reset game
        emulator.reset()
        self.loss_list = list()

        # loop vars
        total_score = 0
        terminal = False

        # nstep vars, rewards and states are stored in a list of length self.async_update_step
        t_start = self.thread_steps
        train_states = np.zeros((self.async_update_step, self.phi_length, 84, 84), dtype=np.float32)
        train_actions = list()
        rewards = list()

        # get initial state
        state = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

        while not terminal and not self.done:
            # get action
            train_states[self.thread_steps-t_start] = self.frame_buffer_with(state)
            self.add_state_to_buffer(state)
            action = self.get_game_action(self.frame_buffer)

            # step and get new state
            reward = emulator.step(action, clip=1)
            state_tp1 = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

            # check for terminal
            terminal = emulator.get_terminal()

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
                train_reward = 0 if terminal else self.cnn.get_value_output(self.frame_buffer_with(state_tp1))[0]
                for i in range((self.thread_steps - t_start)-1, -1, -1):  # for a reversed range we have to subtract 1
                    train_reward = rewards[i] + self.discount*train_reward
                    train_rewards[i] = train_reward

                # if terminal cuts off the async step then just take the states that correspond to how many steps we did
                if terminal:
                    train_states = train_states[0:self.thread_steps-t_start]

                # calculate gradients
                loss = self.cnn.accumulate_gradients(train_states, train_actions, train_rewards)
                self.loss_list.append(loss)

                # gradient step and clear grads
                self.cnn.gradient_step(self.cnn.get_gradients())
                self.cnn.clear_gradients()

                self.update_global_vars({"counter": self.thread_steps * self.skip_frame})

                # re-initialize nstep vars
                t_start = self.thread_steps
                train_states = np.zeros((self.async_update_step, self.phi_length, 84, 84), dtype=np.float32)
                train_actions = list()
                rewards = list()
        print(self, 'Stopping')