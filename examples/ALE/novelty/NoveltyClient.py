from reinforcepy.learners.base_async import PipeCmds
from reinforcepy.learners.Deepmind import BaseAsyncTargetLearner
from reinforcepy.environments import BaseEnvironment
import numpy as np


class AsyncNStepNoveltyLearner(BaseAsyncTargetLearner):
    def __init__(self, learner_parms, network_partial, pipe):
        super().__init__(learner_parms, network_partial, pipe)

        learner_parms.required(['discount', 'novelty_dictionary'])

        self.discount = learner_parms.get('discount')
        self.novel_frames = learner_parms.get('novelty_dictionary')

    def run_episode(self, emulator: BaseEnvironment):
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
        state = np.asarray(emulator.get_state()/255.0, dtype=np.float32)

        while not terminal and not self.done:
            # get action
            train_states[self.thread_steps-t_start] = self.frame_buffer_with(state)
            self.add_state_to_buffer(state)
            action = self.get_game_action(self.frame_buffer)

            # step and get new state
            reward = emulator.step(action, clip=1)
            state_tp1 = np.asarray(emulator.get_state()/255.0, dtype=np.float32)
            novelty_plus_reward = reward + self.calc_novel_reward(state, state_tp1)

            # check for terminal
            terminal = emulator.get_episode_over()

            # update nstep vars
            rewards.append(novelty_plus_reward)
            train_actions.append(self.action_handler.game_action_to_action_ind(action))

            # update loop vars
            state = state_tp1
            self.thread_steps += 1
            total_score += reward

            # if update or terminal do a gradient update
            if (self.thread_steps - t_start) % self.async_update_step == 0 or terminal:
                # calculate training rewards
                train_rewards = np.zeros(self.thread_steps - t_start, dtype=np.float32)
                train_reward = 0 if terminal else np.max(self.cnn.get_target_output(self.frame_buffer_with(state_tp1)))
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
                new_params, global_vars = self.synchronous_update(self.cnn.get_gradients(), self.thread_steps*self.skip_frame, self.novel_frames)
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

    def calc_novel_reward(self, old_frame, new_frame):
        # novelty reward
        frame_hash = hash(new_frame.data.tobytes())

        # if already in table
        if frame_hash in self.novel_frames:
            novelty_reward = 0
            self.novel_frames[frame_hash] += 1
        # new state
        else:
            novelty_reward = abs(np.sum(old_frame-new_frame))/10
            self.novel_frames[frame_hash] = 1

        return novelty_reward
