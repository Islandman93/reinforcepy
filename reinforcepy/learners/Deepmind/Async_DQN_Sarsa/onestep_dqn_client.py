import numpy as np
from reinforcepy.learners.base_async.PipeCmds import PipeCmds
from .BaseAsyncTargetLearner import BaseAsyncTargetLearner
from reinforcepy.environments.base_environment import BaseEnvironment


class Async1StepDQNLearner(BaseAsyncTargetLearner):
    def run_episode(self, environment: BaseEnvironment):
        # reset game
        environment.reset()
        self.loss_list = list()
        total_score = 0

        # get initial state
        state = np.asarray(environment.get_state() / 255.0, dtype=np.float32)

        # run until terminal
        terminal = False
        while not terminal and not self.done:
            # get action
            self.frame_buffer.add_state_to_buffer(state)
            action = self.get_game_action(self.frame_buffer.get_buffer())

            # step and get new state
            reward = 0
            for counter in range(self.skip_frame):
                env_reward = environment.step(action)
                reward += np.clip(env_reward, 0, 1)
            total_score += reward
            state_tp1 = np.asarray(environment.get_state() / 255.0, dtype=np.float32)

            # check for terminal
            terminal = environment.get_episode_over()

            # accumulate gradients
            loss = self.cnn.accumulate_gradients(self.frame_buffer.get_buffer(), self.action_handler.game_action_to_action_ind(action),
                                          reward, self.frame_buffer.get_buffer_with(state_tp1), terminal)
            self.loss_list.append(float(loss))

            state = state_tp1
            self.thread_steps += 1

            if self.thread_steps % self.async_update_step == 0 or terminal:
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

                # if terminal send stats
                if terminal:
                    stats = {'score': total_score, 'frames': self.thread_steps*self.skip_frame, 'loss': self.loss_list}
                    self.pipe.send((PipeCmds.ClientSendingStats, stats))
        print(self, 'ending episode. Step counter:', self.thread_steps,
              'Score:', total_score, 'Current Rand Val:', self.action_handler.curr_rand_val)