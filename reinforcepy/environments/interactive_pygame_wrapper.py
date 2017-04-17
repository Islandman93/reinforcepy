import pygame
from pygame import surfarray
import numpy as np
from .base_environment import BaseEnvironment


class PygameWrapper(BaseEnvironment):
    def __init__(self, environment, keys_to_actions, use_mouse=False):
        self.environment = environment
        self.state_shape = environment.get_state_shape()
        self.convert_keys_to_actions = keys_to_actions
        self.use_mouse = use_mouse

        pygame.init()
        self.pg_screen = pygame.display.set_mode(self.state_shape[::-1])
        pygame.display.flip()

        self.pg_clock = pygame.time.Clock()

    def run_keyboard_episode(self, num_episodes=1):
        episode = 0
        reward = 0
        self.reset()
        while episode < num_episodes:
            # display screen
            self.display_new_screen()

            # get keyboard action
            # lock processing until a key is pressed
            pressed_keys = [0]
            while sum(pressed_keys) <= 1:
                pressed_keys = pygame.key.get_pressed()
                # push through other events to pygame
                pygame.event.pump()
            reward += self.step_keyboard(pressed_keys)

            # set frame cap
            self.pg_clock.tick(35.)

            # if terminal reset and new episode
            if self.get_terminal():
                self.print_end_of_game_status(reward)
                self.reset()
                episode += 1
                reward = 0

    def print_end_of_game_status(self, reward):
        print('Episode completed, reward: {}'.format(reward))

    def step_keyboard(self, pressed_keys):
        action = self.convert_keys_to_actions(pressed_keys)
        return self.step(action)

    def step(self, action):
        self.display_new_screen()
        if self.use_mouse:
            x_delta, y_delta = pygame.mouse.get_rel()
            # by default get_rel is inverted
            y_delta *= -1
            return self.environment.step(action, (x_delta, y_delta))
        else:
            return self.environment.step(action)

    def display_new_screen(self):
        self.pg_screen.fill((0, 0, 0))
        gamescreen = self.environment.get_state()

        # if grayscale convert to duplicate color channels
        if len(gamescreen.shape) == 2:
            gamescreen = np.tile(gamescreen[:, :, np.newaxis], 3)
        frames = surfarray.make_surface(np.swapaxes(gamescreen, 0, 1))
        self.pg_screen.blit(frames, (0, 0))
        pygame.display.flip()

        # TODO: display additional text game vars

    def reset(self):
        self.environment.reset()

    def get_state(self):
        return self.environment.get_state()

    def get_state_shape(self):
        return self.environment.get_state_shape

    def get_terminal(self):
        return self.environment.get_terminal()

    def get_num_actions(self):
        return self.environment.get_num_actions()

    def get_step_count(self):
        return self.environment.get_step_count()
