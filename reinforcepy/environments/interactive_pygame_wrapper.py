import pygame
from pygame import surfarray
import numpy as np
from .base_environment import BaseEnvironment


class PygameWrapper(BaseEnvironment):
    def __init__(self, environment, keys_to_actions, use_mouse=False, max_fps=60, resize_screen=1.0):
        self.resize_screen = resize_screen
        self.environment = environment
        self.convert_keys_to_actions = keys_to_actions
        self.use_mouse = use_mouse
        self.max_fps = max_fps

        pygame.init()
        self.pg_screen = pygame.display.set_mode(self.get_state_shape()[0:2])
        pygame.display.flip()

        self.pg_clock = pygame.time.Clock()

    def run_keyboard_episode(self, num_episodes=1):
        episode = 0
        reward = 0
        self.reset()
        try:
            while episode < num_episodes:
                # display screen
                self.display_new_screen()

                # get keyboard action
                # lock processing until a key is pressed
                pressed_keys = [0]
                while sum(pressed_keys) <= 1:
                    # pump must go before get_pressed https://www.pygame.org/docs/ref/key.html#comment_pygame_key_get_pressed
                    pygame.event.pump()
                    pressed_keys = pygame.key.get_pressed()
                    # CTRL+C
                    if (pressed_keys[pygame.K_LCTRL] or pressed_keys[pygame.K_RCTRL]) and pressed_keys[pygame.K_c]:
                        raise KeyboardInterrupt
                    # set frame cap
                    self.pg_clock.tick(self.max_fps)
                reward += self.step_keyboard(pressed_keys)

                # if terminal reset and new episode
                if self.get_terminal():
                    self.print_end_of_game_status(reward)
                    self.reset()
                    episode += 1
                    reward = 0
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
            self.environment.close()
            pygame.quit()

    def print_end_of_game_status(self, reward):
        print('Episode completed, reward: {}'.format(reward))

    def step_keyboard(self, pressed_keys):
        action = self.convert_keys_to_actions(pressed_keys)
        return self.step(action)

    def step(self, action):
        if self.use_mouse:
            x_delta, y_delta = pygame.mouse.get_rel()
            # by default get_rel is inverted
            y_delta *= -1
            return self.environment.step((action, x_delta))
        else:
            return self.environment.step(action)

    def display_new_screen(self):
        self.pg_screen.fill((0, 0, 0))
        gamescreen = self.environment.get_state()

        # if grayscale convert to duplicate color channels
        if len(gamescreen.shape) == 2:
            gamescreen = np.tile(gamescreen[:, :, np.newaxis], 3)
            # swap height width
            gamescreen = np.swapaxes(gamescreen, 0, 1)
        # if rgb channels on first dim move to last
        if gamescreen.shape[0] == 3:
            # this also swaps height width
            gamescreen = np.swapaxes(gamescreen, 0, 2)

        frames = surfarray.make_surface(gamescreen)

        # resize screen
        if self.resize_screen != 1:
            new_width = int(gamescreen.shape[0] * self.resize_screen)
            new_height = int(gamescreen.shape[1] * self.resize_screen)
            frames = pygame.transform.scale(frames, (new_width, new_height))

        self.pg_screen.blit(frames, (0, 0))
        pygame.display.flip()

        # TODO: display additional text game vars

    def reset(self):
        self.environment.reset()

    def get_state(self):
        return self.environment.get_state()

    def get_state_shape(self):
        state_shape = self.environment.get_state_shape()
        new_shape = []
        # if color first dim
        if len(state_shape) > 2 and state_shape[0] == 3:
            # swap height width and put color at end
            new_shape = [state_shape[2], state_shape[1], 3]
        # else swap height width
        else:
            new_shape = [state_shape[1], state_shape[0], 3]
        new_shape[0] = int(new_shape[0] * self.resize_screen)
        new_shape[1] = int(new_shape[1] * self.resize_screen)
        return new_shape

    def get_terminal(self):
        return self.environment.get_terminal()

    def get_num_actions(self):
        return self.environment.get_num_actions()

    def get_step_count(self):
        return self.environment.get_step_count()
