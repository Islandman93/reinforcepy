import pygame
import numpy as np


def generate_keymapping(legal_ale_actions=None):
    if legal_ale_actions is None:
        # there are 18 ale actions
        legal_ale_actions = np.arange(18)

    def key_mapping(pressed_keys, mouse_rel=None):
        # default to no-op
        action = 0
        # fire
        if pressed_keys[pygame.K_SPACE]:
            action = 1
        # only up
        elif np.sum(pressed_keys) == 2 and pressed_keys[pygame.K_w]:
            action = 2
        # only right
        elif np.sum(pressed_keys) == 2 and pressed_keys[pygame.K_d]:
            action = 3
        # only left
        elif np.sum(pressed_keys) == 2 and pressed_keys[pygame.K_a]:
            action = 4
        # only down
        elif np.sum(pressed_keys) == 2 and pressed_keys[pygame.K_s]:
            action = 5
        # up right
        elif pressed_keys[pygame.K_w] and pressed_keys[pygame.K_d]:
            action = 6
        # up left
        elif pressed_keys[pygame.K_w] and pressed_keys[pygame.K_a]:
            action = 7
        # down right
        elif pressed_keys[pygame.K_s] and pressed_keys[pygame.K_d]:
            action = 8
        # down left
        elif pressed_keys[pygame.K_s] and pressed_keys[pygame.K_a]:
            action = 9
        # up fire
        elif pressed_keys[pygame.K_w] and pressed_keys[pygame.K_SPACE]:
            action = 10
        # right fire
        elif pressed_keys[pygame.K_d] and pressed_keys[pygame.K_SPACE]:
            action = 11
        # left fire
        elif pressed_keys[pygame.K_a] and pressed_keys[pygame.K_SPACE]:
            action = 12
        # down fire
        elif pressed_keys[pygame.K_s] and pressed_keys[pygame.K_SPACE]:
            action = 13
        # up right fire
        elif pressed_keys[pygame.K_w] and pressed_keys[pygame.K_d] and pressed_keys[pygame.K_SPACE]:
            action = 14
        # up left fire
        elif pressed_keys[pygame.K_w] and pressed_keys[pygame.K_a] and pressed_keys[pygame.K_SPACE]:
            action = 15
        # down right fire
        elif pressed_keys[pygame.K_s] and pressed_keys[pygame.K_d] and pressed_keys[pygame.K_SPACE]:
            action = 16
        # down left fire
        elif pressed_keys[pygame.K_s] and pressed_keys[pygame.K_a] and pressed_keys[pygame.K_SPACE]:
            action = 17
        else:
            action = 0

        if action not in legal_ale_actions:
            print("You didn't press a legal action for this game. Executing no-op")
            action = 0

        # return the index of the ale action in the legal actions array
        return np.where(action == legal_ale_actions)[0][0]
    # return the generated function
    return key_mapping


def print_key_mapping():
    print('Keys are mapped to WASD for movement, and space for fire. Press any non mapped key to execute a no-op action',
          'Press multiple keys together to execute their repsective actions (like up-left or down-right-fire)')

